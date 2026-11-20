"""megatron-core's MTP roll gives the same values and gradients with the all-gather CP exchange as with its P2P one
(packed and unpacked, gloo, CP=4), and when the patch installs that exchange."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

mtp_module = pytest.importorskip(
    "megatron.core.transformer.multi_token_prediction", reason="requires the megatron extra"
)
if not hasattr(mtp_module, "_exchange_cp_roll_boundaries"):
    pytest.skip("megatron-core predates _exchange_cp_roll_boundaries", allow_module_level=True)

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

CP = 4
CU_SEQLENS = [0, 16, 24, 48]  # packed documents of 16, 8 and 24 tokens (multiples of 2 * CP): 12 tokens per rank


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _no_p2p(ops):
    raise AssertionError("P2P on the CP group while the all-gather exchange is installed")


def _roll(mtp, x, dim, group, params):
    """megatron-core's roll with the installed exchange: the rolled values and the gradient of their sum."""
    x = x.clone().requires_grad_(x.is_floating_point())
    y, y_sum = mtp.roll_tensor(x, shifts=-1, dims=dim, cp_group=group, packed_seq_params=params)
    return y.detach(), torch.autograd.grad(y_sum, x)[0] if x.requires_grad else None


def _worker(rank, port, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=CP)
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import multi_token_prediction as mtp

    from skyrl.backends.skyrl_train.patches.megatron.patch_mtp_roll_allgather import (
        exchange_cp_roll_boundaries_allgather,
    )

    group = dist.new_group(list(range(CP)))
    cu = torch.tensor(CU_SEQLENS, dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    p2p = (mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv)
    gen, n = torch.Generator().manual_seed(rank), CU_SEQLENS[-1] // CP  # distinct per rank: a wrong neighbour shows
    errors = []
    # A float (with gradients) rolled on its first dim, and a bool (gathered as uint8) on its last.
    for x, dim in ((torch.randn(n, 1, 6, generator=gen), 0), (torch.rand(1, n, generator=gen) < 0.5, -1)):
        for params in (packed, None):  # None: a single unpacked sequence, roll_tensor's own CP branch
            mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv = exchange_cp_roll_boundaries_allgather, _no_p2p
            y, grad = _roll(mtp, x, dim, group, params)
            mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv = p2p
            p2p_y, p2p_grad = _roll(mtp, x, dim, group, params)
            case = f"{x.dtype}, {'packed' if params else 'unpacked'}"
            if not torch.equal(y, p2p_y):
                errors.append(f"{case}: rolled values differ from the P2P exchange's")
            if grad is not None and not torch.equal(grad, p2p_grad):
                errors.append(f"{case}: gradients differ from the P2P exchange's")
    failures[rank] = errors
    dist.destroy_process_group()


def test_allgather_exchange_matches_p2p():
    with mp.Manager() as m:
        failures = m.dict()
        mp.spawn(_worker, args=(_free_port(), failures), nprocs=CP, join=True)
        assert dict(failures) == {rank: [] for rank in range(CP)}


@pytest.fixture
def patch(monkeypatch):
    """The patch module, not yet applied; megatron-core's exchange is restored afterwards."""
    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_mtp_roll_allgather as patch,
    )

    monkeypatch.delenv("SKYRL_MTP_ROLL_ALLGATHER", raising=False)
    monkeypatch.setattr(patch, "_APPLIED", False)
    monkeypatch.setattr(patch, "_ORIGINAL", None)
    monkeypatch.setattr(mtp_module, "_exchange_cp_roll_boundaries", mtp_module._exchange_cp_roll_boundaries)
    return patch


@pytest.mark.parametrize(
    ("env", "mtp_enabled", "cp", "installed"),
    [
        (None, True, 4, True),
        (None, True, 1, False),
        (None, False, 4, False),
        ("0", True, 4, False),
        ("1", False, 1, True),
    ],
)
def test_installed_once_for_mtp_with_cp_unless_overridden(patch, monkeypatch, env, mtp_enabled, cp, installed):
    if env is not None:
        monkeypatch.setenv("SKYRL_MTP_ROLL_ALLGATHER", env)
    original = mtp_module._exchange_cp_roll_boundaries
    expected = patch.exchange_cp_roll_boundaries_allgather if installed else original
    assert patch.apply_mtp_roll_allgather_patch(mtp_enabled=mtp_enabled, context_parallel_size=cp) is installed
    assert mtp_module._exchange_cp_roll_boundaries is expected
    # A second call installs nothing.
    assert not patch.apply_mtp_roll_allgather_patch(mtp_enabled=mtp_enabled, context_parallel_size=cp)
    assert mtp_module._exchange_cp_roll_boundaries is expected


def test_no_op_with_a_warning_without_the_exchange(patch, monkeypatch):
    from loguru import logger

    monkeypatch.delattr(mtp_module, "_exchange_cp_roll_boundaries")  # megatron-core before NVIDIA/Megatron-LM#8389
    messages = []
    handle = logger.add(messages.append, level="WARNING")
    try:
        assert not patch.apply_mtp_roll_allgather_patch(mtp_enabled=True, context_parallel_size=4)
    finally:
        logger.remove(handle)
    assert not hasattr(mtp_module, "_exchange_cp_roll_boundaries")
    assert any("_exchange_cp_roll_boundaries" in m for m in messages)
