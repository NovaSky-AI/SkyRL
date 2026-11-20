"""megatron-core's MTP roll on NCCL: the all-gather CP exchange gives the same values and gradients as megatron-core's
P2P one, at CP = 2, 4 and 8 on one node (packed and unpacked).

The CPU test (``tests/backends/skyrl_train/patches/megatron/test_mtp_roll_allgather.py``) runs on gloo; this covers the
NCCL collective the patch actually uses in training, including the bool -> uint8 cast gloo cannot show. It does not
reproduce the cross-node P2P hang the patch avoids, which needs a CP group spanning nodes.

Run (needs 8 GPUs on one node)::

    uv run --isolated --extra dev --extra megatron pytest -s \
        tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_mtp_roll_allgather_nccl.py
"""

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

pytestmark = pytest.mark.megatron


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


def _worker(rank, cp, port, failures):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=cp, device_id=torch.device("cuda", rank))
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer import multi_token_prediction as mtp

    from skyrl.backends.skyrl_train.patches.megatron.patch_mtp_roll_allgather import (
        exchange_cp_roll_boundaries_allgather,
    )

    device = torch.device("cuda", rank)
    group = dist.new_group(list(range(cp)))
    # Packed documents whose lengths are multiples of 2 * cp (the zigzag split), uneven so a wrong segment shows.
    lengths = [2 * cp * k for k in (3, 1, 5, 2)]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    n = int(cu[-1]) // cp
    p2p = (mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv)
    gen = torch.Generator(device=device).manual_seed(rank)  # distinct per rank: a wrong neighbour shows
    errors = []
    # A float (with gradients) rolled on its first dim, and a bool (gathered as uint8) on its last.
    inputs = (
        (torch.randn(n, 1, 6, generator=gen, device=device), 0),
        (torch.rand(1, n, generator=gen, device=device) < 0.5, -1),
    )
    for x, dim in inputs:
        for params in (packed, None):  # None: a single unpacked sequence, roll_tensor's own CP branch
            mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv = exchange_cp_roll_boundaries_allgather, _no_p2p
            y, grad = _roll(mtp, x, dim, group, params)
            mtp._exchange_cp_roll_boundaries, dist.batch_isend_irecv = p2p
            p2p_y, p2p_grad = _roll(mtp, x, dim, group, params)
            case = f"cp={cp}, {x.dtype}, {'packed' if params else 'unpacked'}"
            if not torch.equal(y, p2p_y):
                errors.append(f"{case}: rolled values differ from the P2P exchange's")
            if grad is not None and not torch.equal(grad, p2p_grad):
                errors.append(f"{case}: gradients differ from the P2P exchange's")
    torch.cuda.synchronize()
    failures[rank] = errors
    dist.destroy_process_group()


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_allgather_exchange_matches_p2p_on_nccl(cp):
    if torch.cuda.device_count() < cp:
        pytest.skip(f"needs {cp} GPUs")
    with mp.Manager() as m:
        failures = m.dict()
        mp.spawn(_worker, args=(cp, _free_port(), failures), nprocs=cp, join=True)
        assert dict(failures) == {rank: [] for rank in range(cp)}
