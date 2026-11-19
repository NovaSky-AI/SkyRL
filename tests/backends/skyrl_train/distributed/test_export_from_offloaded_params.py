"""CPU tests for ``export_from_offloaded_params``: a Megatron-Bridge export reading offloaded DDP
params from their CPU copies (colocated full-weight sync, ``weight_sync_from_offloaded_params``).

The DDP buffers are faked the way ``_ParamAndGradBuffer.offload_to_cpu`` leaves them: params are
views of a flat ``param_data`` whose storage is resized to 0, the data lives in ``param_data_cpu``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.megatron

try:
    from megatron.bridge.models.conversion import model_bridge

    from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


class _FakeDDP:
    def __init__(self, buffers, expert_buffers=()):
        self.buffers = list(buffers)
        self.expert_parallel_buffers = list(expert_buffers)


def _buffer(shapes, offload: bool):
    numel = sum(torch.Size(s).numel() for s in shapes)
    flat = torch.arange(numel, dtype=torch.bfloat16)
    params, index_map, start = [], {}, 0
    for shape in shapes:
        end = start + torch.Size(shape).numel()
        param = torch.nn.Parameter(flat[start:end].view(shape))
        index_map[param] = (start, end, 0)
        params.append((param, flat[start:end].clone().view(shape)))
        start = end
    buffer = SimpleNamespace(param_data=flat, param_data_cpu=None, param_index_map=index_map)
    if offload:
        buffer.param_data_cpu = flat.clone()
        flat.untyped_storage().resize_(0)
    return buffer, params


@pytest.fixture
def fake_ddp(monkeypatch):
    monkeypatch.setattr(megatron_utils, "DDP", _FakeDDP)


def test_offloaded_params_read_from_cpu_copies(fake_ddp):
    dense, dense_params = _buffer([(4, 3), (5,)], offload=True)
    experts, expert_params = _buffer([(2, 2, 3)], offload=True)
    model = [_FakeDDP([dense], [experts])]
    original = model_bridge._gather_gtp_weight
    with megatron_utils.export_from_offloaded_params(model):
        for param, expected in dense_params + expert_params:
            assert param.untyped_storage().size() == 0
            got = model_bridge._gather_gtp_weight(param)
            assert got.shape == param.shape
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert model_bridge._gather_gtp_weight is original
    # The training-side storage is never touched: the params stay offloaded.
    assert all(p.untyped_storage().size() == 0 for p, _ in dense_params + expert_params)


def test_resident_params_and_unknown_tensors_pass_through(fake_ddp):
    resident, resident_params = _buffer([(3, 3)], offload=False)
    model = [_FakeDDP([resident])]
    other = torch.ones(2)
    with megatron_utils.export_from_offloaded_params(model):
        # Nothing offloaded: the bridge's own read is left in place.
        param = resident_params[0][0]
        assert model_bridge._gather_gtp_weight(param) is param
        assert model_bridge._gather_gtp_weight(other) is other


def test_non_ddp_chunks_are_ignored(fake_ddp):
    original = model_bridge._gather_gtp_weight
    with megatron_utils.export_from_offloaded_params([torch.nn.Linear(2, 2)]):
        assert model_bridge._gather_gtp_weight is original
