import sys
import types

import pytest
import torch

from skyrl.utils import token_metadata


@pytest.fixture
def parallel_state(monkeypatch):
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    return mpu


def test_microbatch_rows_share_one_packed_layout(monkeypatch, parallel_state):
    monkeypatch.setattr(token_metadata, "get_packed_seq_align_size", lambda *args, **kwargs: 4)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    routes = torch.tensor(
        [
            [[[0, 1]], [[10, 11]], [[12, 13]], [[14, 15]]],
            [[[0, 1]], [[0, 1]], [[20, 21]], [[22, 23]]],
        ],
        dtype=torch.int16,
    )
    router_mask = torch.tensor([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=torch.bool)

    layout = token_metadata.build_token_metadata_layout(
        attention_mask,
        routes.device,
        packed=True,
        fp8_enabled=False,
    )
    packed_routes = token_metadata.align_token_metadata(
        routes,
        layout,
        torch.tensor([0, 1], dtype=routes.dtype),
    )
    packed_mask = token_metadata.align_token_metadata(router_mask, layout, True)

    assert packed_routes[0, :, 0].tolist() == [
        [10, 11],
        [12, 13],
        [14, 15],
        [0, 1],
        [20, 21],
        [22, 23],
        [0, 1],
        [0, 1],
    ]
    assert packed_mask.tolist() == [[False, False, True, True, False, False, True, True]]
    assert layout.cu_seqlens_padded.tolist() == [0, 4, 8]


def test_packed_layout_aligns_next_token_metadata_and_scatters_rows(monkeypatch, parallel_state):
    monkeypatch.setattr(token_metadata, "get_packed_seq_align_size", lambda *args, **kwargs: 4)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    metadata = torch.tensor([[0, 10, 11, 12], [0, 0, 20, 21]], dtype=torch.int32)
    layout = token_metadata.build_token_metadata_layout(
        attention_mask,
        metadata.device,
        packed=True,
        fp8_enabled=False,
    )

    aligned = token_metadata.align_token_metadata(metadata, layout, -1, next_token=True)
    batch_values = token_metadata.scatter_packed_token_values_to_batch(
        torch.arange(1, 9, dtype=torch.float32).unsqueeze(0),
        layout,
        0,
    )

    assert aligned.tolist() == [[11, 12, -1, -1, 21, -1, -1, -1]]
    assert batch_values.tolist() == [[0.0, 1.0, 2.0], [0.0, 0.0, 5.0]]
