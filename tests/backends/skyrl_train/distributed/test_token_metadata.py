import sys
import types

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron import token_metadata
from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataTrace,
)
from skyrl.backends.skyrl_train.utils.routed_experts import RoutedExpertTrace


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


def test_token_metadata_trace_chunks_and_independent_schema() -> None:
    trace, other = TokenMetadataTrace(), TokenMetadataTrace()
    trace.append(np.ones((2, 3), dtype=np.int32), expected_rows=2)
    trace.append(np.zeros((1, 3), dtype=np.int32), expected_rows=1)
    other.append(np.empty((0, 4), dtype=np.float32), expected_rows=0)

    with pytest.raises(ValueError, match="expected 4"):
        trace.finalize(expected_rows=4)
    result = trace.finalize(expected_rows=3)
    assert result.shape == (3, 3)
    assert other.finalize(expected_rows=0).shape == (0, 4)
    with pytest.raises(RuntimeError, match="already finalized"):
        trace.finalize(expected_rows=3)


@pytest.mark.parametrize(
    ("rows", "expected", "match"),
    [
        (np.ones((2, 2), dtype=np.int32), 1, "has 2 rows"),
        (np.ones((2, 2), dtype=np.int32)[:, ::2], 2, "contiguous"),
        (np.ones((1, 3), dtype=np.int32), 1, "schema changed"),
        (np.ones((1, 2), dtype=np.float32), 1, "dtype changed"),
        (np.ones((1, 2), dtype=np.int8), 1, "dtype changed"),
    ],
)
def test_token_metadata_trace_rejects_invalid_chunks(rows, expected, match) -> None:
    trace = TokenMetadataTrace()
    if rows.shape[0] == 1:
        trace.append(np.ones((1, 2), dtype=np.uint8), expected_rows=1)
    with pytest.raises(ValueError, match=match):
        trace.append(rows, expected_rows=expected)


@pytest.mark.parametrize(
    ("dtypes", "expected_dtype"),
    [
        ((np.uint8, np.int16), np.int16),
        ((np.int16, np.uint8), np.int16),
        ((np.uint8, np.int16, np.int32), np.int32),
        ((np.int32, np.int32), np.int32),
    ],
)
def test_token_metadata_trace_widens_dtype_across_chunks(dtypes, expected_dtype) -> None:
    trace = TokenMetadataTrace()
    chunks = [np.full((2, 2), 100 + i, dtype=dtype) for i, dtype in enumerate(dtypes)]
    for chunk in chunks:
        trace.append(chunk, expected_rows=2)
    assert trace.dtype == np.dtype(expected_dtype)

    result = trace.finalize(expected_rows=2 * len(chunks))
    assert result.dtype == np.dtype(expected_dtype)
    assert np.array_equal(result, np.concatenate([chunk.astype(expected_dtype) for chunk in chunks]))


def test_token_metadata_trace_single_chunk_keeps_its_dtype() -> None:
    trace = TokenMetadataTrace()
    chunk = np.ones((3, 2), dtype=np.uint8)
    trace.append(chunk, expected_rows=3)
    assert trace.finalize(expected_rows=3) is chunk


def routes(rows: int) -> np.ndarray:
    return np.arange(rows * 4, dtype=np.int32).reshape(rows, 2, 2) % 8


def test_routed_expert_trace_tracks_multiturn_suffix_and_terminal_gap() -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=routes(4))
    assert trace.prompt_start == 4
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=routes(4))

    result = trace.finalize(token_count=9, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1])
    assert result.shape == (9, 2, 2) and result.dtype == np.uint8
    assert np.array_equal(result[-1, 0], [0, 1])


def test_routed_expert_trace_widens_when_a_later_turn_routes_to_a_high_expert() -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=routes(4))
    high = routes(4).copy()
    high[0, 0, 0] = 300
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=high)

    result = trace.finalize(token_count=9, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1])
    assert result.dtype == np.int16
    assert np.array_equal(result[:4], routes(4))
    assert result[4, 0, 0] == 300
    assert np.array_equal(result[-1, 0], [0, 1])


@pytest.mark.parametrize("active", [False, True])
def test_routed_expert_trace_only_pads_masked_suffix(active: bool) -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=1, routed_experts=routes(3))
    mask = [0, 0, 0, 0, int(active)]
    if active:
        with pytest.raises(ValueError, match="loss-active target"):
            trace.finalize(token_count=5, loss_mask=mask)
    else:
        result = trace.finalize(token_count=5, loss_mask=mask)
        assert np.array_equal(result[-2:, 0], [[0, 1], [0, 1]])
