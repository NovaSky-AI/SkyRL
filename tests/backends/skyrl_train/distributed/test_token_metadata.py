import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron import token_metadata
from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
)
from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenLayout,
    TokenMetadataTrace,
)
from skyrl.backends.skyrl_train.utils.routed_experts import RoutedExpertTrace


def _align(monkeypatch, packed: int = 1, unpacked: int = 1) -> None:
    """Fix the packed per-segment and the unpacked row alignment."""
    monkeypatch.setattr(token_metadata, "get_packed_seq_align_size", lambda *args, **kwargs: packed)
    monkeypatch.setattr(token_metadata, "get_unpacked_seq_align_size", lambda *args, **kwargs: unpacked)


def test_rows_place_their_real_tokens_into_one_packed_layout(monkeypatch):
    _align(monkeypatch, packed=4)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    router_mask = torch.tensor([[1, 0, 0, 1], [1, 1, 0, 0]], dtype=torch.bool)

    layout = TokenLayout.build(attention_mask, packed=True, tp_size=1)

    # Every segment is padded to the alignment.
    assert layout.cu_seqlens_padded.tolist() == [0, 4, 8]
    assert layout.ordinals.tolist() == [[0, 1, 2, -1, 3, 4, -1, -1]]
    assert layout.place(router_mask, True).tolist() == [[False, False, True, True, False, False, True, True]]
    assert layout.padding_mask.tolist() == [[False, False, False, True, False, False, True, True]]


def test_packed_layout_places_next_tokens_and_scatters_predictions_back(monkeypatch):
    _align(monkeypatch, packed=4)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    metadata = torch.tensor([[0, 10, 11, 12], [0, 0, 20, 21]], dtype=torch.int32)
    layout = TokenLayout.build(attention_mask, packed=True, tp_size=1)

    aligned = layout.place(metadata, -1, next_token=True)
    scatter = layout.packed_scatter()
    batch_values = scatter.scatter(torch.arange(1, 9, dtype=torch.float32).unsqueeze(0), 0)
    # Integer indices, computed on the host: scattering on the device takes no boolean mask, so no sync.
    assert {scatter.positions.dtype, scatter.rows.dtype, scatter.cols.dtype} == {torch.int64}

    assert aligned.tolist() == [[11, 12, -1, -1, 21, -1, -1, -1]]
    assert layout.next_token_mask().tolist() == [[True, True, False, False, True, False, False, False]]
    # Position t predicts token t + 1: the predictions of packed positions 0, 1 and 4.
    assert batch_values.tolist() == [[0.0, 1.0, 2.0], [0.0, 0.0, 5.0]]


def test_cp_fp8_segments_are_each_padded_to_the_fp8_alignment():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    align = get_packed_seq_align_size(1, 2, fp8_enabled=True)

    layout = TokenLayout.build(attention_mask, packed=True, tp_size=1, cp_size=2, fp8_enabled=True)

    assert layout.cu_seqlens_padded.tolist() == [0, align, 2 * align]
    assert layout.ordinals.shape == (1, align)


def test_cp_ranks_hold_a_front_and_a_mirrored_back_chunk_of_each_segment(monkeypatch):
    _align(monkeypatch, packed=8)
    attention_mask = torch.tensor([[1] * 7, [0, 0, 0, 1, 1, 1, 1]])
    ids = torch.arange(14).reshape(2, 7)
    layouts = [TokenLayout.build(attention_mask, packed=True, tp_size=1, cp_size=2, cp_rank=r) for r in (0, 1)]

    # Segment 0 holds tokens 0..6 plus one pad, segment 1 tokens 10..13 plus four; chunks of 2.
    assert layouts[0].place(ids, -1).tolist() == [[0, 1, 6, -1, 10, 11, -1, -1]]
    assert layouts[1].place(ids, -1).tolist() == [[2, 3, 4, 5, 12, 13, -1, -1]]
    assert layouts[0].full().place(ids, -1).tolist() == [[0, 1, 2, 3, 4, 5, 6, -1, 10, 11, 12, 13, -1, -1, -1, -1]]


@pytest.mark.parametrize("packed", [False, True])
def test_per_token_rows_place_from_their_ordinals(monkeypatch, packed):
    _align(monkeypatch, packed=4, unpacked=4)
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    rows = torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32)
    layout = TokenLayout.build(attention_mask, packed=packed, tp_size=1)

    aligned = layout.place_ordinals(rows, -1)

    if packed:
        assert aligned.tolist() == [[10, 11, 12, -1, 20, 21, -1, -1]]
    else:
        assert aligned.tolist() == [[10, 11, 12, -1], [20, 21, -1, -1]]
    assert layout.to_batch(rows, -1).tolist() == [[-1, 10, 11, 12], [-1, -1, 20, 21]]


def test_unpacked_layout_restores_canonical_rows(monkeypatch):
    _align(monkeypatch, unpacked=4)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 0, 1, 1]])
    layout = TokenLayout.build(attention_mask, packed=False, tp_size=1)

    destination, source, counts = layout.recover()

    assert layout.ordinals.tolist() == [[0, 1, 2, 3], [4, 5, -1, -1]]
    assert destination.tolist() == [0, 1, 2, 3, 4, 5]
    assert source.tolist() == [1, 2, 3, 4, 8, 9]
    assert counts == [4, 2]


def test_sub_sequences_are_segments_at_the_collator_row_offsets(monkeypatch):
    _align(monkeypatch, packed=4)
    ids = torch.tensor([[1, 2, 0, 0, 3, 4, 5, 0]])
    attention_mask = ids != 0
    layout = TokenLayout.build(attention_mask, packed=True, tp_size=1, sub_seq_lengths=[[2, 3]])

    assert layout.cu_seqlens_padded.tolist() == [0, 4, 8]
    assert layout.place(ids, -1).tolist() == [[1, 2, -1, -1, 3, 4, 5, -1]]
    assert layout.place(ids, -1, next_token=True).tolist() == [[2, -1, -1, -1, 4, 5, -1, -1]]
    with pytest.raises(ValueError, match="beyond the packed batch row"):
        TokenLayout.build(attention_mask, packed=True, tp_size=1, sub_seq_lengths=[[2, 5]])


def test_sub_sequences_under_cp_refuse_models_that_align_cp_chunks(monkeypatch):
    monkeypatch.setattr(token_metadata, "get_cp_chunk_token_align", lambda: 4)
    ids = torch.tensor([[1, 2, 0, 0, 3, 4, 5, 0]])
    with pytest.raises(NotImplementedError, match="align CP chunks"):
        TokenLayout.build(ids != 0, packed=True, tp_size=1, cp_size=2, sub_seq_lengths=[[2, 3]])


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

    # `finalize` returns only the rows the engine actually captured -- `prompt_start`, one short
    # of `token_count`, because the last sampled token has no subsequent decode forward. The
    # trailing dummy row is built during collation instead, where `make_router_padding_mask` can
    # mark it so Megatron excludes it from router accounting.
    result = trace.finalize(token_count=9, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1])
    assert trace.prompt_start == 8
    assert result.shape == (8, 2, 2) and result.dtype == np.uint8
    # The last row is a real captured route now, not a dummy `arange(topk)` pad row.
    assert np.array_equal(result[-1, 0], [4, 5])


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
    assert np.array_equal(result[-1, 0], [4, 5])


@pytest.mark.parametrize("active", [False, True])
def test_routed_expert_trace_refuses_a_loss_active_target_without_a_row(active: bool) -> None:
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=1, routed_experts=routes(3))
    mask = [0, 0, 0, 0, int(active)]
    if active:
        with pytest.raises(ValueError, match="loss-active target"):
            trace.finalize(token_count=5, loss_mask=mask)
    else:
        # A masked suffix is no longer dummy-padded up to `token_count`: the captured rows are
        # returned verbatim and the gap is filled during collation.
        result = trace.finalize(token_count=5, loss_mask=mask)
        assert result.shape == (3, 2, 2)
        assert np.array_equal(result, routes(3).astype(result.dtype))


def test_append_padding_extends_the_established_schema():
    trace = token_metadata.TokenMetadataTrace()
    trace.append(np.array([[7, 8], [9, 10]], dtype=np.int32), expected_rows=2)

    trace.append_padding(0)
    assert trace.num_rows == 2

    trace.append_padding(2)
    padded = trace.finalize(expected_rows=4)

    assert padded.dtype == np.int32
    assert padded.flags.c_contiguous
    assert padded.tolist() == [[7, 8], [9, 10], [-1, -1], [-1, -1]]


def test_append_padding_needs_a_schema_and_a_valid_count():
    with pytest.raises(ValueError, match="before any rows are captured"):
        token_metadata.TokenMetadataTrace().append_padding(1)

    trace = token_metadata.TokenMetadataTrace()
    trace.append(np.zeros((1, 2), dtype=np.int32), expected_rows=1)
    for count in (-1, True, 1.0):
        with pytest.raises(ValueError, match="padding count"):
            trace.append_padding(count)


def test_unpacked_layout_honors_extra_seq_align(monkeypatch):
    # seq_align pads unpacked microbatches to an extra alignment (mlp_chunks_for_training, DSV4 compression ratios).
    monkeypatch.setattr(token_metadata, "get_unpacked_seq_align_size", lambda *args, **kwargs: 8)
    attention_mask = torch.ones((2, 211), dtype=torch.bool)

    default = token_metadata.TokenLayout.build(attention_mask, packed=False, tp_size=1)
    aligned = token_metadata.TokenLayout.build(attention_mask, packed=False, tp_size=1, seq_align=128)

    assert default.ordinals.shape == (2, 216)
    assert aligned.ordinals.shape == (2, 256)


def _collated_ids(sub_seq_lengths, align):
    """Rows as the packed collator lays them out: sub-sequences at aligned offsets, unique nonzero ids."""
    width = max(sum(length + (-length % align) for length in lengths) for lengths in sub_seq_lengths)
    ids = torch.zeros((len(sub_seq_lengths), width), dtype=torch.long)
    next_id = 1
    for row, lengths in enumerate(sub_seq_lengths):
        offset = 0
        for length in lengths:
            ids[row, offset : offset + length] = torch.arange(next_id, next_id + length)
            next_id += length
            offset += length + (-length % align)
    return ids


@pytest.mark.parametrize("fp8_enabled", [False, True])
@pytest.mark.parametrize(
    "tp_size,cp_size,sub_seq_lengths",
    [
        (1, 2, [[6, 5], [7, 3]]),
        (2, 2, [[5, 9], [3, 6]]),
        # Ranks 0..3 hold chunks {0, 7}, {1, 6}, {2, 5}, {3, 4} of every segment.
        (1, 4, [[10, 6], [13, 3]]),
        (1, 2, [[7], [10]]),
    ],
)
def test_cp_shards_reassemble_into_the_full_packed_stream(tp_size, cp_size, sub_seq_lengths, fp8_enabled):
    ids = _collated_ids(sub_seq_lengths, get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled))
    layouts = [
        TokenLayout.build(
            ids != 0,
            packed=True,
            tp_size=tp_size,
            cp_size=cp_size,
            cp_rank=rank,
            sub_seq_lengths=sub_seq_lengths,
            fp8_enabled=fp8_enabled,
        )
        for rank in range(cp_size)
    ]
    shards = [layout.place(ids, 0).squeeze(0) for layout in layouts]
    cu = layouts[0].cu_seqlens_padded.tolist()
    assert all(shard.numel() == cu[-1] // cp_size for shard in shards)
    if fp8_enabled:
        assert shards[0].numel() % 16 == 0

    # Undo the zigzag: rank r holds chunk r and its mirror 2 * CP - 1 - r of each segment.
    first_id = 1
    lengths = [length for row in sub_seq_lengths for length in row]
    for start, end, length in zip(cu[:-1], cu[1:], lengths):
        local = (end - start) // cp_size
        halves = [shard[start // cp_size : start // cp_size + local].chunk(2) for shard in shards]
        segment = torch.cat([front for front, _ in halves] + [back for _, back in reversed(halves)])
        expected = torch.zeros(end - start, dtype=torch.long)
        expected[:length] = torch.arange(first_id, first_id + length)
        assert torch.equal(segment, expected)
        first_id += length


@pytest.mark.parametrize("tp_size,cp_size,real_tokens", [(4, 2, 2), (4, 2, 1), (2, 2, 2), (1, 2, 2)])
def test_segments_shorter_than_the_cp_alignment_pack(tp_size, cp_size, real_tokens):
    attention_mask = torch.ones((2, 32), dtype=torch.bool)
    attention_mask[0, real_tokens:] = False
    ids = torch.arange(1, 65).reshape(2, 32) * attention_mask

    shards = [
        TokenLayout.build(attention_mask, packed=True, tp_size=tp_size, cp_size=cp_size, cp_rank=rank, fp8_enabled=True)
        for rank in range(cp_size)
    ]

    assert shards[0].cu_seqlens_padded[-1] % (16 * cp_size) == 0
    for layout in shards:
        assert layout.ordinals.shape[1] % 16 == 0
    placed = torch.cat([layout.place(ids, 0) for layout in shards], dim=1)
    assert sorted(placed[placed != 0].tolist()) == ids[attention_mask].tolist()


@pytest.mark.parametrize(
    "tp_size,fp8_enabled,real_tokens,width",
    [(1, False, 6541, 6541), (1, True, 6541, 6544), (2, True, 8552, 8704)],
)
def test_unpacked_width_aligns_for_tp_and_fp8(tp_size, fp8_enabled, real_tokens, width):
    attention_mask = torch.zeros((1, real_tokens + 500), dtype=torch.bool)
    attention_mask[0, :real_tokens] = True

    layout = TokenLayout.build(attention_mask, packed=False, tp_size=tp_size, fp8_enabled=fp8_enabled)

    assert layout.ordinals.shape == (1, width)
    assert int((~layout.padding_mask).sum()) == real_tokens
