"""Tests for ``PackedTensor`` batch operations."""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.packed_tensor import (
    CU_SEQLENS_DTYPE,
    PackedTensor,
    cu_seqlens_from_lengths,
    lengths_from_offsets,
    row_index_from_offsets,
)

SEGMENT_LENGTHS = [3, 1, 4, 2]


def _segments(lengths=SEGMENT_LENGTHS, *, row_shape=(2, 3)) -> list[torch.Tensor]:
    """Distinct rows per segment so any misplacement is visible."""
    segments = []
    next_value = 0
    for length in lengths:
        size = length * torch.Size(row_shape).numel()
        segments.append(torch.arange(next_value, next_value + size, dtype=torch.int16).reshape(length, *row_shape))
        next_value += size
    return segments


@pytest.mark.parametrize(
    ("lengths", "expected_offsets"),
    [
        (SEGMENT_LENGTHS, [0, 3, 4, 8, 10]),
        ([0, 2, 0], [0, 0, 2, 2]),
    ],
)
def test_offsets_round_trip_lengths_in_int32(lengths, expected_offsets):
    offsets = cu_seqlens_from_lengths(lengths)

    assert offsets.tolist() == expected_offsets
    assert offsets.dtype == CU_SEQLENS_DTYPE
    assert lengths_from_offsets(offsets).tolist() == lengths
    assert lengths_from_offsets(offsets).dtype == CU_SEQLENS_DTYPE


def test_cu_seqlens_reject_negative_lengths_and_extra_dimensions():
    with pytest.raises(ValueError, match="non-negative"):
        cu_seqlens_from_lengths([3, -1])
    with pytest.raises(ValueError, match="must be 1-D"):
        cu_seqlens_from_lengths(torch.zeros((2, 2), dtype=torch.int32))


def test_row_index_from_offsets_lays_selected_segments_back_to_back():
    starts = torch.tensor([8, 0])
    lengths = torch.tensor([2, 3])

    assert row_index_from_offsets(starts, lengths).tolist() == [8, 9, 0, 1, 2]


def test_from_segments_round_trips_every_segment():
    segments = _segments()
    packed = PackedTensor.from_segments(segments)

    assert len(packed) == len(segments)
    assert packed.values.shape == (sum(SEGMENT_LENGTHS), 2, 3)
    assert packed.sequence_lengths.tolist() == SEGMENT_LENGTHS
    assert packed.row_shape == torch.Size((2, 3))
    assert packed.dtype == torch.int16
    assert packed.device == segments[0].device
    for index, segment in enumerate(segments):
        assert torch.equal(packed.segment(index), segment)


def test_from_segments_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="empty list of segments"):
        PackedTensor.from_segments([])


def test_negative_and_out_of_range_segment_indices():
    packed = PackedTensor.from_segments(_segments())

    assert torch.equal(packed.segment(-1), packed.segment(len(packed) - 1))
    with pytest.raises(IndexError, match="out of range"):
        packed.segment(len(packed))
    with pytest.raises(IndexError, match="out of range"):
        packed.segment(-len(packed) - 1)


def test_integer_index_returns_that_segment():
    segments = _segments()
    packed = PackedTensor.from_segments(segments)

    assert torch.equal(packed[2], segments[2])
    assert torch.equal(packed[torch.tensor(2)], segments[2])


@pytest.mark.parametrize("bounds", [(0, 4), (1, 3), (2, 4)])
def test_contiguous_slice_selects_the_same_segments(bounds):
    segments = _segments()
    packed = PackedTensor.from_segments(segments)
    start, stop = bounds

    sliced = packed[start:stop]

    assert len(sliced) == stop - start
    assert sliced.cu_seqlens[0] == 0
    for offset, segment in enumerate(segments[start:stop]):
        assert torch.equal(sliced.segment(offset), segment)


@pytest.mark.parametrize("indices", [[2, 0], [3, 3, 1], [0, 1, 2, 3], [1]])
def test_gather_selects_segments_in_the_requested_order(indices):
    segments = _segments()
    packed = PackedTensor.from_segments(segments)

    for gathered in (packed[torch.tensor(indices)], packed[indices], packed[tuple(indices)]):
        assert gathered.sequence_lengths.tolist() == [SEGMENT_LENGTHS[index] for index in indices]
        for position, index in enumerate(indices):
            assert torch.equal(gathered.segment(position), segments[index])


def test_empty_slice_is_rejected_like_an_empty_tensor_list():
    """``TensorBatch`` fields cannot hold zero batch entries, so neither can a slice."""
    packed = PackedTensor.from_segments(_segments())

    with pytest.raises(ValueError, match="at least two offsets"):
        packed[2:2]


def test_strided_slice_falls_back_to_a_gather():
    segments = _segments()
    packed = PackedTensor.from_segments(segments)

    strided = packed[::2]

    assert len(strided) == 2
    assert torch.equal(strided.segment(0), segments[0])
    assert torch.equal(strided.segment(1), segments[2])


def test_cat_joins_batches_end_to_end():
    left = PackedTensor.from_segments(_segments([3, 1]))
    right = PackedTensor.from_segments(_segments([4, 2]))

    joined = PackedTensor.cat([left, right])

    assert joined.sequence_lengths.tolist() == [3, 1, 4, 2]
    assert torch.equal(joined.segment(0), left.segment(0))
    assert torch.equal(joined.segment(2), right.segment(0))


def test_cat_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty list of packed batches"):
        PackedTensor.cat([])


def test_repeat_tiles_and_repeat_interleave_duplicates():
    packed = PackedTensor.from_segments(_segments([3, 1]))

    tiled = packed.repeat(2)
    interleaved = packed.repeat_interleave(2)

    assert tiled.sequence_lengths.tolist() == [3, 1, 3, 1]
    assert interleaved.sequence_lengths.tolist() == [3, 3, 1, 1]
    assert torch.equal(tiled.segment(2), packed.segment(0))
    assert torch.equal(interleaved.segment(1), packed.segment(0))


def test_to_and_contiguous_preserve_the_batch():
    packed = PackedTensor.from_segments(_segments())

    widened = packed.to(dtype=torch.int32)

    assert widened.dtype == torch.int32
    assert widened.cu_seqlens.dtype == CU_SEQLENS_DTYPE
    assert torch.equal(widened.values, packed.values.to(torch.int32))
    assert packed.contiguous() == packed


def test_equality_compares_values_and_offsets():
    packed = PackedTensor.from_segments(_segments())

    assert packed == PackedTensor.from_segments(_segments())
    assert packed != PackedTensor.from_segments(_segments([3, 1, 4, 2], row_shape=(1, 3)))
    assert packed != PackedTensor.from_segments(_segments([4, 4, 2]))
    assert packed != packed.values


def test_rejects_mismatched_device_or_offset_dtype():
    values = torch.zeros((4, 2), dtype=torch.int16)

    with pytest.raises(ValueError, match="must be torch.int32"):
        PackedTensor(values, torch.tensor([0, 4], dtype=torch.int64))
    with pytest.raises(ValueError, match="at least two offsets"):
        PackedTensor(values, torch.tensor([0], dtype=CU_SEQLENS_DTYPE))
    with pytest.raises(ValueError, match="at least two offsets"):
        PackedTensor(values, torch.zeros((2, 2), dtype=CU_SEQLENS_DTYPE))


def test_rejects_offsets_that_do_not_span_the_buffer():
    values = torch.zeros((4, 2), dtype=torch.int16)

    with pytest.raises(ValueError, match="must run from 0"):
        PackedTensor(values, torch.tensor([0, 3], dtype=CU_SEQLENS_DTYPE))
    with pytest.raises(ValueError, match="must run from 0"):
        PackedTensor(values, torch.tensor([1, 4], dtype=CU_SEQLENS_DTYPE))


def test_rejects_values_without_a_token_row_dimension():
    with pytest.raises(ValueError, match="token-row dimension"):
        PackedTensor(torch.tensor(1), torch.tensor([0, 1], dtype=CU_SEQLENS_DTYPE))


def test_segments_and_contiguous_slices_are_views():
    """Reading a batch entry must not copy: alignment walks every segment per micro-batch."""
    packed = PackedTensor.from_segments(_segments())

    assert packed.segment(1).data_ptr() == packed.values[3].data_ptr()
    assert packed[1:3].values.data_ptr() == packed.values[3].data_ptr()


@pytest.mark.parametrize(
    "operation",
    [
        lambda packed: packed[torch.tensor([1, 0])],
        lambda packed: packed[::2],
        lambda packed: packed.repeat(2),
        lambda packed: packed.repeat_interleave(2),
        lambda packed: PackedTensor.cat([packed, packed]),
        lambda packed: packed.to(dtype=torch.int32),
    ],
    ids=["gather", "strided_slice", "repeat", "repeat_interleave", "cat", "to_dtype"],
)
def test_reordering_operations_allocate_rather_than_alias(operation):
    """A duplicated or reordered segment must own its rows; the source stays untouched."""
    packed = PackedTensor.from_segments(_segments())
    original = packed.values.clone()

    produced = operation(packed)
    produced.values[:] = -1

    assert torch.equal(packed.values, original)
