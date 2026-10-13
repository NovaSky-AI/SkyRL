"""Tests for two-level ragged batch operations."""

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import (
    PackedRaggedTensor,
    RaggedRows,
    packed_ragged_padding_segments,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    CU_SEQLENS_DTYPE,
    PackedTensor,
    cu_seqlens_from_lengths,
)

PADDING = -1
MEMBERS: list[list[list[int]]] = [
    [[7, 8, 9], [4], [1, 2]],
    [[]],
    [[5, 6, 7, 8], [3, 4]],
]


# A second column, in another dtype, derived from each member so tests can check alignment.
SCORE_SCALE = 0.5


def _build(members=MEMBERS) -> PackedRaggedTensor:
    flat = [member for entry in members for row in entry for member in row]
    row_lengths = [len(row) for entry in members for row in entry]
    ids = torch.tensor(flat, dtype=torch.int32)
    return PackedRaggedTensor(
        [ids, ids.to(torch.float32) * SCORE_SCALE],
        cu_seqlens_from_lengths(row_lengths),
        cu_seqlens_from_lengths([len(entry) for entry in members]),
    )


def _read(packed: PackedRaggedTensor) -> list[list[list[int]]]:
    """Each entry's rows of members, after checking the score column still tracks them."""
    ids, scores = packed.columns
    assert torch.equal(scores, ids.to(torch.float32) * SCORE_SCALE)
    entries = []
    row = 0
    for length in packed.sequence_lengths.tolist():
        entries.append([packed.row(row + offset).tolist() for offset in range(length)])
        row += length
    return entries


def _padded(members=MEMBERS, *, width=4) -> PackedTensor:
    rows = [row + [PADDING] * (width - len(row)) for entry in members for row in entry]
    return PackedTensor(
        torch.tensor(rows, dtype=torch.int32),
        cu_seqlens_from_lengths([len(entry) for entry in members]),
    )


def test_an_entry_with_no_rows_survives_the_round_trip():
    ragged = _build([[]])

    assert len(ragged) == 1
    assert ragged.num_rows == 0
    assert ragged.num_members == 0
    assert _read(ragged) == [[]]


def test_the_outer_offsets_mean_what_they_mean_on_packed_tensor():
    ragged = _build()
    padded = _padded()

    assert len(ragged) == len(padded)
    assert ragged.sequence_lengths.tolist() == padded.sequence_lengths.tolist()
    assert torch.equal(ragged.cu_seqlens, padded.cu_seqlens)
    assert ragged.num_rows == padded.values.shape[0]
    assert int(ragged.cu_seqlens[-1]) == int(padded.cu_seqlens[-1])


@pytest.mark.parametrize("bounds", [(0, 3), (1, 3), (0, 1), (2, 3)])
def test_contiguous_slice_selects_the_same_entries(bounds):
    ragged = _build()
    start, stop = bounds

    sliced = ragged[start:stop]

    assert _read(sliced) == MEMBERS[start:stop]
    assert int(sliced.row_offsets[0]) == 0
    assert int(sliced.cu_seqlens[0]) == 0


@pytest.mark.parametrize("indices", [[2, 0], [1, 1, 2], [0, 1, 2], [1]])
def test_gather_selects_entries_in_the_requested_order(indices):
    ragged = _build()

    for gathered in (ragged[torch.tensor(indices)], ragged[indices], ragged[tuple(indices)]):
        assert _read(gathered) == [MEMBERS[index] for index in indices]


def test_integer_and_negative_indices_return_that_entry():
    ragged = _build()

    assert _read(ragged[1]) == [MEMBERS[1]]
    assert _read(ragged.segment(-1)) == [MEMBERS[-1]]
    assert _read(ragged[torch.tensor(2)]) == [MEMBERS[2]]
    with pytest.raises(IndexError, match="out of range"):
        ragged.segment(len(ragged))


def test_row_addresses_the_whole_batch_and_rejects_a_missing_row():
    ragged = _build()

    assert ragged.row(0).tolist() == MEMBERS[0][0]
    assert ragged.row(0, column=1).tolist() == [member * SCORE_SCALE for member in MEMBERS[0][0]]
    assert ragged.row(3).tolist() == []
    assert ragged.row(-1).tolist() == MEMBERS[2][1]
    with pytest.raises(IndexError, match="out of range"):
        ragged.row(ragged.num_rows)


def test_strided_slice_falls_back_to_a_gather():
    ragged = _build()

    assert _read(ragged[::2]) == [MEMBERS[0], MEMBERS[2]]


def test_cat_joins_batches_end_to_end():
    left = _build(MEMBERS[:1])
    right = _build(MEMBERS[1:])

    joined = PackedRaggedTensor.cat([left, right])

    assert _read(joined) == MEMBERS
    with pytest.raises(ValueError, match="empty list of packed ragged batches"):
        PackedRaggedTensor.cat([])


def test_cat_needs_one_column_schema_among_batches_with_members():
    with_members = _build(MEMBERS[:1])
    ids_only = PackedRaggedTensor(with_members.columns[:1], with_members.row_offsets, with_members.cu_seqlens)
    memberless = PackedRaggedTensor(
        [torch.empty(0, dtype=torch.int64)], cu_seqlens_from_lengths([0, 0]), cu_seqlens_from_lengths([2])
    )

    with pytest.raises(ValueError, match="must share member columns"):
        PackedRaggedTensor.cat([with_members, ids_only])
    joined = PackedRaggedTensor.cat([memberless, with_members, memberless])

    assert joined.column_dtypes == with_members.column_dtypes
    assert _read(joined) == [[[], []], MEMBERS[0], [[], []]]


def test_repeat_tiles_and_repeat_interleave_duplicates():
    ragged = _build(MEMBERS[:2])

    assert _read(ragged.repeat(2)) == MEMBERS[:2] + MEMBERS[:2]
    assert _read(ragged.repeat_interleave(2)) == [MEMBERS[0], MEMBERS[0], MEMBERS[1], MEMBERS[1]]


def test_to_contiguous_and_equality_preserve_both_levels():
    ragged = _build()

    moved = ragged.to(device="cpu")

    assert moved.column_dtypes == (torch.int32, torch.float32)
    assert moved.row_offsets.dtype == CU_SEQLENS_DTYPE
    assert moved.cu_seqlens.dtype == CU_SEQLENS_DTYPE
    assert _read(moved) == MEMBERS
    assert ragged.contiguous() == ragged
    assert ragged == _build()
    assert ragged != _build(MEMBERS[:2])
    assert ragged != PackedRaggedTensor(ragged.columns[:1], ragged.row_offsets, ragged.cu_seqlens)
    assert ragged != ragged.columns[0]


def test_equality_sees_a_regrouping_that_leaves_the_members_alone():
    regrouped = _build([[[7, 8, 9], [4]], [[1, 2], []], [[5, 6, 7, 8], [3, 4]]])

    assert all(torch.equal(mine, theirs) for mine, theirs in zip(regrouped.columns, _build().columns))
    assert regrouped != _build()


def test_repr_names_the_batch_the_rows_and_the_members():
    assert repr(_build()) == ("PackedRaggedTensor(batch=3, rows=6, members=12, columns=[torch.int32, torch.float32])")


def test_rejects_columns_that_are_missing_not_flat_or_misaligned():
    offsets, cu_seqlens = cu_seqlens_from_lengths([3, 3]), cu_seqlens_from_lengths([2])

    with pytest.raises(ValueError, match="at least one flat member column"):
        PackedRaggedTensor([], offsets, cu_seqlens)
    with pytest.raises(ValueError, match="at least one flat member column"):
        PackedRaggedTensor([torch.zeros((2, 3), dtype=torch.int32)], offsets, cu_seqlens)
    with pytest.raises(ValueError, match="columns must align"):
        PackedRaggedTensor([torch.zeros(6, dtype=torch.int32), torch.zeros(5)], offsets, cu_seqlens)


def test_rejects_offsets_with_the_wrong_dtype_or_rank():
    values = [torch.zeros(4, dtype=torch.int32)]

    with pytest.raises(ValueError, match="row_offsets must be torch.int32"):
        PackedRaggedTensor(values, torch.tensor([0, 4], dtype=torch.int64), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="cu_seqlens must be torch.int32"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([4]), torch.tensor([0, 1], dtype=torch.int64))
    with pytest.raises(ValueError, match="row_offsets must be a non-empty 1-D"):
        PackedRaggedTensor(values, torch.zeros((2, 2), dtype=CU_SEQLENS_DTYPE), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="at least two offsets"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([4]), torch.zeros(1, dtype=CU_SEQLENS_DTYPE))


def test_rejects_offsets_that_do_not_span_their_level():
    values = [torch.zeros(4, dtype=torch.int32)]

    with pytest.raises(ValueError, match="must run from 0 to the 4 packed members"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([3]), cu_seqlens_from_lengths([1]))
    with pytest.raises(ValueError, match="must run from 0 to the 2 packed rows"):
        PackedRaggedTensor(values, cu_seqlens_from_lengths([2, 2]), cu_seqlens_from_lengths([1]))


def test_contiguous_slices_view_the_member_columns():
    ragged = _build()

    # Batch entry 0 spends the first six members, and row 1 the fourth.
    for column, sliced in zip(ragged.columns, ragged[1:3].columns):
        assert sliced.data_ptr() == column[6:].data_ptr()
    assert ragged.row(1).data_ptr() == ragged.columns[0][3:].data_ptr()


@pytest.mark.parametrize(
    "operation",
    [
        lambda packed: packed[torch.tensor([1, 0])],
        lambda packed: packed[::2],
        lambda packed: packed.repeat(2),
        lambda packed: packed.repeat_interleave(2),
        lambda packed: PackedRaggedTensor.cat([packed, packed]),
    ],
    ids=["gather", "strided_slice", "repeat", "repeat_interleave", "cat"],
)
def test_reordering_operations_allocate_rather_than_alias(operation):
    ragged = _build()
    originals = [column.clone() for column in ragged.columns]

    for column in operation(ragged).columns:
        column[:] = -9

    assert all(torch.equal(column, original) for column, original in zip(ragged.columns, originals))


def test_padding_segments_hold_the_requested_rows_with_no_members():
    padding = packed_ragged_padding_segments(_build(), segment_lengths=[2, 0, 1])

    assert padding.sequence_lengths.tolist() == [2, 0, 1]
    assert padding.row_lengths.tolist() == [0, 0, 0]
    assert padding.num_members == 0
    assert padding.column_dtypes == (torch.int32, torch.float32)


def _rows(ids: list[list[int]], dtypes=(np.int32, np.float32)) -> RaggedRows:
    flat = np.array([member for row in ids for member in row], dtype=np.int64)
    return RaggedRows.from_lengths(tuple(flat.astype(dtype) for dtype in dtypes), [len(row) for row in ids])


def test_ragged_rows_slice_and_concat_carry_every_column():
    rows = RaggedRows.concat([_rows([[1, 2], [3]]), _rows([[]]), _rows([[4, 5, 6]])])

    assert rows.row_lengths.tolist() == [2, 1, 0, 3]
    assert rows.column_dtypes == (np.dtype(np.int32), np.dtype(np.float32))
    sliced = rows[1:4]
    assert [column.tolist() for column in sliced.columns] == [[3, 4, 5, 6], [3.0, 4.0, 5.0, 6.0]]
    assert sliced.empty_like(2).column_dtypes == rows.column_dtypes


def test_ragged_rows_concat_needs_one_column_schema_among_blocks_with_members():
    with pytest.raises(ValueError, match="must share member columns"):
        RaggedRows.concat([_rows([[1]]), _rows([[2]], dtypes=(np.int32,))])
    joined = RaggedRows.concat([_rows([[]], dtypes=(np.int64,)), _rows([[1, 2]])])

    assert joined.row_lengths.tolist() == [0, 2]
    assert joined.column_dtypes == (np.dtype(np.int32), np.dtype(np.float32))


def test_ragged_rows_reject_missing_or_misaligned_columns():
    offsets = np.array([0, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="at least one flat member column"):
        RaggedRows((), offsets)
    with pytest.raises(ValueError, match="columns must align"):
        RaggedRows((np.zeros(2, dtype=np.int32), np.zeros(1, dtype=np.float32)), offsets)
    with pytest.raises(ValueError, match="span the members"):
        RaggedRows((np.zeros(3, dtype=np.int32),), offsets)
