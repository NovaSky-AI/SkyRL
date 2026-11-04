import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    CompactSampleSupport,
    SampleSupportColumn,
    SampleSupportTrace,
    packed_member_values,
)
from tests.backends.skyrl_train.sample_support_helpers import csr, dense


def _rows(count: int, first_id: int = 0) -> CompactSampleSupport:
    return csr([[first_id + i, first_id + i + 100] for i in range(count)])


def test_finalize_drops_exactly_the_declared_trailing_rows():
    trace = SampleSupportTrace()
    trace.append(_rows(3), expected_rows=3)
    trace.append_padding(2)

    support = trace.finalize(token_count=3, extra_rows=2)

    np.testing.assert_array_equal(dense(support), dense(_rows(3)))


def test_finalize_rejects_a_trace_shorter_than_the_response():
    trace = SampleSupportTrace()
    trace.append(_rows(2), expected_rows=2)

    with pytest.raises(ValueError, match="2 rows for 3 tokens plus 0 trailing rows"):
        trace.finalize(token_count=3, extra_rows=0)


def test_finalize_rejects_an_unexpected_overshoot():
    trace = SampleSupportTrace()
    trace.append(_rows(3), expected_rows=3)
    trace.append_padding(2)

    with pytest.raises(ValueError, match="5 rows for 3 tokens plus 1 trailing rows"):
        trace.finalize(token_count=3, extra_rows=1)


def test_padding_rows_are_empty():
    trace = SampleSupportTrace()
    trace.append(_rows(1), expected_rows=1)
    trace.append_padding(2)

    support = trace.finalize(token_count=3, extra_rows=0)

    assert support.token_ids.dtype == SAMPLE_SUPPORT_DTYPE
    assert support.row_lengths.tolist() == [2, 0, 0]


def test_finalize_rejects_blocks_with_mixed_top_k():
    trace = SampleSupportTrace()
    with pytest.raises(ValueError, match="expected 3"):
        trace.append(_rows(2), expected_rows=3)
    trace.append(_rows(1), expected_rows=1)
    trace.append(csr([[1, 2, 3]]), expected_rows=1)
    with pytest.raises(ValueError, match="share top_k"):
        trace.finalize(token_count=2, extra_rows=0)


def test_compact_support_preserves_empty_rows_and_slices():
    rows = np.array([[1, 2, -1], [-1, -1, -1], [3, -1, -1]])
    support = csr(rows)

    assert support.offsets.tolist() == [0, 2, 2, 3]
    assert support.top_k == 3
    np.testing.assert_array_equal(dense(support[1:3]), rows[1:])
    assert support[1:3].top_k == 3


def test_from_masked_rows_keeps_members_that_are_not_a_prefix():
    ids = np.array([[5, 6, 7], [8, 9, 10]], dtype=SAMPLE_SUPPORT_DTYPE)
    present = np.array([[False, True, True], [True, False, True]])

    support = CompactSampleSupport.from_masked_rows(ids, present)

    assert support.token_ids.tolist() == [6, 7, 8, 10]
    assert support.offsets.tolist() == [0, 2, 4]


def test_concat_joins_rows_and_rejects_mixed_top_k():
    joined = CompactSampleSupport.concat([csr([[1, -1]]), CompactSampleSupport.empty(1, 2), csr([[2, 3]])])

    assert joined.row_lengths.tolist() == [1, 0, 2]
    assert joined.token_ids.tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="share top_k"):
        CompactSampleSupport.concat([csr([[1]]), csr([[1, 2]])])


def test_contains_checks_each_row_for_its_own_token():
    support = csr([[1, 2], [-1, -1], [3, 4]])

    assert support.contains(np.array([2, 2, 5])).tolist() == [True, False, False]


@pytest.mark.parametrize(
    "token_ids,offsets,top_k,message",
    [
        (np.array([1], dtype=np.int64), np.array([0, 1]), 1, "int32"),
        (np.array([1], dtype=np.int32), np.array([0, 2]), 2, "span"),
        (np.array([1, 2], dtype=np.int32), np.array([0, 2, 1, 2]), 2, "ascend"),
        (np.array([1, 2, 3], dtype=np.int32), np.array([0, 3]), 2, "at most"),
        (np.array([-1], dtype=np.int32), np.array([0, 1]), 1, "nonnegative"),
    ],
)
def test_compact_support_rejects_malformed_rows(token_ids, offsets, top_k, message):
    with pytest.raises(ValueError, match=message):
        CompactSampleSupport((token_ids,), offsets.astype(np.int32), top_k)


def _rows_with_logprobs(rows) -> CompactSampleSupport:
    """Support whose logprob column holds each member's own ID, so a misplaced logprob is visible."""
    rows = np.asarray(rows)
    return csr(rows, logprobs=rows.astype(np.float32))


def test_logprobs_follow_slices_padding_and_concatenation():
    support = _rows_with_logprobs([[1, 2, -1], [-1, -1, -1], [3, -1, -1]])

    for block in (support[1:3], support[:1], support.empty_like(2)):
        np.testing.assert_array_equal(block.logprobs, block.token_ids.astype(np.float32))
    joined = CompactSampleSupport.concat([support[2:], support[:2]])
    np.testing.assert_array_equal(joined.logprobs, joined.token_ids.astype(np.float32))


def test_the_trace_carries_logprobs_through_padding_and_finalize():
    trace = SampleSupportTrace()
    trace.append(_rows_with_logprobs([[4, 5], [6, -1]]), expected_rows=2)
    trace.append_padding(1)
    trace.append(_rows_with_logprobs([[7, 8]]), expected_rows=1)

    finalized = trace.finalize(token_count=3, extra_rows=1)

    assert finalized.row_lengths.tolist() == [2, 1, 0]
    np.testing.assert_array_equal(finalized.logprobs, [4.0, 5.0, 6.0])


def test_concat_joins_memberless_blocks_without_logprobs():
    with_logprobs = _rows_with_logprobs([[1, 2]])

    joined = CompactSampleSupport.concat([CompactSampleSupport.empty(2, 2), with_logprobs])

    assert joined.row_lengths.tolist() == [0, 0, 2]
    np.testing.assert_array_equal(joined.logprobs, [1.0, 2.0])
    with pytest.raises(ValueError, match="must share member columns"):
        CompactSampleSupport.concat([with_logprobs, csr([[3, -1]])])


def test_from_masked_rows_keeps_the_logprobs_of_kept_cells():
    ids = np.array([[5, 6, 7], [8, 9, 10]], dtype=SAMPLE_SUPPORT_DTYPE)
    present = np.array([[False, True, True], [True, False, True]])
    logprobs = np.array([[-np.inf, -0.5, -1.5], [-0.25, -np.inf, -2.0]])

    support = CompactSampleSupport.from_masked_rows(ids, present, logprobs)

    assert support.logprobs.dtype == np.float32
    np.testing.assert_array_equal(support.logprobs, [-0.5, -1.5, -0.25, -2.0])
    assert CompactSampleSupport.from_masked_rows(ids, present).logprobs is None


@pytest.mark.parametrize(
    "logprobs, message",
    [
        (np.zeros(3, dtype=np.float64), "optional float32 logprobs"),
        (np.zeros(2, dtype=np.float32), "columns must align"),
        (np.array([0.0, np.nan, 1.0], dtype=np.float32), "logprobs must be finite"),
    ],
)
def test_compact_support_rejects_malformed_logprobs(logprobs, message):
    with pytest.raises(ValueError, match=message):
        CompactSampleSupport.from_token_ids(
            np.array([1, 2, 3], dtype=SAMPLE_SUPPORT_DTYPE), [2, 1], top_k=2, logprobs=logprobs
        )


def test_candidate_grid_pads_short_and_empty_rows():
    support = csr([[5, 6, -1], [-1, -1, -1], [7, 8, 9]])

    np.testing.assert_array_equal(support.candidate_grid(), [[5, 6, -1], [-1, -1, -1], [7, 8, 9]])


def test_with_teacher_scores_builds_sampled_then_candidate_rows():
    # Row 0 holds its sampled token 6 as a candidate too, row 1 has no captured members, row 2 is short.
    support = csr([[5, 6], [-1, -1], [7, -1]], logprobs=[[-0.5, -1.0], [0.0, 0.0], [-0.25, 0.0]])
    sampled = np.array([6, 3, 4], dtype=SAMPLE_SUPPORT_DTYPE)
    candidates = np.concatenate([sampled[:, None], support.candidate_grid()], axis=1)
    scores = np.array([[-2.0, -1.5, -2.0], [-3.0, -np.inf, -np.inf], [-0.5, -0.75, -np.inf]], dtype=np.float32)

    rows = support.with_teacher_scores(candidates, scores)

    assert rows.top_k == 3
    np.testing.assert_array_equal(dense(rows), [[6, 5, 6], [-1, -1, -1], [4, 7, -1]])
    # The sampled slot's sampler logprob is unused (0); the members keep their own.
    np.testing.assert_array_equal(rows.logprobs, [0.0, -0.5, -1.0, 0.0, -0.25])
    np.testing.assert_array_equal(rows.member_values, [-2.0, -1.5, -2.0, -0.5, -0.75])


def test_member_values_follow_slices_padding_and_concatenation():
    support = csr([[5, 6], [7, -1]], logprobs=[[-0.5, -1.0], [-0.25, 0.0]])
    candidates = np.array([[1, 5, 6], [2, 7, -1]], dtype=SAMPLE_SUPPORT_DTYPE)
    rows = support.with_teacher_scores(candidates, -candidates.astype(np.float32))

    for block in (rows[1:], rows[:1], CompactSampleSupport.concat([rows[1:], rows.empty_like(1), rows[:1]])):
        np.testing.assert_array_equal(block.member_values, -block.token_ids.astype(np.float32))


def test_with_teacher_scores_needs_member_logprobs_and_a_matching_grid():
    with pytest.raises(ValueError, match="sampler's member logprobs"):
        csr([[5, 6]]).with_teacher_scores(np.array([[1, 5, 6]]), np.zeros((1, 3), dtype=np.float32))
    support = csr([[5, 6]], logprobs=[[-0.5, -1.0]])
    with pytest.raises(ValueError, match=r"expected \[1, 3\] candidates"):
        support.with_teacher_scores(np.array([[1, 5]]), np.zeros((1, 2), dtype=np.float32))


@pytest.mark.parametrize(
    "logprobs, member_values, message",
    [
        (None, np.zeros(3, dtype=np.float32), "need logprobs"),
        (np.zeros(3, dtype=np.float32), np.array([0.0, np.inf, 1.0], dtype=np.float32), "values must be finite"),
    ],
)
def test_compact_support_rejects_malformed_member_values(logprobs, member_values, message):
    with pytest.raises(ValueError, match=message):
        CompactSampleSupport.from_token_ids(
            np.array([1, 2, 3], dtype=SAMPLE_SUPPORT_DTYPE),
            [2, 1],
            top_k=2,
            logprobs=logprobs,
            member_values=member_values,
        )


def test_member_values_ride_the_packed_batch():
    support = csr([[5, 6], [7, -1]], logprobs=[[-0.5, -1.0], [-0.25, 0.0]])
    candidates = np.array([[1, 5, 6], [2, 7, -1]], dtype=SAMPLE_SUPPORT_DTYPE)
    rows = support.with_teacher_scores(candidates, -candidates.astype(np.float32))

    packed = PackedRaggedTensor.from_rows([rows, rows[1:], rows.empty_like(2)])

    values = packed_member_values(packed)
    assert values is not None and values.dtype == torch.float32
    np.testing.assert_array_equal(values, -packed.columns[SampleSupportColumn.TOKEN_IDS].float())
    gathered = packed[[1, 0]]
    np.testing.assert_array_equal(
        packed_member_values(gathered), -gathered.columns[SampleSupportColumn.TOKEN_IDS].float()
    )
    assert packed_member_values(PackedRaggedTensor.from_rows([support])) is None
    assert packed_member_values(None) is None
