import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_LOGPROBS_DTYPE,
    SAMPLE_SUPPORT_LOGPROBS_PADDING,
    SAMPLE_SUPPORT_PADDING,
    SampleSupportLogprobsTrace,
    SampleSupportTrace,
    sample_support_width,
    validate_sample_support_logprobs,
)


def _rows(count: int, first_id: int = 0) -> np.ndarray:
    return np.array([[first_id + i, first_id + i + 100] for i in range(count)], dtype=SAMPLE_SUPPORT_DTYPE)


def test_finalize_drops_exactly_the_declared_trailing_rows():
    trace = SampleSupportTrace()
    trace.append(_rows(3), expected_rows=3)
    trace.append_padding(2)

    support = trace.finalize(token_count=3, extra_rows=2)

    np.testing.assert_array_equal(support, _rows(3))


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


def test_padding_rows_are_all_padding_sentinels():
    trace = SampleSupportTrace()
    trace.append(_rows(1), expected_rows=1)
    trace.append_padding(2)

    support = trace.finalize(token_count=3, extra_rows=0)

    assert support.dtype == SAMPLE_SUPPORT_DTYPE
    np.testing.assert_array_equal(support[1:], np.full((2, 2), SAMPLE_SUPPORT_PADDING, dtype=SAMPLE_SUPPORT_DTYPE))


def _logprob_rows(count: int) -> np.ndarray:
    return np.array([[-0.1 * (i + 1), -1.0 - i] for i in range(count)], dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE)


def test_logprobs_trace_pads_with_negative_infinity_and_drops_trailing_rows():
    trace = SampleSupportLogprobsTrace()
    trace.append(_logprob_rows(2), expected_rows=2)
    trace.append_padding(1)
    trace.append_padding(2)

    logprobs = trace.finalize(token_count=3, extra_rows=2)

    assert logprobs.dtype == SAMPLE_SUPPORT_LOGPROBS_DTYPE
    np.testing.assert_array_equal(logprobs[:2], _logprob_rows(2))
    assert np.all(logprobs[2] == SAMPLE_SUPPORT_LOGPROBS_PADDING)


def test_logprobs_trace_rejects_a_length_mismatch():
    trace = SampleSupportLogprobsTrace()
    trace.append(_logprob_rows(2), expected_rows=2)

    with pytest.raises(ValueError, match="2 rows for 3 tokens plus 0 trailing rows"):
        trace.finalize(token_count=3, extra_rows=0)


def test_validate_sample_support_logprobs_accepts_rows_aligned_with_the_support():
    support = np.array([[10, 11], [12, SAMPLE_SUPPORT_PADDING]], dtype=SAMPLE_SUPPORT_DTYPE)
    logprobs = np.array([[-0.5, -1.5], [-0.2, SAMPLE_SUPPORT_LOGPROBS_PADDING]], dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE)

    assert validate_sample_support_logprobs(logprobs, support) is logprobs


@pytest.mark.parametrize(
    "logprobs, match",
    [
        (np.array([[-0.5, -1.5], [-0.2, -0.3]], dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE), "-inf on padding"),
        (
            np.array(
                [[-0.5, SAMPLE_SUPPORT_LOGPROBS_PADDING], [-0.2, SAMPLE_SUPPORT_LOGPROBS_PADDING]], dtype=np.float32
            ),
            "finite on captured",
        ),
        (np.array([[-0.5, -1.5]], dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE), "does not match support shape"),
        (np.array([[-0.5, -1.5], [-0.2, -0.3]], dtype=np.float64), "float32"),
        (np.array([[np.nan, -1.5], [-0.2, -0.3]], dtype=np.float32), "finite or -inf"),
    ],
)
def test_validate_sample_support_logprobs_rejects_misaligned_rows(logprobs, match):
    support = np.array([[10, 11], [12, SAMPLE_SUPPORT_PADDING]], dtype=SAMPLE_SUPPORT_DTYPE)

    with pytest.raises(ValueError, match=match):
        validate_sample_support_logprobs(logprobs, support)


@pytest.mark.parametrize(
    "sampling_params, expected",
    [
        ({"top_k": 8, "logprobs": 32}, 8),
        ({"top_k": -1, "logprobs": 32}, 32),
        ({"top_k": 1, "logprobs": 32}, 32),
        ({"top_k": -1, "logprobs": None}, 0),
        ({"top_k": -1}, 0),
        ({"top_k": True, "logprobs": 4}, 4),
    ],
)
def test_sample_support_width_prefers_a_truncating_top_k_then_logprobs(sampling_params, expected):
    assert sample_support_width(sampling_params) == expected
