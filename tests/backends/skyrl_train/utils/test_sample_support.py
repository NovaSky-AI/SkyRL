import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_PADDING,
    SampleSupportTrace,
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
    """A trailing row count the caller did not declare means the trace and the response disagree
    about which tokens were sampled, which silent truncation would hide."""
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
