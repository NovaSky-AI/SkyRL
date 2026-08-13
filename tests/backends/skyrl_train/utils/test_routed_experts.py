import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.routed_experts import (
    compact_routed_expert_indices,
)


@pytest.mark.parametrize(
    "routes,expected_dtype",
    [
        (np.arange(12).reshape(3, 2, 2), np.uint8),
        (np.array([[[2**8 - 1]]]), np.uint8),
        (np.array([[[0, 2**8]]]), np.int16),
        (np.array([[[0, 2**15 - 1]]]), np.int16),
        (np.array([[[0, 2**15]]]), np.int32),
        (np.array([[[0, 2**31 - 1]]], dtype=np.int64), np.int32),
        (np.empty((0, 2, 2), dtype=np.int64), np.uint8),
    ],
)
def test_compaction_picks_smallest_safe_dtype(routes, expected_dtype):
    compact = compact_routed_expert_indices(routes)

    assert compact.dtype == expected_dtype
    assert compact.flags.c_contiguous
    assert np.array_equal(compact, routes)


def test_compaction_makes_read_only_arrays_writable():
    routes = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    routes.flags.writeable = False

    compact = compact_routed_expert_indices(routes)

    assert compact.dtype == np.uint8
    assert compact.flags.c_contiguous
    assert compact.flags.writeable


def test_compaction_copies_non_contiguous_input():
    compact = compact_routed_expert_indices(np.arange(24).reshape(6, 2, 2)[::2])

    assert compact.flags.c_contiguous
    assert np.array_equal(compact, np.arange(24).reshape(6, 2, 2)[::2])


def test_compaction_rejects_nested_lists():
    with pytest.raises(TypeError, match="NumPy array"):
        compact_routed_expert_indices([[[1, 2]]])


@pytest.mark.parametrize(
    "routes",
    [
        np.array([1, 2]),  # not 3-D
        np.array([[[1.0]]]),  # not integral
        np.array([[[-1]]]),  # negative expert id
        np.array([[[2**31]]], dtype=np.uint64),  # exceeds int32
    ],
)
def test_compaction_rejects_invalid_routes(routes):
    with pytest.raises(ValueError):
        compact_routed_expert_indices(routes)
