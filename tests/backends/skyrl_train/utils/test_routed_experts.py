import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.routed_experts import (
    RoutedExpertTrace,
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


def _turn_routes(num_rows):
    return np.arange(num_rows * 4, dtype=np.int16).reshape(num_rows, 2, 2) % 8


def test_trace_returns_only_the_rows_it_captured():
    """The row count is what tells the trainer where the capture stops, so the trace never
    fabricates rows for the tail it did not cover: collation dummy-fills that tail and the
    router padding mask keeps those rows out of router accounting.
    """
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_turn_routes(4))
    trace.record_generation(prompt_token_count=7, generated_token_count=2, routed_experts=_turn_routes(4))

    routes = trace.finalize(token_count=10, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0])

    assert routes.shape == (8, 2, 2)
    assert np.array_equal(routes, np.concatenate((_turn_routes(4), _turn_routes(4))))


def test_trace_keeps_full_coverage_when_every_token_has_a_route():
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_turn_routes(4))

    routes = trace.finalize(token_count=4, loss_mask=[0, 0, 0, 1])

    assert routes.shape == (4, 2, 2)


def test_trace_rejects_an_uncaptured_loss_active_target():
    """The uncovered tail is only safe because every loss-active target inside it is masked:
    a forced route at a masked position can perturb nothing but later masked positions.
    """
    trace = RoutedExpertTrace()
    trace.record_generation(prompt_token_count=3, generated_token_count=2, routed_experts=_turn_routes(4))

    with pytest.raises(ValueError, match="missing routed-expert row for loss-active target at token 5"):
        trace.finalize(token_count=6, loss_mask=[0, 0, 0, 1, 1, 1])
