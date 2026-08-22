"""
uv run --isolated --extra dev pytest tests/train/dataset/test_parallel_fill.py
"""

import threading

import pytest

from skyrl.train.dataset.parallel_fill import fill_batch_rows


@pytest.mark.parametrize("workers", [None, 1, 4, 64])
def test_every_index_is_filled_exactly_once(workers):
    num_rows = 32
    calls = [0] * num_rows

    def fill_row(index: int) -> None:
        calls[index] += 1

    fill_batch_rows(fill_row, num_rows, workers=workers)

    assert calls == [1] * num_rows


def test_single_worker_runs_serially_on_the_calling_thread():
    order = []
    threads = set()

    def fill_row(index: int) -> None:
        order.append(index)
        threads.add(threading.current_thread())

    fill_batch_rows(fill_row, 4, workers=1)

    assert order == [0, 1, 2, 3]
    assert threads == {threading.current_thread()}


def test_multiple_workers_leave_the_calling_thread():
    threads = set()

    def fill_row(index: int) -> None:
        threads.add(threading.current_thread())

    fill_batch_rows(fill_row, 8, workers=4)

    assert threading.current_thread() not in threads


def test_zero_rows_is_a_no_op():
    def fill_row(index: int) -> None:
        raise AssertionError("fill_row must not be called for an empty batch")

    fill_batch_rows(fill_row, 0)


def test_negative_row_count_raises():
    with pytest.raises(ValueError, match="row count must be non-negative"):
        fill_batch_rows(lambda index: None, -1)


def test_non_positive_worker_count_raises():
    with pytest.raises(ValueError, match="worker count must be positive"):
        fill_batch_rows(lambda index: None, 4, workers=0)


@pytest.mark.parametrize("workers", [1, 4])
def test_callback_exception_propagates(workers):
    def fill_row(index: int) -> None:
        if index == 2:
            raise RuntimeError("row 2 failed")

    with pytest.raises(RuntimeError, match="row 2 failed"):
        fill_batch_rows(fill_row, 8, workers=workers)
