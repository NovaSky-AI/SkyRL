"""Shared helper utilities for TinkerEngine backends."""

import time
from contextlib import contextmanager

import numpy as np

from tx.utils.log import logger


@contextmanager
def log_timing(request: str):
    """Context manager to log execution time for a request."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        logger.info(f"(timing) {request} took {elapsed:.3f}s")


def pad(xs, pad_to: int, *, fill):
    """Pad a list to a specified length with a fill value."""
    return xs + ([fill] * (pad_to - len(xs)))


def pad_batch(sequences: list[list], max_length: int, dtype) -> np.ndarray:
    """Pad a batch of sequences to max_length.

    Args:
        sequences: List of sequences to pad.
        max_length: Target length for all sequences.
        dtype: NumPy dtype for the output array.

    Returns:
        A NumPy array of shape (batch_size, max_length) with the padded sequences.
    """
    batch_size = len(sequences)
    padded = np.zeros((batch_size, max_length), dtype=dtype)
    for i, seq in enumerate(sequences):
        assert len(seq) <= max_length, f"Sequence length {len(seq)} exceeds max_length {max_length}"
        padded[i, : len(seq)] = seq
    return padded


def pad_to_fsdp(arr: np.ndarray, fsdp_size: int) -> np.ndarray:
    """Pad array's first dimension to be divisible by FSDP size."""
    batch_size = arr.shape[0]
    pad_size = (fsdp_size - batch_size % fsdp_size) % fsdp_size
    if pad_size == 0:
        return arr
    return np.pad(arr, [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1))
