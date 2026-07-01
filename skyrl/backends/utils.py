"""Shared helper utilities for TinkerEngine backends."""

import time
from contextlib import contextmanager

import numpy as np

from skyrl.utils.log import logger


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


def pad_batch_topk(flat_sequences: list[list], lengths: list[int], max_length: int, topk: int, dtype) -> np.ndarray:
    """Pad row-major-flattened (n_i, K) sequences to (batch, max_length, K).

    Used for soft top-K distillation targets/weights, which arrive as
    row-major-flattened 2D tensors. ``flat_sequences[i]`` has length
    ``lengths[i] * topk``.

    Args:
        flat_sequences: Per-example flat (row-major) data of length n_i * topk.
        lengths: Per-example number of tokens n_i.
        max_length: Target sequence length to pad to.
        topk: Number of candidate tokens per position (K).
        dtype: NumPy dtype for the output array.

    Returns:
        A NumPy array of shape (batch_size, max_length, topk).
    """
    batch_size = len(flat_sequences)
    padded = np.zeros((batch_size, max_length, topk), dtype=dtype)
    for i, (flat, n) in enumerate(zip(flat_sequences, lengths)):
        assert n <= max_length, f"Sequence length {n} exceeds max_length {max_length}"
        assert len(flat) == n * topk, f"Expected {n * topk} values for (n={n}, K={topk}), got {len(flat)}"
        padded[i, :n, :] = np.asarray(flat, dtype=dtype).reshape(n, topk)
    return padded


def pad_to_fsdp(arr: np.ndarray, fsdp_size: int) -> np.ndarray:
    """Pad array's first dimension to be divisible by FSDP size."""
    batch_size = arr.shape[0]
    pad_size = (fsdp_size - batch_size % fsdp_size) % fsdp_size
    if pad_size == 0:
        return arr
    return np.pad(arr, [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1))
