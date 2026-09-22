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


def convert_vllm_prompt_logprobs(
    prompt_token_ids: list[int],
    raw_prompt_logprobs: list[dict[str, dict] | None] | None,
    topk: int = 0,
) -> tuple[list[float | None] | None, list[list[tuple[int, float]] | None] | None]:
    """Convert vLLM prompt logprobs into the Tinker response shape.

    vLLM returns one entry per prompt token, each a
    ``{str(token_id): {"logprob": float, ...}}`` dict (``None`` at position 0,
    which has no preceding context). Tinker returns a flat list of the prompt
    tokens' own logprobs plus, when ``topk > 0``, a list of ``(token_id,
    logprob)`` pairs per position.

    Args:
        prompt_token_ids: The prompt tokens the logprobs were computed for.
        raw_prompt_logprobs: vLLM's per-position logprob dicts, or None.
        topk: Number of top entries to return per position (0 disables).

    Returns:
        ``(prompt_logprobs, topk_prompt_logprobs)``. Both are None when
        ``raw_prompt_logprobs`` is None; the second is also None when
        ``topk <= 0``.
    """
    if raw_prompt_logprobs is None:
        return None, None

    prompt_logprobs: list[float | None] = [
        (pos_dict.get(str(tid)) or {}).get("logprob") if pos_dict is not None else None
        for tid, pos_dict in zip(prompt_token_ids, raw_prompt_logprobs)
    ]

    if topk <= 0:
        return prompt_logprobs, None

    # vLLM returns k or k+1 logprobs per position (the extra entry is the prompt
    # token when it falls outside the top-k). Tinker returns exactly top-k, so
    # sort by logprob and truncate.
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] = [
        (
            sorted(
                [(int(tid), entry["logprob"]) for tid, entry in pos_dict.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:topk]
            if pos_dict is not None
            else None
        )
        for pos_dict in raw_prompt_logprobs[: len(prompt_token_ids)]
    ]
    return prompt_logprobs, topk_prompt_logprobs


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
