"""Rollout diagnostics for on-policy distillation, kept free of trainer state so they can be tested."""

import zlib
from typing import List, Optional

import torch

REPETITION_TAIL_CHARS = 10_000
REPETITION_RATIO = 10.0


def truncation_rate(stop_reasons: List[str]) -> float:
    """Fraction of rollouts stopped by the length budget rather than EOS."""
    return sum(reason == "length" for reason in stop_reasons) / len(stop_reasons)


def is_repetitive(text: str) -> bool:
    """Whether the last ``REPETITION_TAIL_CHARS`` characters compress by more than ``REPETITION_RATIO``.

    A shorter text is never repetitive.
    """
    if len(text) < REPETITION_TAIL_CHARS:
        return False
    tail = text[-REPETITION_TAIL_CHARS:].encode("utf-8")
    return len(tail) / len(zlib.compress(tail)) > REPETITION_RATIO


def rollout_correct(reward) -> bool:
    """pass@n's rule: a positive reward, the last token's for per-token rewards."""
    if isinstance(reward, (int, float)):
        return reward > 0
    return len(reward) > 0 and float(reward[-1]) > 0


def informativeness(log_ratio: torch.Tensor, loss_mask: torch.Tensor, correct: torch.Tensor) -> Optional[float]:
    """Mean per-rollout ``log q - log p`` on correct rollouts minus the same on incorrect ones.

    ``log_ratio`` and ``loss_mask`` are ``[rows, tokens]`` and ``correct`` is a bool ``[rows]``. Rows with an all-zero
    mask (batch padding, overlong filtering) are left out. ``None`` until both groups have a rollout.
    """
    mask = loss_mask.to(log_ratio.dtype)
    counts = mask.sum(dim=1)
    means = (log_ratio * mask).sum(dim=1) / counts.clamp(min=1)
    measured = counts > 0
    right, wrong = means[measured & correct], means[measured & ~correct]
    return (right.mean() - wrong.mean()).item() if len(right) and len(wrong) else None
