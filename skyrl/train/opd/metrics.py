"""Rollout diagnostics for on-policy distillation, kept free of trainer state so they can be tested."""

import array
import zlib
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport
    from skyrl.train.opd.teacher_client import CandidateScores

REPETITION_TAIL_TOKENS = 2_500
REPETITION_RATIO = 10.0


def truncation_rate(stop_reasons: List[str]) -> float:
    """Fraction of rollouts stopped by the length budget rather than EOS."""
    if not stop_reasons:
        return 0.0
    return sum(reason == "length" for reason in stop_reasons) / len(stop_reasons)


def is_repetitive(token_ids: List[int]) -> bool:
    """Whether the last ``REPETITION_TAIL_TOKENS`` ids, as int32 bytes, compress by more than ``REPETITION_RATIO``."""
    if len(token_ids) < REPETITION_TAIL_TOKENS:
        return False
    tail = array.array("i", token_ids[-REPETITION_TAIL_TOKENS:]).tobytes()
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


def term_metrics(
    advantages: torch.Tensor,
    opd_term: torch.Tensor,
    reverse_kl: torch.Tensor,
    divergence_clipped: torch.Tensor,
    advantage_clipped: torch.Tensor,
    mask: torch.Tensor,
    correct: Optional[torch.Tensor],
) -> Dict[str, float]:
    """The OPD term's metrics over the rows given, one teacher's (see ``apply_opd_to_advantages``)."""
    mask_f = mask.to(dtype=advantages.dtype)
    metrics = {
        "reverse_kl": masked_mean(reverse_kl, mask_f).item(),
        "reverse_kl_abs_max": reverse_kl.abs().max().item() if reverse_kl.numel() else 0.0,
        "adv_rl_abs_mean": masked_mean(advantages.abs(), mask_f).item(),
        "adv_opd_abs_mean": masked_mean(opd_term.abs(), mask_f).item(),
        "divergence_clip_fraction": masked_mean(divergence_clipped.to(mask_f.dtype), mask_f).item(),
        "advantage_clip_fraction": masked_mean(advantage_clipped.to(mask_f.dtype), mask_f).item(),
    }
    log_ratio = -reverse_kl[mask]
    if log_ratio.numel():
        # Percentiles of the raw log-ratio, to pick the advantage bounds from a run. NumPy, because torch.quantile
        # refuses inputs above 2**24 elements and a math step has about 25M tokens.
        low, high = np.quantile(log_ratio.float().cpu().numpy(), [0.01, 0.99])
        metrics["log_ratio_p01"], metrics["log_ratio_p99"] = float(low), float(high)
    if correct is not None:
        # -reverse_kl is log q - log p on trainable tokens and 0 elsewhere, which the mask leaves out anyway.
        value = informativeness(-reverse_kl, mask, correct)
        if value is not None:
            metrics["informativeness"] = value
    return metrics


@dataclass
class TopKStats:
    """Sums over response positions, so rows scored concurrently add up before the means are taken."""

    positions: int = 0
    overlap: float = 0.0
    advantage: float = 0.0
    advantage_positions: int = 0

    def __add__(self, other: "TopKStats") -> "TopKStats":
        return TopKStats(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def metrics(self, prefix: str = "opd/") -> Dict[str, float]:
        """The means, keyed for the tracker under ``prefix``; none until a position was measured."""
        if not self.positions:
            return {}
        metrics = {
            f"{prefix}overlap_ratio": self.overlap / self.positions,
            f"{prefix}overlap_positions": self.positions,
        }
        if self.advantage_positions:
            metrics[f"{prefix}overlap_token_advantage"] = self.advantage / self.advantage_positions
        return metrics


def _renormalized(logprobs: np.ndarray) -> np.ndarray:
    """Each row's log distribution over its finite entries: ``logprobs`` minus their row's ``logsumexp``.

    ``-inf`` entries stay ``-inf``; a row with no finite entry becomes NaN, which callers mask out.
    """
    peak = np.max(logprobs, axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        return logprobs - (np.log(np.exp(logprobs - peak).sum(axis=-1, keepdims=True)) + peak)


def topk_stats(
    student_ids: np.ndarray,
    student_logprobs: np.ndarray,
    teacher_ids: np.ndarray,
    teacher_logprobs: np.ndarray,
    mask: np.ndarray,
) -> TopKStats:
    """One row's overlap sums over its ``mask``-ed response positions (arXiv 2604.13016, Eqs. 6 and 7).

    Each array is ``[tokens, k]``: the student's and the teacher's top ``k`` per position with their logprobs, the
    student's ``-1`` and ``-inf`` past a short row; ``mask`` is ``[tokens]``. The overlap ratio is
    ``|S(p) & S(q)| / k``. The overlap-token advantage is ``(1/|I|) sum_{v in I} p_bar(v) (log q_bar(v) - log
    p_bar(v))``, with both distributions renormalized over the intersection ``I``; positions with an empty ``I``
    are left out of it.
    """
    mask = np.asarray(mask, dtype=bool)
    student_ids, student_logprobs = student_ids[mask], student_logprobs[mask].astype(np.float64)
    teacher_ids, teacher_logprobs = teacher_ids[mask], teacher_logprobs[mask].astype(np.float64)
    matches = student_ids[:, :, None] == teacher_ids[:, None, :]  # [positions, student's k, teacher's k]
    in_both = matches.any(axis=2)
    count = in_both.sum(axis=1)
    nonempty = count > 0
    teacher_on_student = np.take_along_axis(teacher_logprobs, matches.argmax(axis=2), axis=1)
    with np.errstate(invalid="ignore"):
        log_p_bar = _renormalized(np.where(in_both, student_logprobs, -np.inf))
        log_q_bar = _renormalized(np.where(in_both, teacher_on_student, -np.inf))
        cells = np.where(in_both, np.exp(log_p_bar) * (log_q_bar - log_p_bar), 0.0)
    return TopKStats(
        positions=len(count),
        overlap=float((count / student_ids.shape[1]).sum()),
        advantage=float((cells.sum(axis=1)[nonempty] / count[nonempty]).sum()),
        advantage_positions=int(nonempty.sum()),
    )


def row_topk_stats(support: "CompactSampleSupport", scores: "CandidateScores", loss_mask: List[int]) -> TopKStats:
    """One row's overlap sums. A token with no captured members (an appended EOS, an observation token) has no
    student top-K and is left out."""
    if len(support) != len(loss_mask):
        raise ValueError(f"{len(support)} support rows for {len(loss_mask)} response tokens")
    mask = np.asarray(loss_mask, dtype=bool) & (support.row_lengths > 0)
    return topk_stats(support.candidate_grid(), support.logprob_grid(), scores.topk_ids, scores.topk_logprobs, mask)
