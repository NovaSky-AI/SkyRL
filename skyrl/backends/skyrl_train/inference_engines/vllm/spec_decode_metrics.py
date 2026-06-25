# Cumulative speculative-decoding (MTP draft) acceptance accounting for vLLM v1. vLLM's async engine
# has no ``get_metrics()``, so we attach a tiny stat logger in the AsyncLLM frontend that accumulates
# the per-iteration draft/accept counters. The generator turns the cumulative counts into per-step
# acceptance rates -- one overall and one per draft position (grows with num_speculative_tokens).

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


def _add_per_pos(into: list, add) -> None:
    """Elementwise-add a per-position count list into ``into``, growing ``into`` as needed."""
    add = add or []
    if len(add) > len(into):
        into.extend([0] * (len(add) - len(into)))
    for i, n in enumerate(add):
        into[i] += int(n)


def make_spec_decode_stat_logger_class():
    """Return a ``StatLoggerBase`` subclass that sums spec-decode counts.

    Imported lazily and built as a class factory so this module stays importable without vLLM
    (e.g. on CPU-only hosts / unit tests).
    """
    from vllm.v1.metrics.loggers import StatLoggerBase

    class SpecDecodeStatLogger(StatLoggerBase):
        """Accumulate cumulative spec-decode draft/accept counts across scheduler iterations."""

        # Process-global one-shot guard: warn once if vLLM stops exposing `spec_decoding_stats`.
        _warned_missing_spec_field = False

        def __init__(self, vllm_config, engine_index: int = 0):
            self.engine_index = engine_index
            self.num_drafts = 0
            self.num_draft_tokens = 0
            self.num_accepted_tokens = 0
            # Accepted count per draft position (index 0 = first speculated token). vLLM reports a
            # list of length num_speculative_tokens each iteration; we grow on demand instead of
            # pre-sizing so the logger needs no knowledge of the configured depth.
            self.num_accepted_tokens_per_pos: list = []

        def record(self, scheduler_stats=None, iteration_stats=None, mm_cache_stats=None, engine_idx: int = 0):
            if scheduler_stats is None:
                return
            # Absent field (vs. present-but-None = no drafting this iter) means vLLM renamed it; warn
            # once so the metrics don't silently go empty.
            if not hasattr(scheduler_stats, "spec_decoding_stats"):
                if not SpecDecodeStatLogger._warned_missing_spec_field:
                    SpecDecodeStatLogger._warned_missing_spec_field = True
                    logger.warning(
                        "vLLM SchedulerStats has no 'spec_decoding_stats' field; spec-decode (MTP "
                        "draft) acceptance metrics will be empty. vLLM's metrics API likely changed "
                        "-- update inference_engines/vllm/spec_decode_metrics.py."
                    )
                return
            stats = scheduler_stats.spec_decoding_stats
            if stats is not None:
                self.num_drafts += stats.num_drafts
                self.num_draft_tokens += stats.num_draft_tokens
                self.num_accepted_tokens += stats.num_accepted_tokens
                _add_per_pos(self.num_accepted_tokens_per_pos, getattr(stats, "num_accepted_tokens_per_pos", None))

        def log_engine_initialized(self):
            pass

    return SpecDecodeStatLogger


def sum_spec_decode_loggers(loggers) -> Optional[dict]:
    """Sum cumulative counters across a list of ``SpecDecodeStatLogger`` instances."""
    if not loggers:
        return None
    per_pos: list = []
    for lg in loggers:
        _add_per_pos(per_pos, getattr(lg, "num_accepted_tokens_per_pos", None))
    return {
        "num_drafts": sum(int(lg.num_drafts) for lg in loggers),
        "num_draft_tokens": sum(int(lg.num_draft_tokens) for lg in loggers),
        "num_accepted_tokens": sum(int(lg.num_accepted_tokens) for lg in loggers),
        "num_accepted_tokens_per_pos": per_pos,
    }


def merge_spec_decode_counters(totals: dict, stats: dict) -> None:
    """Merge one engine's counter dict into ``totals`` in place (cross-engine aggregation).

    Scalar counters add; per-position lists add elementwise (padding to the longest list, so
    engines configured with different draft depths still merge correctly).
    """
    for key, value in stats.items():
        if isinstance(value, list):
            _add_per_pos(totals.setdefault(key, []), value)
        else:
            totals[key] = totals.get(key, 0) + int(value)


def acceptance_rate_metrics(
    cumulative: Optional[dict], prev: Optional[dict], prefix: str = "vllm/"
) -> tuple[dict, Optional[dict]]:
    """Turn cumulative spec-decode counters into per-step (delta) metrics.

    Args:
        cumulative: counters read this step (from the engines), or None if speculative decoding is
            disabled / unsupported.
        prev: the ``cumulative`` from the previous step (None on the first step).
        prefix: metric-key prefix. ``"vllm/"`` for the train rollout; pass ``"vllm/eval/"`` to
            attribute the eval rollout's draft/accept delta separately.

    Returns:
        ``(metrics, new_prev)`` where ``metrics`` has ``{prefix}draft_*`` keys (empty when there are
        no stats) and ``new_prev`` is the snapshot to pass back next step.
    """
    if not cumulative:
        return {}, prev
    prev = prev or {}
    drafts = cumulative.get("num_drafts", 0) - prev.get("num_drafts", 0)
    drafted = cumulative.get("num_draft_tokens", 0) - prev.get("num_draft_tokens", 0)
    accepted = cumulative.get("num_accepted_tokens", 0) - prev.get("num_accepted_tokens", 0)
    metrics: dict[str, Any] = {
        f"{prefix}draft_num_draft_tokens": drafted,
        f"{prefix}draft_num_accepted_tokens": accepted,
    }
    if drafted > 0:
        # Acceptance rate = accepted draft tokens / total drafted tokens this step.
        metrics[f"{prefix}draft_acceptance_rate"] = accepted / drafted
    # Per-position rates: fraction of draft rounds this step whose k-th speculated token was
    # accepted (same definition vLLM uses in its own per-position logging). Position keys are
    # 1-based: pos_1 = first drafted token. Acceptance halts at the first rejection, so the rates
    # are non-increasing in k — the decay shows how much each extra draft position actually pays.
    per_pos = cumulative.get("num_accepted_tokens_per_pos") or []
    prev_per_pos = prev.get("num_accepted_tokens_per_pos") or []
    if drafts > 0:
        for i, n in enumerate(per_pos):
            prev_n = int(prev_per_pos[i]) if i < len(prev_per_pos) else 0
            metrics[f"{prefix}draft_acceptance_rate_pos_{i + 1}"] = (int(n) - prev_n) / drafts
    return metrics, cumulative
