# Cumulative speculative-decoding (MTP draft) acceptance accounting for vLLM v1.
#
# vLLM's async engine (the one SkyRL uses for generation + weight sync) does not expose
# ``get_metrics()``, so we attach a tiny custom stat logger that runs in the AsyncLLM frontend
# process (= the SkyRL engine actor) and accumulates the per-iteration spec-decode counters
# (``num_draft_tokens`` / ``num_accepted_tokens``). The engine reads these cumulative counters and
# the generator turns them into a per-step acceptance rate (accepted / drafted).

from __future__ import annotations

from typing import Optional


def make_spec_decode_stat_logger_class():
    """Return a ``StatLoggerBase`` subclass that sums spec-decode counts.

    Imported lazily and built as a class factory so this module stays importable without vLLM
    (e.g. on CPU-only hosts / unit tests).
    """
    from vllm.v1.metrics.loggers import StatLoggerBase

    class SpecDecodeStatLogger(StatLoggerBase):
        """Accumulate cumulative spec-decode draft/accept counts across scheduler iterations."""

        def __init__(self, vllm_config, engine_index: int = 0):
            self.engine_index = engine_index
            self.num_drafts = 0
            self.num_draft_tokens = 0
            self.num_accepted_tokens = 0

        def record(self, scheduler_stats=None, iteration_stats=None, mm_cache_stats=None, engine_idx: int = 0):
            stats = getattr(scheduler_stats, "spec_decoding_stats", None) if scheduler_stats is not None else None
            if stats is not None:
                self.num_drafts += stats.num_drafts
                self.num_draft_tokens += stats.num_draft_tokens
                self.num_accepted_tokens += stats.num_accepted_tokens

        def log_engine_initialized(self):
            pass

    return SpecDecodeStatLogger


def sum_spec_decode_loggers(loggers) -> Optional[dict]:
    """Sum cumulative counters across a list of ``SpecDecodeStatLogger`` instances."""
    if not loggers:
        return None
    return {
        "num_drafts": sum(int(lg.num_drafts) for lg in loggers),
        "num_draft_tokens": sum(int(lg.num_draft_tokens) for lg in loggers),
        "num_accepted_tokens": sum(int(lg.num_accepted_tokens) for lg in loggers),
    }


def acceptance_rate_metrics(cumulative: Optional[dict], prev: Optional[dict]) -> tuple[dict, Optional[dict]]:
    """Turn cumulative spec-decode counters into per-step (delta) metrics.

    Args:
        cumulative: counters read this step (from the engines), or None if speculative decoding is
            disabled / unsupported.
        prev: the ``cumulative`` from the previous step (None on the first step).

    Returns:
        ``(metrics, new_prev)`` where ``metrics`` has ``vllm/draft_*`` keys (empty when there are no
        stats) and ``new_prev`` is the snapshot to pass back next step.
    """
    if not cumulative:
        return {}, prev
    prev = prev or {}
    drafted = cumulative.get("num_draft_tokens", 0) - prev.get("num_draft_tokens", 0)
    accepted = cumulative.get("num_accepted_tokens", 0) - prev.get("num_accepted_tokens", 0)
    metrics = {
        "vllm/draft_num_draft_tokens": drafted,
        "vllm/draft_num_accepted_tokens": accepted,
    }
    if drafted > 0:
        # Acceptance rate = accepted draft tokens / total drafted tokens this step.
        metrics["vllm/draft_acceptance_rate"] = accepted / drafted
    return metrics, cumulative
