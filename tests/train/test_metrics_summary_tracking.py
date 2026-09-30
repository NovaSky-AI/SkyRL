"""Tests for writing summaries before tracker shutdown."""

from types import SimpleNamespace
from unittest.mock import Mock

from skyrl.train.utils.tracking import Tracking


def test_failure_status_and_summary_precede_finish():
    tracker = Tracking("test", "test", backend="console")
    tracker.backend = "wandb"
    events = []
    summary = Mock()
    summary.update.side_effect = lambda data: events.append(("summary", data))
    logger = Mock()
    logger.run = SimpleNamespace(summary=summary)
    logger.finish.side_effect = lambda **kwargs: events.append(("finish", kwargs))
    tracker.logger = logger
    tracker.set_summary_provider(lambda: {"vllm_correct_aggregate/train/output_tokens_total": 10})
    tracker.finish(exit_code=1)
    tracker.finish(exit_code=1)
    assert events == [
        ("summary", {"vllm_correct_aggregate/train/output_tokens_total": 10, "run_status": "failed"}),
        ("finish", {"exit_code": 1}),
    ]
