"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

from unittest.mock import MagicMock, patch

from skyrl.train.utils.tracking import Tracking


def test_wandb_init_receives_tags():
    """Tags passed to Tracking are forwarded to wandb.init."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
            tags=["foo", "bar"],
        )

        wandb_mock.init.assert_called_once()
        kwargs = wandb_mock.init.call_args.kwargs
        assert kwargs["tags"] == ["foo", "bar"]
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "exp"


def test_wandb_init_tags_default_none():
    """When tags are not provided, wandb.init receives tags=None."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
        )

        wandb_mock.init.assert_called_once()
        assert wandb_mock.init.call_args.kwargs["tags"] is None


def test_vllm_history_metrics_disable_automatic_summaries_once():
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        tracker = Tracking("proj", "exp", backend="wandb", config={})
        tracker.log({"vllm/train/generation_throughput_tok_s": 50, "train/loss": 1}, step=1)
        tracker.log({"vllm/train/generation_throughput_tok_s": 25}, step=2)
        wandb_mock.define_metric.assert_called_once_with("vllm/train/generation_throughput_tok_s", summary="none")
        assert wandb_mock.log.call_count == 2
        assert wandb_mock.log.call_args.kwargs["data"] == {"vllm/train/generation_throughput_tok_s": 25}
        tracker.finish()


def test_finish_removes_regular_sdk_summary_but_preserves_other_metrics():
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        summary = {"vllm/train/rate": 25, "train/loss": 1}
        wandb_mock.run.summary = summary
        tracker = Tracking("proj", "exp", backend="wandb", config={})
        tracker.log({"vllm/train/rate": 25}, step=1)
        tracker.update_summary({"vllm_correct_aggregate/train/rate": 30})
        tracker.finish()
        assert summary == {"train/loss": 1, "vllm_correct_aggregate/train/rate": 30, "run_status": "success"}
        wandb_mock.Api.assert_not_called()
        wandb_mock.finish.assert_called_once()
