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


def test_log_mirrors_numeric_metrics_to_prometheus():
    """Every numeric metric is mirrored as a skyrl_-prefixed gauge; non-numerics and bools are not."""
    created = {}

    def fake_gauge(name, description=None):
        created[name] = MagicMock()
        return created[name]

    with patch("ray.util.metrics.Gauge", side_effect=fake_gauge):
        t = Tracking(project_name="proj", experiment_name="exp", backend="console")
        t.log({"timing/run_training": 1.5, "generate/n": 2, "note": "text", "flag": True}, step=3)
        t.log({"timing/run_training": 2.5}, step=4)

    assert set(created) == {"skyrl_timing_run_training", "skyrl_generate_n"}
    created["skyrl_timing_run_training"].set.assert_called_with(2.5)
    created["skyrl_generate_n"].set.assert_called_with(2.0)
