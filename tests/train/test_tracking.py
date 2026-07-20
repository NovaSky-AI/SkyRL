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
    """Numeric metrics are mirrored to skyrl_-prefixed gauges. Strings and bools are skipped."""
    created = {}

    def fake_gauge(name, description=None):
        created[name] = MagicMock()
        return created[name]

    with patch("ray.util.metrics.Gauge", side_effect=fake_gauge):
        t = Tracking(project_name="proj", experiment_name="exp", backend="console")
        t.log({"timing/run_training": 1.5, "generate/n": 2, "note": "text", "flag": True}, step=3)

    assert set(created) == {"skyrl_timing_run_training", "skyrl_generate_n"}
    created["skyrl_timing_run_training"].set.assert_called_with(1.5)


def test_log_gauge_shares_registry_with_mirror():
    """A gauge set via log_gauge and the same-named mirrored metric are one gauge object."""
    created = {}

    def fake_gauge(name, description=None):
        created[name] = MagicMock()
        return created[name]

    with patch("ray.util.metrics.Gauge", side_effect=fake_gauge):
        t = Tracking(project_name="proj", experiment_name="exp", backend="console")
        t.log_gauge("skyrl_trainer_global_step", 5, "step")  # step-start edge
        t.log({"trainer/global_step": 5}, step=5)  # mirror at commit, same name

    assert list(created) == ["skyrl_trainer_global_step"]
    created["skyrl_trainer_global_step"].set.assert_called_with(5.0)
