"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

from unittest.mock import MagicMock, call, patch

from loguru import logger

from skyrl.train.utils.tracking import Tracking, _WandbAdapter


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


def test_wandb_adapter_declares_global_step_axis():
    """The adapter declares the custom step axis once, at init, and hides it from the panel grid."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(project_name="proj", experiment_name="exp", backend="wandb", config={})

        run = wandb_mock.init.return_value
        run.define_metric.assert_any_call(_WandbAdapter.STEP_METRIC, hidden=True)
        run.define_metric.assert_any_call("*", step_metric=_WandbAdapter.STEP_METRIC)


def test_wandb_adapter_injects_step_and_never_passes_step_kwarg():
    """Every write carries the step key and commits its own row, on the run object. ``step=`` is
    never passed to wandb, so a later write at an older step is not subject to the monotonic
    ``_step`` rule."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        tracker = Tracking(project_name="proj", experiment_name="exp", backend="wandb", config={})

        tracker.log({"trainer/loss": 0.1}, step=47, commit=True)  # the deprecated kwarg is ignored
        tracker.log({"eval/score": 0.5}, step=40)

        assert wandb_mock.init.return_value.log.call_args_list == [
            call({_WandbAdapter.STEP_METRIC: 47, "trainer/loss": 0.1}, commit=True),
            call({_WandbAdapter.STEP_METRIC: 40, "eval/score": 0.5}, commit=True),
        ]
        wandb_mock.log.assert_not_called()  # run-scoped, never the module-level log


def test_wandb_adapter_is_inert_without_a_run():
    """If ``wandb.init`` hands back None (a mocked or disabled wandb), the adapter logs nothing
    and nothing crashes."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        wandb_mock.init.return_value = None
        tracker = Tracking(project_name="proj", experiment_name="exp", backend="wandb", config={})

        tracker.log({"a": 1.0}, step=1)
        tracker.finish()

        wandb_mock.log.assert_not_called()
        wandb_mock.finish.assert_not_called()


def test_log_warns_once_when_commit_is_passed():
    """The deprecated ``commit`` kwarg is ignored and warned about once per tracker, not per call."""
    records = []
    handler_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        tracker = Tracking(project_name="proj", experiment_name="exp", backend="console")
        tracker.log({"a": 1.0}, step=1, commit=True)
        tracker.log({"a": 2.0}, step=2, commit=False)
        tracker.log({"a": 3.0}, step=3)
    finally:
        logger.remove(handler_id)

    assert len(records) == 1
    assert "commit" in records[0]["message"] and "deprecated" in records[0]["message"]
    # opt(depth=1): attributed to the caller's line, not to tracking.py.
    assert records[0]["function"] == "test_log_warns_once_when_commit_is_passed"


def test_tensorboard_and_console_log_at_given_step(monkeypatch, tmp_path):
    """Step-tolerant backends log at the step they are given, in any order."""
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path))
    with patch("torch.utils.tensorboard.SummaryWriter") as writer_cls:
        tracker = Tracking(project_name="proj", experiment_name="exp", backend="tensorboard")
        tracker.log({"trainer/loss": 0.1}, step=47)
        tracker.log({"eval/score": 0.5}, step=40)
        assert writer_cls.return_value.add_scalar.call_args_list == [
            call("trainer/loss", 0.1, 47),
            call("eval/score", 0.5, 40),
        ]

    Tracking(project_name="proj", experiment_name="exp", backend="console").log({"eval/score": 0.5}, step=40)
