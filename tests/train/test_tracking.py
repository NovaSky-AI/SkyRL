"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

import gc
import signal
import weakref
from unittest.mock import MagicMock, patch

import pytest

from skyrl.train.utils.tracking import Tracking


@pytest.fixture
def mlflow_mock():
    """Provide a mocked mlflow module with no active run so the adapter starts one."""
    mlflow = MagicMock()
    mlflow.active_run.return_value = None
    with patch.dict("sys.modules", {"mlflow": mlflow}):
        yield mlflow


@pytest.fixture(autouse=True)
def restore_sigterm():
    """Tracking installs a SIGTERM handler; restore the default after each test."""
    original = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, original)


def test_wandb_init_receives_tags():
    """Tags passed to Tracking are forwarded to wandb.init."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backends=["wandb"],
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
            backends=["wandb"],
            config={},
        )

        wandb_mock.init.assert_called_once()
        assert wandb_mock.init.call_args.kwargs["tags"] is None


def test_mlflow_finish_default_status_finished(mlflow_mock):
    """Tracking.finish() ends the MLflow run with FINISHED by default."""
    tracker = Tracking(project_name="proj", experiment_name="exp", backends=["mlflow"], config=None)

    tracker.finish()

    mlflow_mock.end_run.assert_called_once_with(status="FINISHED")


def test_mlflow_finish_forwards_status(mlflow_mock):
    """A status passed to Tracking.finish is forwarded to mlflow.end_run."""
    tracker = Tracking(project_name="proj", experiment_name="exp", backends=["mlflow"], config=None)

    tracker.finish(status="KILLED")

    mlflow_mock.end_run.assert_called_once_with(status="KILLED")


def test_mlflow_finish_is_idempotent(mlflow_mock):
    """Repeated finish() calls (signal + atexit + __del__) only end the run once."""
    tracker = Tracking(project_name="proj", experiment_name="exp", backends=["mlflow"], config=None)

    tracker.finish(status="KILLED")
    tracker.finish()

    mlflow_mock.end_run.assert_called_once_with(status="KILLED")


def test_sigterm_handler_finishes_run_as_killed(mlflow_mock):
    """A SIGTERM after Tracking init ends the MLflow run with status KILLED.

    Regression test for runs left in RUNNING when the process is killed by a
    signal (e.g. SLURM scancel) rather than exiting cleanly via __del__.
    """
    tracker = Tracking(project_name="proj", experiment_name="exp", backends=["mlflow"], config=None)

    # Tracking should have installed its SIGTERM handler.
    handler = signal.getsignal(signal.SIGTERM)
    assert handler == tracker._sigterm_handler

    # Previous disposition was the default; chaining would re-raise SIGTERM and
    # kill the test process, so swap it for a no-op before invoking the handler.
    tracker._previous_sigterm_handler = lambda *args: None
    handler(signal.SIGTERM, None)

    mlflow_mock.end_run.assert_called_once_with(status="KILLED")


def test_finish_releases_instance_for_gc(mlflow_mock):
    """After finish(), the atexit/SIGTERM hooks must not keep the instance alive."""
    tracker = Tracking(project_name="proj", experiment_name="exp", backends=["mlflow"], config=None)
    ref = weakref.ref(tracker)
    tracker.finish()

    del tracker
    gc.collect()
    assert ref() is None
