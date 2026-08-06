import tarfile
from unittest.mock import MagicMock

import pytest

from skyrl.backends.skyrl_train.workers.worker import Worker
from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_checkpoint_restores_requested_training_state(tmp_path, optimizer):
    checkpoint_path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(checkpoint_path, "w"):
        pass

    backend = object.__new__(SkyRLTrainBackend)
    backend._model_ids_to_role = {"model_test": "policy"}
    backend._dispatch = MagicMock()

    backend.load_checkpoint(str(checkpoint_path), "model_test", optimizer=optimizer)

    backend._dispatch.load_checkpoint.assert_called_once()
    call = backend._dispatch.load_checkpoint.call_args
    assert call.kwargs["load_optimizer_states"] is optimizer
    assert call.kwargs["load_lr_scheduler_states"] is optimizer


def test_weights_only_load_does_not_reset_live_optimizer():
    worker = object.__new__(Worker)
    worker.model = MagicMock()
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.strategy = MagicMock()
    worker.strategy.load_checkpoint.return_value = ("/checkpoint", {})
    live_optimizer = worker.optimizer
    live_scheduler = worker.scheduler

    Worker.load_checkpoint(
        worker,
        "/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
    assert worker.optimizer is live_optimizer
    assert worker.scheduler is live_scheduler

    worker.strategy.load_checkpoint.assert_called_once_with(
        model=worker.model,
        optimizer=None,
        scheduler=None,
        ckpt_dir="/checkpoint",
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
    )
