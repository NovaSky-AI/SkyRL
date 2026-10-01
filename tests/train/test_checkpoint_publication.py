"""Check checkpoint publication and recovery with native serializers on local storage."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.trainer_utils import ResumeMode, cleanup_old_checkpoints


def _trainer(tmp_path, trainer_type=RayPPOTrainer):
    trainer = object.__new__(trainer_type)
    trainer.cfg = SkyRLTrainConfig()
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    trainer.cfg.trainer.max_ckpts_to_keep = 1
    trainer.cfg.trainer.critic.model.path = None
    trainer.resume_mode = ResumeMode.FROM_PATH
    trainer.global_step = 1
    trainer._dataloader_epoch = 0
    trainer.epoch = 2
    trainer.tokenizer = None
    trainer.all_timings = {}
    trainer.train_dataloader = StatefulDataLoader(
        list(range(6)), batch_size=2, generator=torch.Generator().manual_seed(42)
    )
    trainer.async_train_dataloader = MagicMock()
    trainer.async_train_dataloader.get_consumed_uids_list.return_value = ["trained", "filtered"]
    trainer.async_train_dataloader.get_filtered_uids_list.return_value = ["filtered"]
    trainer.dispatch = MagicMock()
    # Exercise real retention locally without allocating Ray workers.
    trainer._cleanup_old_checkpoints = lambda: cleanup_old_checkpoints(str(tmp_path), 1)
    return trainer


class DeferredCheckpointDispatch:
    """Leave model files absent until the caller drains their pending writes."""

    def __init__(self, failing_model=None):
        self.pending = {}
        self.completed = []
        self.failing_model = failing_model

    def save_checkpoint(self, model, directory, tokenizer):
        self.pending[model] = Path(directory)

    def finalize_pending_saves(self, model):
        directory = self.pending[model]
        checkpoint = directory.parent
        assert (checkpoint / "data.pt").is_file()
        assert (checkpoint / "trainer_state.pt").is_file()
        assert (checkpoint.parent / "latest_ckpt_global_step.txt").read_text() == "1"
        if model == self.failing_model:
            raise OSError(f"{model} write failed")
        directory.mkdir()
        torch.save({"model": model}, directory / "model.pt")
        del self.pending[model]
        self.completed.append(model)


@pytest.mark.parametrize("has_critic", [False, True])
def test_publication_waits_for_current_model_writes(tmp_path, has_critic):
    trainer = _trainer(tmp_path)
    previous = Path(trainer.save_checkpoints())
    trainer.global_step = 2
    trainer.cfg.trainer.critic.model.path = "critic" if has_critic else None
    trainer.dispatch = DeferredCheckpointDispatch()
    models = ["policy", "critic"] if has_critic else ["policy"]

    def publish(path):
        for model in models:
            state = torch.load(Path(path, model, "model.pt"), weights_only=False)
            assert state == {"model": model}
        assert trainer.dispatch.pending == {}
        assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "2"
        assert previous.is_dir()

    trainer._on_checkpoint_saved = publish

    trainer.save_checkpoints()

    assert trainer.dispatch.completed == models
    assert not previous.exists()


@pytest.mark.parametrize("failing_model", ["policy", "critic"])
def test_pending_model_failure_keeps_previous_checkpoint_unpublished(tmp_path, failing_model):
    trainer = _trainer(tmp_path)
    previous = Path(trainer.save_checkpoints())
    trainer.global_step = 2
    trainer.cfg.trainer.critic.model.path = "critic"
    trainer.dispatch = DeferredCheckpointDispatch(failing_model)
    trainer._on_checkpoint_saved = MagicMock()

    with pytest.raises(OSError, match=f"{failing_model} write failed"):
        trainer.save_checkpoints()

    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "1"
    assert previous.is_dir()
    trainer._on_checkpoint_saved.assert_not_called()
    assert trainer.dispatch.completed == (["policy"] if failing_model == "critic" else [])


@pytest.mark.parametrize("failure", ["state", "write"])
def test_dataloader_save_failure_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    trainer = _trainer(tmp_path)
    previous = Path(trainer.save_checkpoints())
    trainer.global_step = 2
    published = MagicMock()
    trainer._on_checkpoint_saved = published
    if failure == "state":
        monkeypatch.setattr(trainer.train_dataloader, "state_dict", MagicMock(side_effect=OSError("data failed")))
    else:
        original_open = io.open_file

        def open_file(path, mode="rb"):
            if path.endswith("global_step_2/data.pt"):
                raise OSError("data failed")
            return original_open(path, mode)

        monkeypatch.setattr(io, "open_file", open_file)

    with pytest.raises(OSError, match="data failed"):
        trainer.save_checkpoints()

    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "1"
    assert previous.is_dir()
    published.assert_not_called()


def test_async_state_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, FullyAsyncRayPPOTrainer)
    previous = Path(trainer.save_checkpoints())
    trainer.global_step = 2
    trainer.dispatch.save_checkpoint.reset_mock()
    original_open = io.open_file

    def open_file(path, mode="rb"):
        if path.endswith("global_step_2/fully_async_state.pt"):
            raise OSError("async state failed")
        return original_open(path, mode)

    monkeypatch.setattr(io, "open_file", open_file)
    with pytest.raises(OSError, match="async state failed"):
        trainer.save_checkpoints()

    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "1"
    assert previous.is_dir()
    trainer.dispatch.save_checkpoint.assert_not_called()


def test_publication_failure_keeps_previous_checkpoint_until_next_publication(tmp_path):
    trainer = _trainer(tmp_path, FullyAsyncRayPPOTrainer)
    previous = Path(trainer.save_checkpoints())
    trainer.global_step = 2
    events = []

    def save_policy(*args):
        # Snapshotting consumed UIDs must precede a blocking distributed save.
        state = torch.load(tmp_path / "global_step_2/fully_async_state.pt", weights_only=False)
        assert state == {"consumed_uids": ["trained", "filtered"], "filtered_uids": ["filtered"], "epoch": 2}
        events.append("policy")

    def fail_publication(path):
        assert Path(path, "data.pt").is_file()
        assert Path(path, "trainer_state.pt").is_file()
        assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "2"
        assert previous.is_dir()
        events.append("publish")
        raise OSError("receipt failed")

    trainer.dispatch.save_checkpoint.side_effect = save_policy
    trainer._on_checkpoint_saved = fail_publication
    with pytest.raises(OSError, match="receipt failed"):
        trainer.save_checkpoints()
    assert previous.is_dir()
    assert events == ["policy", "publish"]

    trainer._on_checkpoint_saved = MagicMock()
    path = trainer.save_checkpoints()
    trainer._on_checkpoint_saved.assert_called_once_with(path)
    assert not previous.exists()
    assert Path(path).is_dir()


@pytest.mark.parametrize("prefix", ["", "s3://bucket/checkpoints/", "gs://bucket/checkpoints/"])
@pytest.mark.parametrize("trailing_slash", ["", "/"])
def test_resume_preserves_remote_locator_and_dataloader_position(tmp_path, monkeypatch, prefix, trailing_slash):
    trainer = _trainer(tmp_path)
    assert next(iter(trainer.train_dataloader)).tolist() == [0, 1]
    checkpoint_path = trainer.save_checkpoints()
    locator = f"{prefix}global_step_1" if prefix else checkpoint_path
    resumed = _trainer(tmp_path)
    resumed.cfg.trainer.resume_path = locator + trailing_slash
    if prefix:
        original_open, original_exists = io.open_file, io.exists

        def local_path(path):
            assert path.startswith(prefix), f"remote locator was changed: {path}"
            return str(tmp_path / path.removeprefix(prefix))

        monkeypatch.setattr(io, "open_file", lambda path, mode="rb": original_open(local_path(path), mode))
        monkeypatch.setattr(io, "exists", lambda path: original_exists(local_path(path)))

    assert resumed.load_checkpoints() == (1, locator)
    assert next(iter(resumed.train_dataloader)).tolist() == [2, 3]
    resumed.dispatch.load_checkpoint.assert_called_once_with(
        "policy", os.path.join(locator, "policy"), load_optimizer_states=True, load_lr_scheduler_states=True
    )


@pytest.mark.parametrize("resume_path", [None, ""])
def test_from_path_requires_a_nonempty_locator(tmp_path, resume_path):
    trainer = _trainer(tmp_path)
    trainer.cfg.trainer.resume_path = resume_path
    with pytest.raises(ValueError, match="resume_path.*must be specified"):
        trainer.load_checkpoints()


def test_dataloader_restore_failure_stops_before_loading_policy(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.cfg.trainer.resume_path = trainer.save_checkpoints()
    monkeypatch.setattr(trainer.train_dataloader, "load_state_dict", MagicMock(side_effect=ValueError("bad state")))
    with pytest.raises(ValueError, match="bad state"):
        trainer.load_checkpoints()
    trainer.dispatch.load_checkpoint.assert_not_called()


def test_async_resume_keeps_its_epoch_and_consumed_uid_state(tmp_path):
    trainer = _trainer(tmp_path, FullyAsyncRayPPOTrainer)
    checkpoint = trainer.save_checkpoints()
    resumed = _trainer(tmp_path, FullyAsyncRayPPOTrainer)
    resumed.cfg.trainer.resume_path = checkpoint
    assert resumed.load_checkpoints() == (1, checkpoint, {"trained", "filtered"}, {"filtered"}, 2)
    assert resumed._dataloader_epoch == 0


@pytest.mark.parametrize("epoch", [None, -1, True, 0.5])
def test_resume_rejects_missing_or_invalid_dataloader_epoch(tmp_path, epoch):
    trainer = _trainer(tmp_path)
    checkpoint = Path(trainer.save_checkpoints())
    trainer.cfg.trainer.resume_path = str(checkpoint)
    path = checkpoint / "trainer_state.pt"
    state = torch.load(path, weights_only=False)
    if epoch is None:
        del state["dataloader_epoch"]
    else:
        state["dataloader_epoch"] = epoch
    torch.save(state, path)
    with pytest.raises(ValueError, match="dataloader_epoch"):
        trainer.load_checkpoints()
    trainer.dispatch.load_checkpoint.assert_not_called()
