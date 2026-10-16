"""Checkpoint publication and retention with delayed worker writes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.sft_config import SFTConfig
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.trainer import RayPPOTrainer

TRAINERS = [
    (RayPPOTrainer, False),
    (RayPPOTrainer, True),
    (FullyAsyncRayPPOTrainer, False),
    (FullyAsyncRayPPOTrainer, True),
    (SFTTrainer, False),
]


class DelayedCheckpointDispatch:
    """A worker whose checkpoint becomes readable only after finalization."""

    def __init__(self, root, fail_role=None):
        self.root = root
        self.fail_role = fail_role
        self.pending = {}
        self.completed = []

    def save_checkpoint(self, role, directory, tokenizer):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pending[role] = directory

    def finalize_pending_saves(self, role):
        # The last completed checkpoint must remain usable while writes are pending,
        # including while the critic finishes after a successful policy write.
        assert (self.root / "latest_ckpt_global_step.txt").read_text() == "10"
        assert (self.root / "global_step_10" / "policy" / "metadata.json").exists()
        if role == self.fail_role:
            raise OSError(f"{role} checkpoint write failed")
        directory = self.pending.pop(role)
        (directory / "metadata.json").write_text("{}")
        self.completed.append(role)


def make_trainer(tmp_path, monkeypatch, trainer_cls, has_critic, fail_role=None):
    old_policy = tmp_path / "global_step_10" / "policy"
    old_policy.mkdir(parents=True)
    (old_policy / "metadata.json").write_text("{}")
    (tmp_path / "latest_ckpt_global_step.txt").write_text("10")

    trainer = object.__new__(trainer_cls)
    trainer.global_step = 20
    trainer.tokenizer = None
    trainer.train_dataloader = SimpleNamespace(state_dict=lambda: {"position": 20})
    trainer.dispatch = DelayedCheckpointDispatch(tmp_path, fail_role)

    if trainer_cls is SFTTrainer:
        trainer.sft_cfg = SFTConfig(ckpt_path=str(tmp_path), max_ckpts_to_keep=1)
        trainer._checkpoint_dataloader_state = None
    else:
        trainer.cfg = SkyRLTrainConfig()
        trainer.cfg.trainer.ckpt_path = str(tmp_path)
        trainer.cfg.trainer.max_ckpts_to_keep = 1
        trainer.cfg.trainer.critic.model.path = "critic" if has_critic else None
        trainer.all_timings = {}
        trainer._node_ids = ["test-node"]
        # Keep the real local retention path; no remote workers are needed for filesystem checks.
        monkeypatch.setattr("skyrl.train.trainer.run_on_each_node", lambda *args: None)

    if trainer_cls is FullyAsyncRayPPOTrainer:
        trainer.epoch = 2
        trainer.async_train_dataloader = SimpleNamespace(
            get_consumed_uids_list=lambda: ["trained", "filtered"],
            get_filtered_uids_list=lambda: ["filtered"],
        )
    return trainer


def save_checkpoint(trainer):
    return trainer.save_checkpoint() if isinstance(trainer, SFTTrainer) else trainer.save_checkpoints()


@pytest.mark.parametrize("trainer_cls,has_critic", TRAINERS)
def test_checkpoint_published_only_after_all_models_complete(tmp_path, monkeypatch, trainer_cls, has_critic):
    trainer = make_trainer(tmp_path, monkeypatch, trainer_cls, has_critic)

    checkpoint = Path(save_checkpoint(trainer))

    assert trainer.dispatch.completed == (["policy", "critic"] if has_critic else ["policy"])
    assert not trainer.dispatch.pending
    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "20"
    assert not (tmp_path / "global_step_10").exists()
    assert (checkpoint / "policy" / "metadata.json").exists()
    assert torch.load(checkpoint / "trainer_state.pt", weights_only=False)["global_step"] == 20
    assert torch.load(checkpoint / "data.pt", weights_only=False) == {"position": 20}
    if has_critic:
        assert (checkpoint / "critic" / "metadata.json").exists()
    if trainer_cls is FullyAsyncRayPPOTrainer:
        assert torch.load(checkpoint / "fully_async_state.pt", weights_only=False) == {
            "consumed_uids": ["trained", "filtered"],
            "filtered_uids": ["filtered"],
            "epoch": 2,
        }


@pytest.mark.parametrize(
    "trainer_cls,has_critic,fail_role",
    [
        (cls, has_critic, role)
        for cls, has_critic in TRAINERS
        for role in (["policy", "critic"] if has_critic else ["policy"])
    ],
)
def test_failed_model_write_preserves_previous_checkpoint(tmp_path, monkeypatch, trainer_cls, has_critic, fail_role):
    trainer = make_trainer(tmp_path, monkeypatch, trainer_cls, has_critic, fail_role)

    with pytest.raises(OSError, match=f"{fail_role} checkpoint write failed"):
        save_checkpoint(trainer)

    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "10"
    assert (tmp_path / "global_step_10" / "policy" / "metadata.json").exists()
    assert not (tmp_path / "global_step_20" / fail_role / "metadata.json").exists()


def test_failed_fully_async_state_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, monkeypatch, FullyAsyncRayPPOTrainer, False)
    real_save = torch.save

    def fail_async_state_save(value, file):
        if str(file.name).endswith("fully_async_state.pt"):
            raise OSError("async state write failed")
        return real_save(value, file)

    monkeypatch.setattr(torch, "save", fail_async_state_save)

    with pytest.raises(OSError, match="async state write failed"):
        trainer.save_checkpoints()

    assert (tmp_path / "latest_ckpt_global_step.txt").read_text() == "10"
    assert (tmp_path / "global_step_10" / "policy" / "metadata.json").exists()
