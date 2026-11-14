"""
Tests for the ADAPO adaptive clipping controller and its trainer integration.

uv run --isolated --extra dev -- pytest tests/train/algorithms/test_adapo.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import ADAPOController
from skyrl.train.config import ADAPOConfig
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.utils import validate_cfg
from tests.train.util import example_dummy_config


def test_adapo_first_update_sets_target_entropy():
    controller = ADAPOController(ADAPOConfig(enabled=True))
    controller.update(1.5)
    assert controller.target_entropy == 1.5
    assert controller.eps_clip_high == pytest.approx(0.28)


def test_adapo_increases_when_entropy_below_target():
    controller = ADAPOController(ADAPOConfig(enabled=True, target_entropy=1.0))
    controller.update(0.5)
    assert controller.eps_clip_high == pytest.approx(0.28 * 1.05)
    for _ in range(20):
        controller.update(0.5)
    assert controller.eps_clip_high == pytest.approx(0.32)


def test_adapo_decreases_when_entropy_above_target():
    controller = ADAPOController(ADAPOConfig(enabled=True, target_entropy=1.0))
    controller.update(2.0)
    assert controller.eps_clip_high == pytest.approx(0.28 * 0.95)
    for _ in range(20):
        controller.update(2.0)
    assert controller.eps_clip_high == pytest.approx(0.2)


def test_adapo_unchanged_at_target():
    controller = ADAPOController(ADAPOConfig(enabled=True, target_entropy=1.0))
    controller.update(1.0)
    assert controller.eps_clip_high == pytest.approx(0.28)


def test_adapo_state_dict_round_trip():
    controller = ADAPOController(ADAPOConfig(enabled=True))
    controller.update(1.0)
    controller.update(0.5)

    restored = ADAPOController(ADAPOConfig(enabled=True))
    restored.load_state_dict(controller.state_dict())
    assert restored.eps_clip_high == controller.eps_clip_high
    assert restored.target_entropy == controller.target_entropy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"eps_clip_high_init": 0.1},
        {"eps_clip_high_init": 0.4},
        {"eps_clip_high_min": 0.3, "eps_clip_high_max": 0.25},
        {"increase_factor": 1.0},
        {"decrease_factor": 1.0},
        {"decrease_factor": 0.0},
    ],
)
def test_adapo_config_validation(kwargs):
    with pytest.raises(ValueError):
        ADAPOConfig(enabled=True, **kwargs)


def test_adapo_rejects_losses_without_eps_clip_high():
    cfg = example_dummy_config()
    cfg.trainer.algorithm.adapo.enabled = True
    cfg.trainer.algorithm.policy_loss_type = "cispo"
    with pytest.raises(ValueError, match="adapo"):
        validate_cfg(cfg)


def _make_trainer(cfg) -> RayPPOTrainer:
    return RayPPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )


def _adapo_trainer(ckpt_path: str = None, resume_mode: str = "none") -> RayPPOTrainer:
    cfg = example_dummy_config()
    if ckpt_path is not None:
        cfg.trainer.ckpt_path = ckpt_path
    cfg.trainer.resume_mode = resume_mode
    cfg.trainer.algorithm.adapo.enabled = True
    cfg.trainer.algorithm.adapo.target_entropy = 1.0
    cfg.trainer.update_epochs_per_batch = 1
    return _make_trainer(cfg)


def _training_batch() -> TrainingInputBatch:
    data = TrainingInputBatch(
        {
            "advantages": torch.randn(4, 3),
            "loss_mask": torch.ones(4, 3),
            "response_mask": torch.ones(4, 3),
        }
    )
    data.metadata = {"policy_mini_batch_boundaries": [(0, 4)], "global_step": 0}
    return data


def test_trainer_sends_adapo_eps_clip_high_and_updates_it():
    trainer = _adapo_trainer()
    trainer.dispatch = MagicMock()
    trainer.dispatch.stage_data.return_value = [["chunk_ref"]]
    trainer.dispatch.forward_backward_from_staged.return_value = SimpleNamespace(
        metrics={"policy_loss": 0.1, "policy_entropy": 0.5}
    )
    trainer.dispatch.optim_step.return_value = None

    with patch.object(trainer, "_normalize_advantages", side_effect=lambda data, *args: data):
        trainer.train_critic_and_policy(_training_batch())

    _, kwargs = trainer.dispatch.forward_backward_from_staged.call_args
    assert kwargs["loss_fn_config"] == {"eps_clip_high": pytest.approx(0.28)}
    assert trainer.all_metrics["policy/adapo_eps_clip_high"] == pytest.approx(0.28)
    # Entropy 0.5 is below the target of 1.0, so the next step uses a larger clip bound.
    assert trainer.adapo_controller.eps_clip_high == pytest.approx(0.28 * 1.05)


def test_trainer_checkpoint_round_trips_adapo_state(tmp_path):
    trainer = _adapo_trainer(ckpt_path=str(tmp_path))
    trainer.dispatch = MagicMock()
    trainer.train_dataloader = MagicMock()
    trainer.train_dataloader.state_dict.return_value = {}
    trainer.global_step = 5
    trainer.adapo_controller.load_state_dict({"eps_clip_high": 0.31, "target_entropy": 0.7})

    with patch.object(trainer, "_cleanup_old_checkpoints"):
        trainer.save_checkpoints()

    resumed = _adapo_trainer(ckpt_path=str(tmp_path), resume_mode="latest")
    resumed.dispatch = MagicMock()
    resumed.train_dataloader = MagicMock()
    with patch("skyrl.train.trainer.validate_consistency_for_latest_checkpoint"):
        resumed.load_checkpoints()

    assert resumed.adapo_controller.state_dict() == {"eps_clip_high": 0.31, "target_entropy": 0.7}
