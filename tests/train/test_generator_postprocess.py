"""
Test for token-level rewards support in RayPPOTrainer.postprocess_generator_output method.

Run with:
uv run --isolated --extra dev pytest tests/train/test_generator_postprocess.py
"""

from unittest.mock import MagicMock

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.trainer import RayPPOTrainer


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def create_config(batch_size):
    cfg = SkyRLTrainConfig()
    cfg.trainer.train_batch_size = batch_size
    cfg.trainer.eval_batch_size = batch_size
    cfg.trainer.resume_mode = "none"
    cfg.trainer.seed = 42
    cfg.trainer.epochs = 1
    cfg.generator.n_samples_per_prompt = 1
    return cfg


def test_response_level_rewards():
    """Test postprocess_generator_output with response-level rewards (List[float])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": [1.0],  # Response-level reward
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 0.0, 1.0]]

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": [1.0, 0.5],  # Response-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])

    # Verify conversion to per-token rewards
    assert result["rewards"] == [[0.0, 1.0], [0.0, 0.0, 0.5]]


def test_token_level_rewards():
    """Test postprocess_generator_output with token-level rewards (List[List[float]])."""

    # Test length=1
    config = create_config(1)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.2, 0.3]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2]],
        "response_ids": [[3, 4, 5]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1, 1]],
        "stop_reasons": ["stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards

    # Test length=2
    config = create_config(2)
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    per_token_rewards = [[0.1, 0.3], [0.2, 0.1, 0.1]]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8, 9]],
        "rewards": per_token_rewards,  # Token-level rewards
        "loss_masks": [[1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
    }

    result = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2"])

    # Verify token-level rewards are unchanged
    assert result["rewards"] == per_token_rewards


def test_stepwise_zero_variance_filter_masks_entire_trajectory():
    config = create_config(2)
    config.generator.step_wise_trajectories = True
    config.trainer.algorithm.zero_variance_filter = True

    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )

    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1], [1, 2], [3], [3, 4], [5], [5, 6], [7], [7, 8]],
        "response_ids": [[10], [11], [12], [13], [14], [15], [16], [17]],
        "rewards": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "loss_masks": [[1], [1], [1], [1], [1], [1], [1], [1]],
        "stop_reasons": ["tool_call", "stop", "tool_call", "stop", "tool_call", "stop", "tool_call", "stop"],
        "rollout_metrics": None,
        "rollout_logprobs": None,
        "trajectory_ids": [
            TrajectoryID(instance_id="uid-good", repetition_id=0),
            TrajectoryID(instance_id="uid-good", repetition_id=0),
            TrajectoryID(instance_id="uid-good", repetition_id=1),
            TrajectoryID(instance_id="uid-good", repetition_id=1),
            TrajectoryID(instance_id="uid-bad", repetition_id=0),
            TrajectoryID(instance_id="uid-bad", repetition_id=0),
            TrajectoryID(instance_id="uid-bad", repetition_id=1),
            TrajectoryID(instance_id="uid-bad", repetition_id=1),
        ],
        "is_last_step": [False, True, False, True, False, True, False, True],
    }

    result = trainer.postprocess_generator_output(
        generator_output,
        ["uid-good", "uid-good", "uid-good", "uid-good", "uid-bad", "uid-bad", "uid-bad", "uid-bad"],
    )

    assert result["loss_masks"] == [[1], [1], [1], [1], [0], [0], [0], [0]]
    assert result["rewards"] == [[0.0], [0.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
