"""
Tests for DAPO soft overlong punishment with sequence-level and per-token rewards.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest tests/train/algorithms/test_dapo_overlong_punishment.py
"""

from unittest.mock import MagicMock

import pytest

from examples.train.algorithms.dapo.main_dapo import (
    DAPOConfig,
    DAPOTrainer,
    apply_soft_overlong_punishment,
)
from examples.train.algorithms.dapo.main_dapo_fully_async import FullyAsyncDAPOTrainer
from skyrl.train.generators.base import GeneratorOutput

# Responses longer than MAX_RESPONSE_LENGTH - OVERLONG_BUFFER_LEN = 6 tokens are penalized.
MAX_RESPONSE_LENGTH = 10
OVERLONG_BUFFER_LEN = 4
PENALTY_FACTOR = 1.0


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def _make_config(batch_size: int):
    cfg = DAPOConfig()
    cfg.trainer.train_batch_size = batch_size
    cfg.trainer.eval_batch_size = batch_size
    cfg.trainer.resume_mode = "none"
    cfg.trainer.algorithm.overlong_buffer_len = OVERLONG_BUFFER_LEN
    cfg.trainer.algorithm.overlong_buffer_penalty_factor = PENALTY_FACTOR
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.sampling_params.max_generate_length = MAX_RESPONSE_LENGTH
    return cfg


def _make_generator_output(response_lengths, rewards) -> GeneratorOutput:
    return {
        "prompt_token_ids": [[1, 2] for _ in response_lengths],
        "response_ids": [[3] * length for length in response_lengths],
        "rewards": rewards,
        "loss_masks": [[1] * length for length in response_lengths],
        "stop_reasons": ["stop"] * len(response_lengths),
        "rollout_metrics": None,
    }


@pytest.mark.parametrize(
    "response_length, expected_reward",
    [
        (4, 1.0),  # before the overlong buffer
        (6, 1.0),  # at the start of the overlong buffer
        (7, 0.75),  # 1 token into the buffer
        (9, 0.25),  # 3 tokens into the buffer
        (10, 0.0),  # fills the buffer
        (12, 0.0),  # past the max response length
    ],
)
def test_sequence_level_rewards(response_length, expected_reward):
    rewards = apply_soft_overlong_punishment(
        [1.0], [response_length], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert rewards == [pytest.approx(expected_reward)]


def test_sequence_level_rewards_past_max_length_are_zeroed():
    """Rewards past the max response length are set to 0, not penalized from the original reward."""
    rewards = apply_soft_overlong_punishment(
        [-1.0, 2.0], [11, 12], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert rewards == [0.0, 0.0]


def test_sequence_level_rewards_input_not_mutated():
    rewards = [1.0, 1.0]
    penalized = apply_soft_overlong_punishment(
        rewards, [8, 12], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert penalized == [pytest.approx(0.5), 0.0]
    assert rewards == [1.0, 1.0]


@pytest.mark.parametrize(
    "response_length, expected_last_token_reward",
    [(4, 1.0), (6, 1.0), (7, 0.75), (9, 0.25), (10, 0.0)],
)
def test_per_token_rewards_penalize_last_token(response_length, expected_last_token_reward):
    token_rewards = [0.0] * (response_length - 1) + [1.0]
    rewards = apply_soft_overlong_punishment(
        [token_rewards], [response_length], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert len(rewards[0]) == response_length
    assert rewards[0][:-1] == [0.0] * (response_length - 1)
    assert rewards[0][-1] == pytest.approx(expected_last_token_reward)


def test_per_token_rewards_keep_earlier_rewards():
    """Rewards before the last token (e.g. per-turn rewards) are kept; only the last token is penalized."""
    token_rewards = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    rewards = apply_soft_overlong_punishment(
        [token_rewards], [8], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert rewards[0] == pytest.approx([0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])


def test_per_token_rewards_past_max_length_are_zeroed():
    token_rewards = [0.5] * 12
    rewards = apply_soft_overlong_punishment(
        [token_rewards], [12], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )
    assert rewards == [[0.0] * 12]


def test_per_token_rewards_inputs_not_mutated():
    """Penalized per-token rewards are new lists, so lists shared with other outputs keep their values."""
    within_buffer = [0.0] * 7 + [1.0]
    past_max = [0.0] * 11 + [1.0]
    unpenalized = [0.0, 1.0]
    rewards = [within_buffer, past_max, unpenalized]

    penalized = apply_soft_overlong_punishment(
        rewards, [8, 12, 2], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR
    )

    assert penalized[0][-1] == pytest.approx(0.5)
    assert penalized[1] == [0.0] * 12
    assert penalized[2] == [0.0, 1.0]
    assert penalized is not rewards
    assert rewards == [[0.0] * 7 + [1.0], [0.0] * 11 + [1.0], [0.0, 1.0]]


def test_empty_rewards():
    assert apply_soft_overlong_punishment([], [], MAX_RESPONSE_LENGTH, OVERLONG_BUFFER_LEN, PENALTY_FACTOR) == []


def _make_dapo_trainer(cfg) -> DAPOTrainer:
    return DAPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=None,
        train_dataset=DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )


@pytest.mark.parametrize(
    "rewards",
    [
        pytest.param([1.0, 1.0, 1.0], id="sequence_level"),
        pytest.param([[0.0, 1.0], [0.0] * 7 + [1.0], [0.0] * 11 + [1.0]], id="per_token"),
    ],
)
def test_dapo_trainer_postprocess_generator_output(rewards):
    """``DAPOTrainer`` applies the penalty for batched (sequence-level) and non-batched (per-token) rewards."""
    trainer = _make_dapo_trainer(_make_config(batch_size=3))
    generator_output = _make_generator_output([2, 8, 12], rewards)

    result, result_uids = trainer.postprocess_generator_output(generator_output, ["uid1", "uid2", "uid3"])

    assert result_uids == ["uid1", "uid2", "uid3"]
    assert result["rewards"][0] == [0.0, 1.0]
    assert result["rewards"][1] == pytest.approx([0.0] * 7 + [0.5])
    assert result["rewards"][2] == [0.0] * 12
    assert trainer.all_metrics["reward/avg_raw_reward"] == pytest.approx(0.5)


def test_dapo_trainer_rejects_step_wise_trajectories():
    cfg = _make_config(batch_size=1)
    cfg.generator.step_wise_trajectories = True
    trainer = _make_dapo_trainer(cfg)
    with pytest.raises(ValueError, match="step_wise_trajectories"):
        trainer.postprocess_generator_output(_make_generator_output([2], [[0.0, 1.0]]), ["uid1"])


def test_fully_async_dapo_trainer_shared_per_token_rewards():
    """Penalizing the trained output and the metrics output, which share inner reward lists, penalizes each once."""
    # `_apply_soft_overlong_punishment` only reads `cfg`, so the trainer is created without running `__init__`.
    trainer = FullyAsyncDAPOTrainer.__new__(FullyAsyncDAPOTrainer)
    trainer.cfg = _make_config(batch_size=1)
    shared_rewards = [0.0] * 7 + [1.0]
    generator_output = _make_generator_output([8], [shared_rewards])
    metrics_generator_output = _make_generator_output([8, 2], [shared_rewards, [0.0, 1.0]])

    trainer._apply_soft_overlong_punishment(generator_output)
    trainer._apply_soft_overlong_punishment(metrics_generator_output)

    assert generator_output["rewards"] == [pytest.approx([0.0] * 7 + [0.5])]
    assert metrics_generator_output["rewards"][0] == pytest.approx([0.0] * 7 + [0.5])
    assert metrics_generator_output["rewards"][1] == [0.0, 1.0]
    assert shared_rewards == [0.0] * 7 + [1.0]
