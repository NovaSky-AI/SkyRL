"""
uv run --extra dev --isolated pytest tests/train/generators/test_generator_output_utils.py
"""

import numpy as np

from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
)


def test_generator_output_concatenation():
    # First ensure that the GeneratorOutput fields are what we expect
    expected_fields = [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
        "rollout_logprobs",
        "rollout_expert_indices",
        # optional but present in the signature
        "trajectory_ids",
        "is_last_step",
        "pixel_values",
        "image_grid_thw",
    ]
    assert set(GeneratorOutput.__annotations__.keys()) == set(expected_fields), (
        "GeneratorOutput fields are not what we expect. "
        "Please update the test and `concatenate_generator_outputs()` to reflect the new fields."
        "It is needed to help Trainer.eval() record the full GeneratorOutput information."
    )

    generator_output_1: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.1, 0.2], [0.3, 0.4]],
    }

    generator_output_2: GeneratorOutput = {
        "prompt_token_ids": [[5, 6, 7], [8]],
        "response_ids": [[5, 6, 7], [8]],
        "rewards": [2.0, 3.0],
        "loss_masks": [[1, 1, 1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.5, 0.6, 0.7], [0.8]],
    }

    generator_outputs = [generator_output_1, generator_output_2]
    concatenated_output = concatenate_generator_outputs(generator_outputs)

    assert concatenated_output["prompt_token_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["response_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["rewards"] == [1.0, 2.0, 2.0, 3.0]
    assert concatenated_output["loss_masks"] == [[1, 1], [1, 1], [1, 1, 1], [1]]
    assert concatenated_output["stop_reasons"] == ["stop", "stop", "stop", "stop"]
    assert concatenated_output["rollout_logprobs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7], [0.8]]

    # Validate rollout metrics
    expected_rollout_metrics = {
        "generate/min_num_tokens": 1,
        "generate/max_num_tokens": 3,
        "generate/avg_num_tokens": 2.0,
        "generate/std_num_tokens": np.std([2, 2, 3, 1]).item(),
        "generate/avg_tokens_non_zero_rewards": 2.0,
        "generate/avg_tokens_zero_rewards": 0,
    }
    assert concatenated_output["rollout_metrics"].keys() == expected_rollout_metrics.keys()
    for key, value in expected_rollout_metrics.items():
        np.testing.assert_allclose(concatenated_output["rollout_metrics"][key], value)


def test_get_metrics_from_generator_output():
    # Per trajectory rewards, where rewards are List[float]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 1.5
    assert metrics["pass_at_n"] == 1.0
    assert metrics["mean_positive_reward"] == 1.5

    # Per token rewards, where rewards are List[List[float]], so for pass_at_n we use the last
    # token's reward to signify the trajectory's reward
    generator_output["rewards"] = [[1.0, 0.0], [0.0, 1.0]]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 1.0
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 1.0

    # Mixed rewards with some negative rewards
    generator_output["rewards"] = [-1.0, 2.0]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 0.5
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 1.0

    # Mixed per-token rewards with negatives - per-token rewards
    generator_output["rewards"] = [[1.0, -1.0], [-0.5, 0.5]]
    uids = ["a", "b"]
    metrics = get_metrics_from_generator_output(generator_output, uids)
    assert metrics["avg_score"] == 0.0
    assert metrics["pass_at_n"] == 0.5
    assert metrics["mean_positive_reward"] == 0.75
