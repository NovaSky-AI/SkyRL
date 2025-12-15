"""Tests for data types used in SkyRLGymGenerator

uv run --isolated --extra dev -- pytest -s tests/cpu/generators/test_datatypes.py
"""

import pytest

from skyrl_train.generators.skyrl_gym_generator import TurnOutput


@pytest.mark.parametrize(
    "output_ids, observation_ids, output_logprobs, added_eos, expected_loss_mask, expected_logprobs",
    [
        # `added_eos` is False - `loss_mask` value is 1.0 for all output tokens. expected logprobs should have value 0.0 for observation tokens
        ([1, 2, 3, 4], [100, 101], [0.9, 0.8, 0.7, 0.6], False, [1, 1, 1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.0, 0.0]),
        # `added_eos` is True - `loss_mask` should mask out the last token in `output_ids`. expected logprobs behaviour doesn't change
        (
            [1, 2, 3, 4],
            [100, 101, 102],
            [0.9, 0.8, 0.7, 0.5],
            True,
            [1, 1, 1, 0, 0, 0, 0],
            [0.9, 0.8, 0.7, 0.5, 0.0, 0.0, 0.0],
        ),
    ],
)
def test_turn_output(output_ids, observation_ids, output_logprobs, added_eos, expected_loss_mask, expected_logprobs):
    turn_output = TurnOutput(
        output="Dummy",
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        new_obs=[],
        obs_ids=observation_ids,
        added_eos=added_eos,
        reward=1.0,
    )

    # test loss mask
    assert turn_output.get_turn_loss_mask() == expected_loss_mask

    # test rollout logprobs
    assert turn_output.get_turn_rollout_logprobs() == expected_logprobs
