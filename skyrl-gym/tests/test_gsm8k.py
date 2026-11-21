import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        # Correct answer in the GSM8K-required format.
        ("The answer is #### 42", "42", 1.0),

        # Correct format, but the extracted answer does not match ground truth.
        ("The answer is #### 42", "43", 0.0),

        # Numerically correct, but missing the required "#### <answer>" format.
        ("The answer is 42", "42", 0.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    """Verify that GSM8K's rule-based reward checks both format and correctness."""

    # Create a GSM8K environment configured to score the model output
    # against the supplied ground-truth answer using the rule-based reward.
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k"}),
        extras={
            "reward_spec": {
                "method": "rule",
                "ground_truth": ground_truth,
            }
        },
    )

    # env.step() evaluates the model's response and returns the reward.
    # No separate env.init() call is needed because this test only exercises
    # the scoring behavior and does not depend on initialized episode state.
    step_output = env.step(output)

    # A valid, correct answer should receive 1.0; all other cases receive 0.0.
    assert step_output["reward"] == expected
