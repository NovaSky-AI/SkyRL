import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("Answer: \\boxed{42}", "42", 1.0),
        ("Answer: 42", "42", 1.0),
        ("Answer: \\boxed{43}", "42", -1.0),
        ("Answer: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}", 1.0),
        ("Answer: \\boxed{0.5}", "\\frac{1}{2}", -1.0),
        ("Answer: \\boxed{\\text{forty-two}}", "42", -1.0),
        # test EOS tokens
        ("<|im_start|>Answer: \\boxed{42}<|im_end|>", "42", 1.0),
        ("<|im_start|>Answer: \\boxed{42}im_end|>", "42", -1.0),
        ("Answer: \\boxed{42}<|eot_id|>", "42", 1.0),
        ("Answer: \\boxed{42}|eot_id|>", "42", -1.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "aime",
        env_config=DictConfig({"env_class": "aime"}),
        extras={"reward_model": {"method": "rule", "ground_truth": ground_truth}},
    )
    step_output = env.step(output)
    assert step_output["reward"] == expected
