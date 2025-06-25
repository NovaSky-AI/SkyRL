import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("<query>Who is the president of France?</query> <answer>Emmanuel Macron</answer>", "Emmanuel Macron", 1.0),
        ("<query>Who is the president of France?</query> <answer>Hollande</answer>", "Emmanuel Macron", 0.0),
        ("<query>President?</query>", "Emmanuel Macron", 0.0),  # No <answer> so not done yet
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig({"env_class": "search"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": {"target": ground_truth}}, "max_turns": 1},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(output)

    # Only assert reward when done is True
    if step_output["done"]:
        assert step_output["reward"] == expected
    else:
        assert step_output["reward"] == 0.0


@pytest.mark.parametrize(
    "output, parse_ground_truth, ground_truth",
    [
        ("<query>Who is the president of France?</query>", ["Who is the president of France?"], "Emmanuel Macron"),
        ("<query>Who is the president of France?", [None], "Emmanuel Macron"),
        ("<qu>President?</qu>", [None], "Emmanuel Macron"),
    ],
)
def test_tool_parsing(output, parse_ground_truth, ground_truth):
    # Test the parsing logic and reward logic with dummy input
    env = skyrl_gym.make(
        "search",
        env_config=DictConfig({"env_class": "search"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": {"target": ground_truth}}, "max_turns": 2},
    )
    # Skip init() since it's not used in this test

    # Step once and get the tool input in `info`a
    _, _, tool_input = env._parse_action(output)

    # assert it matches the parsed ground truth
    assert tool_input == parse_ground_truth
