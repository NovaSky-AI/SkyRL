import pytest
import skyrl_gym
from omegaconf import DictConfig
import numpy as np


@pytest.mark.parametrize(
    "game_name, max_steps",
    [
        ("catch", 5),
        ("tic_tac_toe", 10),
    ],
)
def test_openspiel_games(game_name, max_steps):
    """Test OpenSpielEnv integration for both single- and multi-player games."""

    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "openspiel-env", "game_name": game_name},
    )

    first_obs = env.initial_step_result.observation
    assert hasattr(first_obs, "legal_actions"), "Missing legal_actions in initial observation"
    assert len(first_obs.legal_actions) > 0, "No legal actions available at start"

    action_id = first_obs.legal_actions[0]
    result = None

    for step in range(max_steps):
        result = env.step(f"<action>{action_id}</action>")
        obs = result["metadata"]["observation"]

        assert hasattr(obs, "reward"), f"Step {step}: Missing reward in observation"
        assert hasattr(obs, "done"), f"Step {step}: Missing done flag in observation"

        if obs.done:
            break

        if hasattr(obs, "legal_actions") and obs.legal_actions:
            action_id = obs.legal_actions[0]

    assert result is not None, "No step result obtained"
    assert isinstance(obs.reward, (int, float, type(None))), "Reward must be numeric or None"
    assert hasattr(obs, "info_state"), "Observation missing info_state field"
    assert hasattr(obs, "game_phase"), "Observation missing game_phase field"


@pytest.mark.parametrize(
    "model_response, expected_reward",
    [
        # Correct code: second largest index
        ("""<action>Hello, World!</action>""", 1.3),
        ("""<action>Testing echo environment</action>""", 2.4000000000000004),
        ("""<action>One more message</action>""", 1.6),
    ],
)
def test_echoenv_compute_score(model_response, expected_reward):
    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "echo_env"},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert step_output["reward"] == expected_reward


@pytest.mark.parametrize(
    "model_response, expected_observation",
    [
        # Correct code: second largest index
        ("""<action>print('Hello, World!')</action>""", "Hello, World!\n"),
        ("""<action>x = 5 + 3\nprint(f'Result: {x}')</action>""", "Result: 8\n"),
        (
            """<action>import math\nprint(f'Pi is approximately {math.pi:.4f}')</action>""",
            "Pi is approximately 3.1416\n",
        ),
        (
            """<action># Multi-line calculation\nfor i in range(1, 4):\n    print(f'{i} squared is {i**2}')</action>""",
            "1 squared is 1\n2 squared is 4\n3 squared is 9\n",
        ),
    ],
)
def test_codingenv_exec_code(model_response, expected_observation):
    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "coding_env"},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert "observation" in step_output["metadata"], "observation is required in metadata"
    assert step_output["metadata"]["observation"].stdout == expected_observation


@pytest.mark.parametrize(
    "model_response, expected_shape",
    [
        ("""<action>2</action>""", (210, 160, 3)),
    ],
)
def test_atari_action(model_response, expected_shape):
    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "atari-env"},
    )

    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert "observation" in step_output["metadata"], "observation is required in metadata"

    screen = np.array(step_output["metadata"]["observation"].screen).reshape(
        step_output["metadata"]["observation"].screen_shape
    )

    assert screen.shape == expected_shape, f"Expected {expected_shape}, got {screen.shape}"


@pytest.mark.parametrize("model_response", [("<action>1</action>")])
def test_sumo_rl_action(model_response):
    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "sumo-rl-env"},
    )
    step_output = env.step(model_response)

    assert "observation" in step_output["metadata"], "Observation missing in metadata"
    obs = step_output["metadata"]["observation"]

    assert hasattr(obs, "sim_time"), "Missing sim_time in observation"
    assert hasattr(obs, "reward"), "Missing reward in observation"
    assert isinstance(obs.sim_time, (int, float)), "sim_time must be numeric"
    assert isinstance(obs.reward, (int, float, type(None))), "reward must be numeric or None"
