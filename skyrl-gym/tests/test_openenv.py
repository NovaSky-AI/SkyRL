
import pytest
import skyrl_gym
from omegaconf import DictConfig
import numpy as np

@pytest.mark.parametrize(
    "expected_reward",
    [
        (-1)
    ]
)
def test_openspiel_action(expected_reward):
    env = skyrl_gym.make(
        "openenv",
        env_config=DictConfig({"env_class": "openenv"}),
        extras={"env_name": "openspiel-env"},
    )
    action_id = env.initial_step_result.observation.legal_actions[0]
    
    for i in range(10):
        if i > 0:
            action_id = result["metadata"]["observation"].legal_actions[0]   
        result = env.step(f"<action>{action_id}</action>")     
        print(f"Reward: {result["metadata"]["observation"].reward}")
        
        if result["metadata"]["observation"].done:
            break 
        
    assert result["metadata"]["observation"].reward == expected_reward, f"Reward {result["metadata"]["observation"]} mismatch with {expected_reward}"
    
@pytest.mark.parametrize(
    "model_response, expected_reward",
    [
        # Correct code: second largest index
        (
            """<action>Hello, World!</action>""",
            1.3
        ),
        (
            """<action>Testing echo environment</action>""",
            2.4000000000000004
        ),
        (
            """<action>One more message</action>""",
            1.6
        )
    ]
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
        (
            """<action>print('Hello, World!')</action>""",
            "Hello, World!\n"
        ),
        (
            """<action>x = 5 + 3\nprint(f'Result: {x}')</action>""",
            "Result: 8\n"
        ),
        (
            """<action>import math\nprint(f'Pi is approximately {math.pi:.4f}')</action>""",
            "Pi is approximately 3.1416\n"
        ),
        (
            """<action># Multi-line calculation\nfor i in range(1, 4):\n    print(f'{i} squared is {i**2}')</action>""",
            "1 squared is 1\n2 squared is 4\n3 squared is 9\n"
        ),
    ]
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
        (
            """<action>2</action>""",
            (210, 160, 3)
        ),
    ]
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
    
    screen = np.array(step_output["metadata"]["observation"].screen).reshape(step_output["metadata"]["observation"].screen_shape)
    
    assert screen.shape == expected_shape, f"Expected {expected_shape}, got {screen.shape}"
