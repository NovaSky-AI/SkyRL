from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import re


class MultiplyEnv(BaseTextEnv):
    """
    Environment for multiplication.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.num_turns = 0
        self.max_turns = 1

    def _parse_action(self, action: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", action)
        return match.group(1) if match else None

    # TODO: Implement this method.
    def step(self, action: str) -> BaseTextEnvStepOutput:
        # TODO: Check whether the action is correct, and return the appropriate reward.
        raise NotImplementedError("This environment is not implemented")
