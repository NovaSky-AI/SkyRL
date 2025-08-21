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
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

    def _parse_action(self, action: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", action)
        return match.group(1) if match else None

    # TODO: Extend this method to allow for multiple turns.
    def step(self, action: str) -> BaseTextEnvStepOutput:
        answer = self._parse_action(action)
        is_correct = answer is not None and answer.strip() == str(self.ground_truth).strip()

        # TODO:
        # Check whether we have reached the maximum number of turns.
        # If not, provide some feedback to the agent. For example, if `_parse_action` returns `None`, you can provide feedback like "Please provide your answer in the format \\boxed{your_answer}."

        return BaseTextEnvStepOutput(
            observations=[], reward=1.0 if is_correct else 0.0, done=True, metadata={"parsed_answer": answer}
        )
