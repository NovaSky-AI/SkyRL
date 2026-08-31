from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any


class GSM8kMultiRewardEnv(BaseTextEnv):
    """GSM8k scored as two objectives (``format`` and ``correct``); the reward is their sum.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = int(extras.get("max_turns", 1))

    def _get_reward_components(self, action: str) -> Dict[str, float]:
        return utils.compute_score_components(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        reward_components = self._get_reward_components(action)
        done = self.turns >= self.max_turns or reward_components["correct"] > 0

        observations = (
            []
            if done
            else [
                {
                    "role": "user",
                    "content": "That is not correct. Try again, ending with the answer in the exact "
                    "format: '#### ANSWER'.",
                }
            ]
        )

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=sum(reward_components.values()),
            reward_components=reward_components,
            done=done,
            metadata={},
        )
