from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any


class GSM8kMultiRewardEnv(BaseTextEnv):
    """GSM8k scored as two objectives (``format`` and ``correct``) rather than one scalar, for
    multi-reward algorithms such as GDPO. The reward is their sum."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward_components(self, action: str) -> Dict[str, float]:
        return utils.compute_score_components(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward_components = self._get_reward_components(action)
        return BaseTextEnvStepOutput(
            observations=[],
            reward=sum(reward_components.values()),
            reward_components=reward_components,
            done=done,
            metadata={},
        )
