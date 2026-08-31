from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any


class GSM8kMultiRewardEnv(BaseTextEnv):
    """GSM8k scored as three separate objectives (``correct``, ``format``, ``length``); the reward
    is their sum.

    Rewards judge the final response and are awarded once per trajectory, so intermediate turns of
    a multi-turn rollout score zero.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = int(extras.get("max_turns", 1))
        self.max_response_chars = int(extras.get("max_response_chars", 1024))
        self.condition_length_on_correct = bool(extras.get("condition_length_on_correct", False))

    def _get_reward_components(self, action: str) -> Dict[str, float]:
        return utils.compute_score_components(
            action,
            self.ground_truth,
            max_response_chars=self.max_response_chars,
            condition_length_on_correct=self.condition_length_on_correct,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        reward_components = self._get_reward_components(action)
        done = self.turns >= self.max_turns or reward_components["correct"] > 0

        if not done:
            return BaseTextEnvStepOutput(
                observations=[
                    {
                        "role": "user",
                        "content": "That is not correct. Try again, ending with the answer in the "
                        "exact format: '#### ANSWER'.",
                    }
                ],
                reward=0.0,
                reward_components={name: 0.0 for name in reward_components},
                done=False,
                metadata={},
            )

        return BaseTextEnvStepOutput(
            observations=[],
            reward=sum(reward_components.values()),
            reward_components=reward_components,
            done=True,
            metadata={},
        )
