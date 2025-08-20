from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.aime import utils
from typing import Dict, Any
from omegaconf import DictConfig


class AIMEEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = extras["reward_model"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        score_info = utils.compute_score(action, self.ground_truth)
        reward = score_info["score"]
        metadata = {"acc": score_info["acc"], "pred": score_info["pred"]}

        # No observation in aime, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
