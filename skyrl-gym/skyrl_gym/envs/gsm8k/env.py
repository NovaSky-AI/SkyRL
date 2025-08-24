from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth)

    def _calculate_metrics(self, action: str, reward: float) -> Dict[str, Any]:
        """Calculate environment-specific metrics for training insights."""
        response_length = len(action)
        word_count = len(action.split())
        answer_accuracy = float(reward > 0)
        print("###################### mag gsm8k/env/_calculate_metrics() ############################")
        return {
            "response_length": response_length,
            "word_count": word_count,
            "answer_accuracy": answer_accuracy,
        }
    
    def _aggregate_metrics(self, metrics_list: Dict[str, Any]) -> Dict[str, Any]:
        aggregated = {}
        for key in metrics_list[0].keys():
            aggregated[key] = sum(metric[key] for metric in metrics_list) / len(metrics_list)
        return aggregated

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        metrics = self._calculate_metrics(action, reward)

        print("###################### mag gsm8k/env/step() 1 ############################")
        print("###################### Action:", action, "######################")
        print("###################### Reward:", reward, "######################")
        print("###################### Metrics:", metrics, "######################")

        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={}, metrics=metrics)
    