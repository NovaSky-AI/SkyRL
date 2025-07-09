from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
import os

PROMPT = """
    You are a strict math evaluation assistant.

    Compare the following **gold** and **predicted** math solutions. 
    Determine if the predicted solution follows valid reasoning and reaches the correct final answer, even if the explanation differs in wording.

    Rules:
    - Only answer "1" if the predicted solution is mathematically correct and leads to the same final answer as the gold solution.
    - Otherwise, answer "0".
    - Do not include any explanation or extra text—output only a single character: "1" or "0".
"""


class GSM8kLLMJudgeEnv(BaseTextEnv):
    """
    Example implementtion of GSM8k environment with LLM as judge.

    Use LLM as judge to evaluate the answer similarity with the ground truth.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        # Set up OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_judge_client = OpenAI(api_key=openai_api_key)
        self.model = env_config.model

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
