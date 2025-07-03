from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
import os
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI


class SQLLLMJudgeEnv(BaseTextEnv):
    """
    Environment for one SQL execution task.
    Use LLM as judge to evaluate the SQL query similarity with the gold SQL query.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Initialize the environment
        assert "reward_spec" in extras, "reward_spec field is required"

        self.gold_sql = extras["reward_spec"]["ground_truth"]

        # Set up OpenAI client
        openai_api_key = env_config.OPENAI_API_KEY
        self.llm_judge_client = OpenAI(api_key=openai_api_key)
        self.model = env_config.model

    def _get_reward(self, action: str) -> float:
        prompt = f"""
            You are a strict SQL evaluation assistant. Compare the two SQL queries below and determine if the predicted query is functionally equivalent to the gold query.

            Only respond with "1" if the predicted SQL query is correct and semantically equivalent to the gold SQL query.
            Otherwise, respond with "0".

            GOLD SQL:
            {self.gold_sql}

            PREDICTED SQL:
            {action}

            Answer with only one character: "1" or "0".
        """
        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
            )
            reply = response.choices[0].message.content.strip()
            return 1.0 if reply == "1" else 0.0
        except Exception as e:
            print(f"LLM Judge error: {type(e).__name__}: {e}")
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=done,
            metadata={},
            postprocessed_action=action,
        )
