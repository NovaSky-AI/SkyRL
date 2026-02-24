# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
LLM-as-a-Judge environment using a LOCAL vLLM reward model for scoring.

Functionally equivalent to ``examples/llm_as_a_judge``, but replaces the
external OpenAI API with a self-hosted vLLM reward model managed as a
Ray actor (``RewardInferenceService`` from ``reward_inference.py``).

The reward model uses ``FrozenRewardInferenceClient`` (a subclass of
``InferenceEngineClient``) with frozen vLLM engines (no weight sync),
getting automatic load balancing, placement-group GPU scheduling, and
proper Ray lifecycle management.

The reward service is started by the training entrypoint before training
begins (see ``main_llm_judge_local.py``).
"""

from typing import Any, Dict, Union

import re
from dataclasses import dataclass

import ray
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

PROMPT = """
You are a strict math evaluation assistant.

Compare the following **gold** and **predicted** math solutions. Your job is to determine if the predicted solution is mathematically correct and if the predicted solution ends with a line of the form:

#### <number>

You must only give a score of "1" if:
- The final line of the predicted solution **ends with `#### <number>`**, and
- The number **matches the final answer in the gold solution** exactly.

Instructions:
- You may provide internal reasoning or explanation before giving your final judgment.
- Your final judgment must appear as a separate line at the end of your response, in the format:

### Final Score: 1

or

### Final Score: 0

Do not include any explanation after the final score.
"""


@dataclass
class LLMJudgeLocalEnvConfig:
    """Config for the local LLM-as-a-Judge environment."""

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    temperature: float = 0.0
    max_tokens: int = 512


class GSM8kLLMJudgeLocalEnv(BaseTextEnv):
    """GSM8k environment scored by a **local** vLLM LLM-as-a-Judge model.

    Looks up the named Ray actor ``"reward_inference_service"``
    (a ``RewardInferenceService`` wrapping ``FrozenRewardInferenceClient``)
    which provides load balancing, placement-group scheduling, and proper
    Ray lifecycle management.

    Adapt for your own use case by changing ``PROMPT`` and ``_get_reward``.
    """

    def __init__(
        self,
        env_config: Union[LLMJudgeLocalEnvConfig, DictConfig],
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.model = getattr(env_config, "model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.temperature = getattr(env_config, "temperature", 0.0)
        self.max_tokens = getattr(env_config, "max_tokens", 512)

        # Look up the RewardInferenceService actor (started by main_llm_judge_local.py)
        try:
            self._reward_service = ray.get_actor("reward_inference_service")
        except ValueError:
            raise ValueError(
                "Reward inference service not found. "
                "Use main_llm_judge_local.py which auto-starts the service, "
                "or see reward_inference.py for manual setup."
            )

    def _get_reward(self, action: str) -> float:
        message = (
            PROMPT
            + f"\n\nGOLD SOLUTION:\n{self.ground_truth}"
            + f"\n\nPREDICTED SOLUTION:\n{action}"
            + "\n\nAnswer:"
        )

        try:
            messages = [{"role": "user", "content": message}]
            reply = ray.get(
                self._reward_service.score.remote(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )

            # Parse "### Final Score: 0/1" — flexible: accept with or without ###
            match = re.search(
                r"(?:#{1,3}\s*)?Final\s*Score\s*[:=]\s*([01](?:\.0)?)",
                reply,
                re.IGNORECASE,
            )
            if match:
                return float(match.group(1))

            # Fallback: check last non-empty line for bare "0" or "1"
            last_line = reply.strip().rsplit("\n", 1)[-1].strip()
            if last_line in {"1", "0", "1.0", "0.0"}:
                return float(last_line)

            print(f"[LLMJudgeLocal] Unrecognized output: {reply[-200:]}")
            return 0.0

        except Exception as e:
            print(f"[LLMJudgeLocal] Error: {type(e).__name__}: {e}")
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)
        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=done, metadata={}
        )
