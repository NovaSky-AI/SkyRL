from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
import re as _re

# Matches an integer/decimal (optionally signed, with thousands separators), e.g. "1,800", "-5", "72.0".
_NUM_RE = _re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")


def _norm_num(s: str) -> str:
    """Normalize a parsed number to compare against the (comma-free integer) ground truth."""
    return s.strip().rstrip(".").replace(",", "").replace("$", "")


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        # Reasoning models (e.g. Nemotron-3-Ultra) emit `<reasoning></think><answer>` and may write
        # the final answer in their own style (boxed / natural language) rather than the exact
        # `#### <number>` the strict parser wants. Drop the `<think>` trace (so its intermediate
        # numbers can't be mistaken for the answer), then reward the exact `#### <number>` format if
        # present (strict), else fall back to the last number in the answer (robust to boxed /
        # natural-language answers and comma/$ formatting). Non-reasoning models (no `</think>`) keep
        # the original strict scoring on the full output, so this is a no-op for them.
        answer_segment = action.split("</think>")[-1] if "</think>" in action else action
        reward = utils.compute_score(answer_segment, self.ground_truth, method="strict")
        if reward == 0.0:
            nums = _NUM_RE.findall(answer_segment)
            if nums and _norm_num(nums[-1]) == _norm_num(str(self.ground_truth)):
                reward = 1.0
        return reward

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
