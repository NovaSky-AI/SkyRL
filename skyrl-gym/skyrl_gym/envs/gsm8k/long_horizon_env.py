"""Long-horizon multi-turn GSM8k: a long agent-style rollout around one GSM8k question.

Each turn the model writes a response; the environment answers with a synthetic ~``obs_tokens``-token "tool
observation" (a deterministic log-like listing), until ``max_turns`` turns. The last observation asks for the final
answer in the GSM8k format, which is scored (1.0 correct, 0.0 otherwise). With ~1,065-token turns (vLLM
``min_tokens``) and ~1,870-token observations over 69 turns, a rollout reaches ~200k tokens: the shape of a long
tool-use agent rollout, for exercising long-context multi-turn RL end to end.
"""

import random
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils

_WORDS = (
    "status ok error retry read write file node value count cache query result table index row column sum total "
    "pending"
).split()
# Measured with the DeepSeek-V4.1 tokenizer: ~16.4 tokens per listing line.
_TOKENS_PER_LINE = 16.4


def synthetic_observation(seed: int, turn: int, n_tokens: int) -> str:
    rng = random.Random(seed * 100003 + turn)
    n_lines = max(1, int(n_tokens / _TOKENS_PER_LINE))
    lines = [f"[tool] step {turn}: environment log follows."]
    for i in range(n_lines):
        a, b, c, d, e = (rng.choice(_WORDS) for _ in range(5))
        lines.append(f"{i:04d} {a} {b} {rng.randint(0, 99999)} {c} {rng.random():.4f} {d} {e}")
    return "\n".join(lines)


class GSM8kLongHorizonEnv(BaseTextEnv):
    """``extras``: ``reward_spec.ground_truth``; ``extra_info.max_turns`` (default 69), ``extra_info.obs_tokens``
    (default 1870), ``extra_info.seed``."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        reward_spec = extras.get("reward_spec", {})
        assert "ground_truth" in reward_spec, "reward_spec.ground_truth is required"
        self.ground_truth: str = reward_spec["ground_truth"]
        info = extras.get("extra_info", {}) or {}
        self.max_turns = int(extras.get("max_turns", info.get("max_turns", 69)))
        self.obs_tokens = int(info.get("obs_tokens", 1870))
        self.seed = int(info.get("seed", info.get("index", 0)))

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        if self.turns >= self.max_turns:
            reward = utils.compute_score(
                solution_str=action, ground_truth=self.ground_truth, method="flexible", format_score=0.0, score=1.0
            )
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={})
        msg = synthetic_observation(self.seed, self.turns, self.obs_tokens)
        if self.turns == self.max_turns - 1:
            msg += "\n\nThis is the last step: give only the final numeric answer in the exact format '#### ANSWER'."
        else:
            msg += "\n\nContinue working on the problem."
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": msg}], reward=0.0, done=False, metadata={}
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"steps": self.turns}

    @staticmethod
    def aggregate_metrics(metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        return {"avg_steps": sum(float(m.get("steps", 0)) for m in metrics) / len(metrics)}
