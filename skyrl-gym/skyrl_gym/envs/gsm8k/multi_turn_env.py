from typing import Dict, Any
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils


class GSM8kMultiTurnEnv(BaseTextEnv):
    """
    Multi-turn GSM8k environment with turn-level rewards.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        reward_spec = extras.get("reward_spec", {})
        assert "ground_truth" in reward_spec, "reward_spec.ground_truth is required"

        self.ground_truth: str = reward_spec["ground_truth"]
        self.max_turns = 5
        if "max_turns" in extras:
            self.max_turns = int(extras["max_turns"])
        elif "max_turns" in extras["extra_info"]:
            self.max_turns = int(extras["extra_info"]["max_turns"])

        format_score = 0.2
        self.format_score_per_turn: float = format_score / self.max_turns

    def init(self, prompt):
        # No special pre-processing; return prompt and empty metadata
        return prompt, {}

    def _make_observation(self) -> list[dict]:
        remaining = self.max_turns - self.turns
        if remaining <= 0:
            return []

        if remaining > 1:
            msg = (
                "Please provide your step-by-step reasoning, "
                "and also include a tentative numeric answer at the end in the exact format: '#### ANSWER'."
            )
        else:
            msg = "Now provide only the final numeric answer in the exact format: '#### ANSWER'."

        return [{"role": "user", "content": msg}]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        # Per-turn reward: 1.0 if correct, 0.2/max_turns if well-formatted but incorrect, 0.0 otherwise.
        reward = utils.compute_score(
            solution_str=action,
            ground_truth=self.ground_truth,
            method="strict",
            format_score=self.format_score_per_turn,
            score=1.0,
        )
        done = self.turns >= self.max_turns or reward == 1.0

        observations = [] if done else self._make_observation()

        metadata = {
            "turns": self.turns,
        }

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata=metadata,
        )
