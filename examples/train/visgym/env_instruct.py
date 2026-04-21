from typing import Any, Dict, List, Tuple

import gymnasium

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from examples.train.visgym.utils import extract_relaxed_action, make_image_message, VALID_ACTIONS


_FORMAT_INSTRUCTION = (
    "\n\nIMPORTANT: You must end your response with your chosen action "
    "in this exact format: <action>ACTION</action> "
    "where ACTION is one of: left, right, up, down, stop."
)


class VisGymEnv(BaseTextEnv):
    """Relaxed VisGym wrapper that uses keyword actions instead of tuples.

    The model can generate free-form reasoning, then must end with
    ``<action>keyword</action>`` where keyword is one of:
    left, right, up, down, stop.

    Reward is task-only (no format-reward component); the instruct recipe
    relies on KL regularization rather than a shaped format reward.

    Configuration via extras dict:
        visgym_env_id (str): VisGym environment ID, e.g. "maze_2d/easy"
        seed (int, optional): Random seed for environment reset
        max_turns (int, optional): Maximum steps per episode (default: 10)
        visgym_kwargs (dict, optional): Extra kwargs passed to gymnasium.make()
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "visgym_env_id" in extras, "visgym_env_id is required in extras"
        self.visgym_env_id = extras["visgym_env_id"]
        self.seed_value = extras.get("seed", None)
        self.max_turns = extras.get("max_turns", 10)
        visgym_kwargs = extras.get("visgym_kwargs", {})
        visgym_kwargs["relaxed"] = True

        self.visgym_env = gymnasium.make(self.visgym_env_id, **visgym_kwargs)

        self.step_count = 0
        self.parse_failures = 0

    def _build_parse_error(self) -> str:
        """Build a concise error message for parse failures."""
        valid = ", ".join(sorted(VALID_ACTIONS))
        return (
            "Action parsing failed. Could not find a valid action in your response.\n"
            f"Please end your response with <action>ACTION</action> "
            f"where ACTION is one of: {valid}"
        )

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Reset the VisGym env and return the initial multimodal prompt."""
        obs, info = self.visgym_env.reset(seed=self.seed_value)

        task_prompt = self.visgym_env.get_prompt() + _FORMAT_INSTRUCTION
        image = self.visgym_env.render()

        user_msg = make_image_message(task_prompt, image)
        initial_prompt = [user_msg]

        return initial_prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.step_count += 1

        extracted, matched = extract_relaxed_action(action)

        if not matched:
            self.parse_failures += 1
            done = self.step_count >= self.max_turns

            if not done:
                image = self.visgym_env.render()
                feedback = self._build_parse_error()
                obs_msg = make_image_message(feedback, image)
                observations = [obs_msg]
            else:
                observations = []

            return BaseTextEnvStepOutput(
                observations=observations,
                reward=0.0,
                done=done,
                metadata={
                    "env_feedback": "parse_failure",
                    "terminated": False,
                    "truncated": False,
                    "step_count": self.step_count,
                    "extracted_action": "",
                },
            )

        obs, reward, terminated, truncated, info = self.visgym_env.step(extracted)

        done = terminated or truncated or self.step_count >= self.max_turns

        if not done:
            image = self.visgym_env.render()
            feedback = info.get("env_feedback", None) or ""
            if not feedback:
                feedback = "Action executed. Here is the current state:"
            obs_msg = make_image_message(feedback, image)
            observations = [obs_msg]
        else:
            observations = []

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=float(reward) if done else 0.0,
            done=done,
            metadata={
                "env_feedback": info.get("env_feedback", ""),
                "terminated": terminated,
                "truncated": truncated,
                "step_count": self.step_count,
                "extracted_action": extracted,
            },
        )

    def close(self):
        self.visgym_env.close()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "parse_failures": self.parse_failures,
            "visgym_env_id": self.visgym_env_id,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        numeric_keys = ["step_count", "parse_failures"]
        result = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics if key in m]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
        return result
