import re
from typing import Any, Dict, List, Tuple

import gymnasium

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from examples.train.visgym.utils import extract_action, make_image_message

_TASK_REWARD_COEFF = 0.8
_FORMAT_REWARD_COEFF = 0.2

_XML_TAG_RE = {
    tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for tag in ("observation", "justification", "action")
}


def _has_valid_format(text: str) -> bool:
    """True if all three XML tags are present with non-empty content."""
    for pattern in _XML_TAG_RE.values():
        match = pattern.search(text)
        if not match or not match.group(1).strip():
            return False
    return True


_FORMAT_INSTRUCTION = (
    "\n\nIMPORTANT: Respond with exactly one action as a Python tuple, "
    "e.g. ('move', 0) or ('stop', 'stop'). "
    "Do not wrap the action in backticks, code blocks, or other formatting."
)


class VisGymEnv(BaseTextEnv):
    """Wraps a VisGym environment as a BaseTextEnv for use with SkyRLGymGenerator.

    Bridges VisGym's gymnasium.Env interface (image observations, tuple-string actions,
    binary rewards) to SkyRL-Gym's BaseTextEnv interface (OpenAI message format, raw text
    actions, float rewards).

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

        self.visgym_env = gymnasium.make(self.visgym_env_id, **visgym_kwargs)

        self.step_count = 0
        self.parse_failures = 0
        self.format_successes = 0

    def _build_parse_error(self) -> str:
        """Build a concise, informative error message for parse failures."""
        available_actions = list(self.visgym_env.action_space.get_function_names())
        names = ", ".join(f"'{a}'" for a in available_actions)
        return (
            "Action parsing failed. Could not find a valid action tuple in your response."
            "\nPlease respond with exactly one action as a Python tuple."
            f"\nAvailable actions: {names}"
            f"\nExample: ('{available_actions[0]}', 0) or ('stop', 'stop')"
        )

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Reset the VisGym env and return the initial multimodal prompt."""
        obs, info = self.visgym_env.reset(seed=self.seed_value)

        task_prompt = self.visgym_env.get_prompt() + _FORMAT_INSTRUCTION
        image = self.visgym_env.render()

        user_msg = make_image_message(task_prompt, image)
        initial_prompt = [user_msg]

        return initial_prompt, {}

    def _compute_terminal_reward(self, task_reward: float) -> float:
        format_reward = self.format_successes / self.step_count
        return _TASK_REWARD_COEFF * task_reward + _FORMAT_REWARD_COEFF * format_reward

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.step_count += 1

        if _has_valid_format(action):
            self.format_successes += 1

        extracted, matched = extract_action(action)

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
                reward=self._compute_terminal_reward(0.0) if done else 0.0,
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
            reward=self._compute_terminal_reward(float(reward)) if done else 0.0,
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
            "format_successes": self.format_successes,
            "format_success_rate": (self.format_successes / self.step_count if self.step_count > 0 else 0.0),
            "visgym_env_id": self.visgym_env_id,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        numeric_keys = ["step_count", "parse_failures", "format_successes", "format_success_rate"]
        result = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics if key in m]
            if values:
                result[f"avg_{key}"] = sum(values) / len(values)
        return result
