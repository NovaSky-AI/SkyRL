"""
Fleet Task Environment for SkyRL.

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
It uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

# Import SkyRL base classes
try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
except ImportError as e:
    raise ImportError("skyrl_gym is required. Make sure you're running within the SkyRL environment.") from e

# Import OpenEnv's FleetTaskEnv (from deniz/fleet_client branch)
try:
    from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv
except ImportError as e:
    raise ImportError(
        "OpenEnv is required for Fleet integration. Install from git:\n"
        "pip install git+https://github.com/fleet-ai/OpenEnv.git@deniz/fleet_client"
    ) from e


# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, Dict[str, Any]] = {}


def load_tasks_from_json(tasks_file: str) -> Dict[str, Any]:
    """Load tasks from JSON file with caching.

    Returns a dict mapping task_key -> task_config dict.
    """
    if tasks_file not in _TASK_CACHE:
        expanded_path = os.path.expanduser(tasks_file)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Tasks file not found: {expanded_path}")

        with open(expanded_path, "r") as f:
            data = json.load(f)

        # Handle both formats: array or {"tasks": [...]}
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Invalid JSON format in {tasks_file}: expected array or object with 'tasks' key")

        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")

        # Index by task_key (support both 'key' and 'task_key' fields)
        _TASK_CACHE[tasks_file] = {t.get("key") or t.get("task_key"): t for t in tasks}

    return _TASK_CACHE[tasks_file]


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool call from LLM response.

    Supports tag-based formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <function_call>{"name": "...", "arguments": {...}}</function_call>

    Returns dict with "name" and "arguments" keys, or None if not found.
    """
    # Try common tag formats
    for tag in ["tool_call", "function_call"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                # Normalize keys
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params", {})
                if name:
                    return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                pass

    return None


class FleetTaskEnv(BaseTextEnv):
    """
    SkyRL environment for Fleet-hosted tasks.

    Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
    This provides a clean separation between SkyRL's training interface and
    Fleet's environment management.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)

        # Task configuration from extras (set by dataset)
        self.task_key = extras.get("task_key")
        self.tasks_file = env_config.get("tasks_file") or extras.get("tasks_file")

        if not self.task_key:
            raise ValueError("task_key must be provided in extras (from dataset)")
        if not self.tasks_file:
            raise ValueError("tasks_file must be provided in env_config or extras")

        # Expand path
        self.tasks_file = os.path.expanduser(self.tasks_file)

        # Load task config from JSON
        tasks = load_tasks_from_json(self.tasks_file)
        self.task_config = tasks.get(self.task_key)
        if not self.task_config:
            available_keys = list(tasks.keys())[:5]
            raise ValueError(
                f"Task '{self.task_key}' not found in {self.tasks_file}. " f"Available keys (first 5): {available_keys}"
            )

        # API key
        self.api_key = env_config.get("api_key") or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError("FLEET_API_KEY must be set in env_config or environment")

        # TTL for Fleet environment instances
        self.ttl_seconds = env_config.get("ttl_seconds", 600)

        # Environment state (initialized on init())
        self.openenv_task_env: Optional[OpenEnvFleetTaskEnv] = None
        self.chat_history: ConversationType = []
        self.turns = 0
        self.tools: List[Dict[str, Any]] = []

    def _normalize_task_config(self) -> Dict[str, Any]:
        """Normalize task config to OpenEnv's expected format."""
        config = self.task_config.copy()

        # Map field names if needed
        if "key" in config and "task_key" not in config:
            config["task_key"] = config["key"]
        if "env_id" in config and "env_key" not in config:
            config["env_key"] = config["env_id"]
        if "version" in config and "env_version" not in config:
            config["env_version"] = config["version"]

        return config

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation.

        Creates Fleet environment via OpenEnv's FleetTaskEnv and returns the task prompt.
        """
        # Close any existing environment
        self.close()

        # Create OpenEnv's FleetTaskEnv with normalized config
        task_config = self._normalize_task_config()

        try:
            self.openenv_task_env = OpenEnvFleetTaskEnv(
                task_config=task_config,
                api_key=self.api_key,
                ttl_seconds=self.ttl_seconds,
                max_steps=self.max_turns,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenEnv FleetTaskEnv: {e}") from e

        # Reset the OpenEnv environment
        try:
            obs = self.openenv_task_env.reset()
        except Exception as e:
            raise RuntimeError(f"Failed to reset Fleet environment: {e}") from e

        # Reset state
        self.turns = 0

        # Get tools from observation (if available)
        self.tools = obs.get("tools", [])

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Add tool information if available
        if self.tools:
            tool_names = [t.get("name", "unknown") for t in self.tools]
            tools_info = f"\n\nAvailable tools: {', '.join(tool_names)}\n"
            tools_info += (
                'To use a tool, respond with: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>\n'
            )
            tools_info += "When you have completed the task, include <done> in your response."
        else:
            tools_info = ""

        # Build conversation
        initial_message = {"role": "user", "content": task_prompt + tools_info}
        self.chat_history = [initial_message]

        metadata = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment.

        Parses the action for tool calls, executes via OpenEnv's FleetTaskEnv,
        and returns observation. Reward is computed by the verifier on completion.
        """
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        max_turns_reached = self.turns >= self.max_turns

        # Check if agent signals completion
        agent_done = "<done>" in action.lower() or "[done]" in action.lower()

        # Parse tool call from LLM response
        tool_call = parse_tool_call(action)

        tool_result = None
        error = None
        reward = 0.0

        # Execute tool call if present via OpenEnv
        if tool_call and self.openenv_task_env:
            # Build action dict for OpenEnv
            openenv_action = {
                "tool": tool_call["name"],
                "params": tool_call.get("arguments", {}),
                "done": agent_done,
            }

            try:
                # Use async step method
                obs, reward, done, info = asyncio.get_event_loop().run_until_complete(
                    self.openenv_task_env.step_async(openenv_action)
                )
                tool_result = obs.get("observation")
                if "tool_error" in info:
                    error = info["tool_error"]
            except Exception as e:
                error = str(e)
        elif agent_done and self.openenv_task_env:
            # Agent signaled done without tool call
            openenv_action = {"done": True}
            try:
                obs, reward, done, info = asyncio.get_event_loop().run_until_complete(
                    self.openenv_task_env.step_async(openenv_action)
                )
            except Exception as e:
                error = str(e)

        # Check if episode is done
        episode_done = agent_done or max_turns_reached

        # Build observation message
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"done_reason": "max_turns", "task_key": self.task_key},
            )

        # Build response observation
        if error:
            obs_content = f"Error: {error}"
        elif tool_result:
            if isinstance(tool_result, dict):
                obs_content = f"Tool result:\n{json.dumps(tool_result, indent=2)}"
            else:
                obs_content = f"Tool result:\n{tool_result}"
        elif agent_done:
            obs_content = "Task marked as complete."
        elif not tool_call:
            obs_content = 'No tool call found. Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> format.'
        else:
            obs_content = "Action executed."

        new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)

        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
        }

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.openenv_task_env:
            try:
                self.openenv_task_env.close()
            except Exception as e:
                print(f"Warning: Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics for this episode."""
        return {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "turns": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        total_turns = sum(m.get("turns", 0) for m in metrics)

        # Group by env_key
        env_counts: Dict[str, int] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        return {
            "avg_turns": total_turns / len(metrics),
            "total_episodes": len(metrics),
            "env_distribution": env_counts,
        }


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE
    _TASK_CACHE = {}
