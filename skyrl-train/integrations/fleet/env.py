"""
Fleet Task Environment for SkyRL.

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
It wraps OpenEnv's FleetTaskEnv and adapts it to SkyRL's BaseTextEnv interface.
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

# Import OpenEnv's FleetTaskEnv
from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv

# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, Dict[str, Any]] = {}


def _run_async(coro):
    """Run async coroutine, handling both sync and async contexts."""
    try:
        asyncio.get_running_loop()
        # Already in async context - this shouldn't happen in SkyRL's sync step()
        raise RuntimeError("Cannot call sync step() from async context")
    except RuntimeError:
        # No running loop, safe to use asyncio.run()
        return asyncio.run(coro)


def load_tasks_from_json(tasks_file: str) -> Dict[str, Dict[str, Any]]:
    """Load tasks from JSON file with caching."""
    if tasks_file not in _TASK_CACHE:
        with open(tasks_file, "r") as f:
            data = json.load(f)

        # Handle both formats: array or {"tasks": [...]}
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError("Invalid JSON format: expected array or object with 'tasks' key")

        # Index by task_key
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

    Wraps OpenEnv's FleetTaskEnv and adapts it to SkyRL's BaseTextEnv interface.
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
            raise ValueError(f"Task '{self.task_key}' not found in {self.tasks_file}")

        # API key
        self.api_key = env_config.get("api_key") or os.environ.get("FLEET_API_KEY")

        # Environment state (initialized on init())
        self.fleet_env: Optional[OpenEnvFleetTaskEnv] = None
        self.chat_history: ConversationType = []

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation.

        Creates OpenEnv's FleetTaskEnv, resets it, and returns the task prompt.
        """
        # Create OpenEnv's FleetTaskEnv
        self.fleet_env = OpenEnvFleetTaskEnv(
            task_config=self.task_config,
            api_key=self.api_key,
            ttl_seconds=600,
            max_steps=self.max_turns,
        )

        # Reset the environment (this provisions the Fleet instance)
        obs = self.fleet_env.reset()

        # Reset state
        self.turns = 0

        # Get tools from observation
        tools = obs.get("tools", [])

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Add tool information if available
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
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
            "tools": tools,
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

        # Build action dict for FleetTaskEnv
        fleet_action = {"done": agent_done}

        # Parse and add tool call
        tool_call = parse_tool_call(action)
        if tool_call:
            fleet_action["tool"] = tool_call.get("name")
            fleet_action["params"] = tool_call.get("arguments", {})

        # Execute step via OpenEnv's FleetTaskEnv
        obs, reward, done, info = _run_async(self.fleet_env.step_async(fleet_action))

        # Extract tool result from observation
        tool_result = obs.get("observation")
        error = info.get("tool_error")

        # Determine if episode is done
        episode_done = done or max_turns_reached

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
            "done_reason": info.get("done_reason"),
        }

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.fleet_env:
            try:
                self.fleet_env.close()
            except Exception as e:
                print(f"Warning: Failed to close Fleet environment: {e}")
            self.fleet_env = None

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
        env_counts = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        return {
            "avg_turns": total_turns / len(metrics),
            "total_episodes": len(metrics),
            "env_distribution": env_counts,
        }
