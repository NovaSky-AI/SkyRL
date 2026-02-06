"""
Fleet Task Environment for SkyRL.

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
It uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv

# Reduce MCP client log noise
# - loguru: some MCP libs use loguru
# - standard logging: mcp.client.streamable_http uses standard logging
try:
    from loguru import logger as loguru_logger

    loguru_logger.disable("mcp")
except ImportError:
    pass
logging.getLogger("mcp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

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

    Also handles cases where the closing tag is missing (e.g., when </tool_call>
    is used as the stop string and not included in the output).

    Returns dict with "name" and "arguments" keys, or None if not found.
    """
    # Try common tag formats
    for tag in ["tool_call", "function_call"]:
        # First try with closing tag
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if not match:
            # Try without closing tag (for when </tool_call> is the stop string)
            # Match from opening tag to end of string or next special token
            match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", action, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                # json.loads can return any JSON type, we need a dict
                if not isinstance(parsed, dict):
                    continue
                # Normalize keys
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params", {})
                if name:
                    return {"name": name, "arguments": args}
            except (json.JSONDecodeError, ValueError):
                # ValueError catches Python's integer string conversion limit
                # (e.g., model generates 8000+ digit numbers)
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
        self.tool_calls = 0
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

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation (async version).

        Creates Fleet environment via OpenEnv's FleetTaskEnv and returns the task prompt.
        OpenEnv's FleetTaskEnv.__init__() creates the Fleet env and fetches tools.
        """
        # Close any existing environment
        self.close()

        # Create OpenEnv's FleetTaskEnv with normalized config
        # This creates the Fleet env (fleet.make) and fetches tools (list_tools)
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

        # Reset episode state (tools are already cached from __init__)
        obs = await self.openenv_task_env.reset_async()

        # Reset state
        self.turns = 0
        self.tool_calls = 0

        # Get tools from observation (cached from __init__)
        self.tools = obs.get("tools", [])
        if not self.tools:
            raise RuntimeError(f"Task {self.task_key}: no tools found in observation. Fleet env requires tools.")

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Build system prompt with tool definitions
        tools_json = json.dumps(self.tools, indent=2)
        # Include current date so model uses correct year for date-related tasks
        current_date = datetime.now().strftime("%Y-%m-%d")
        system_content = f"""You are a helpful agent. Complete the task by calling tools.

## Current Date
Today's date is {current_date}. When dates are mentioned without a year, assume the current year ({datetime.now().year}) or a future date.

## Available Tools
{tools_json}

## Tool Call Format
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

## Error Handling
If a tool call returns an error:
- Read the error message carefully
- Do NOT repeat the same call with identical arguments
- Change your approach: use different parameters, try a different tool, or break the task into smaller steps

## Response Format
EVERY response MUST end with exactly ONE of:
1. A tool call: <tool_call>...</tool_call> - to perform an action
2. Done signal: <done> - ONLY when the task is fully complete

NEVER respond with just a message. NEVER say "feel free to ask" or offer further help.
If the task is complete, say <done>. Otherwise, make a tool call."""

        # Build conversation with system prompt
        system_message = {"role": "system", "content": system_content}
        user_message = {"role": "user", "content": task_prompt}
        self.chat_history = [system_message, user_message]

        metadata = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation (sync wrapper).

        For async contexts, use init_async() instead.
        """
        return asyncio.run(self.init_async(prompt))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment (async version).

        Parses the action for tool calls, executes via OpenEnv's FleetTaskEnv,
        and returns observation. Reward is computed by the verifier on completion.
        """
        step_start = time.time()
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
        mcp_time = 0.0

        # Execute tool call if present via OpenEnv
        if tool_call and self.openenv_task_env:
            self.tool_calls += 1
            # Build action dict for OpenEnv
            openenv_action = {
                "tool": tool_call["name"],
                "params": tool_call.get("arguments", {}),
                "done": agent_done,
            }

            try:
                # Use async step method
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
                tool_result = obs.get("observation")
                if "tool_error" in info:
                    error = info["tool_error"]
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)
        elif agent_done and self.openenv_task_env:
            # Agent signaled done without tool call
            openenv_action = {"done": True}
            try:
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
            except Exception as e:
                mcp_time = time.time() - mcp_start
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

        step_time = time.time() - step_start
        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
            "step_time": step_time,
            "mcp_time": mcp_time,
        }

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment (sync wrapper).

        For async contexts, use step_async() instead.
        """
        return asyncio.run(self.step_async(action))

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
            "tool_calls": self.tool_calls,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes with per-env breakdown."""
        if not metrics:
            return {}

        # Group by env_key
        env_data: Dict[str, Dict[str, List[int]]] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            if env_key not in env_data:
                env_data[env_key] = {"turns": [], "tool_calls": []}
            env_data[env_key]["turns"].append(m.get("turns", 0))
            env_data[env_key]["tool_calls"].append(m.get("tool_calls", 0))

        result: Dict[str, Any] = {}
        total_turns = 0
        total_tool_calls = 0
        total_episodes = 0

        # Per-env_key metrics
        for env_key, data in env_data.items():
            turns_list = data["turns"]
            tool_calls_list = data["tool_calls"]

            avg_turns = sum(turns_list) / len(turns_list)
            avg_tool_calls = sum(tool_calls_list) / len(tool_calls_list)
            # Tool calls per turn (avoid div by zero)
            total_env_turns = sum(turns_list)
            total_env_tool_calls = sum(tool_calls_list)
            tool_calls_per_turn = total_env_tool_calls / total_env_turns if total_env_turns > 0 else 0

            result[f"{env_key}/avg_turns"] = avg_turns
            result[f"{env_key}/min_turns"] = min(turns_list)
            result[f"{env_key}/max_turns"] = max(turns_list)
            result[f"{env_key}/avg_tool_calls"] = avg_tool_calls
            result[f"{env_key}/tool_calls_per_turn"] = tool_calls_per_turn
            result[f"{env_key}/num_episodes"] = len(turns_list)

            total_turns += total_env_turns
            total_tool_calls += total_env_tool_calls
            total_episodes += len(turns_list)

        # Overall metrics
        result["avg_turns"] = total_turns / total_episodes if total_episodes > 0 else 0
        result["avg_tool_calls"] = total_tool_calls / total_episodes if total_episodes > 0 else 0
        result["tool_calls_per_turn"] = total_tool_calls / total_turns if total_turns > 0 else 0
        result["total_episodes"] = total_episodes

        return result


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE
    _TASK_CACHE = {}
