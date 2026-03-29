"""Fleet Task Environment for SkyRL-Gym.

Provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments,
keeping a clean separation between SkyRL's training interface and Fleet's
environment management.

Multi-modal support: When the task modality is "computer_use", step() returns
multimodal observations in OpenAI format (image_url content blocks). Upstream
SkyRL's generator already handles these via extract_images_from_conversation()
and passes them as multi_modal_data to vLLM — no upstream changes needed.
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.fleet_task.tool_call_parser import parse_tool_call

# Reduce MCP client log noise
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
            raise ValueError(
                f"Invalid JSON format in {tasks_file}: expected array or object with 'tasks' key"
            )

        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")

        # Index by task_key (support both 'key' and 'task_key' fields)
        _TASK_CACHE[tasks_file] = {
            t.get("key") or t.get("task_key"): t for t in tasks
        }

    return _TASK_CACHE[tasks_file]


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE
    _TASK_CACHE = {}


class FleetTaskEnv(BaseTextEnv):
    """SkyRL environment for Fleet-hosted tasks.

    Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
    This provides a clean separation between SkyRL's training interface and
    Fleet's environment management.

    Constructor signature follows upstream convention:
        __init__(self, env_config=None, extras={})

    Where:
        env_config: Dict or DictConfig from skyrl_gym_config YAML
        extras: Per-sample data from the training dataset (task_key, max_turns, etc.)
    """

    _trace_config: Optional[Dict[str, str]] = None

    @classmethod
    def set_trace_config(cls, job_id: str, model: str):
        """Set trace config for uploading eval traces to Fleet."""
        cls._trace_config = {"job_id": job_id, "model": model}

    @classmethod
    def clear_trace_config(cls):
        """Clear trace config after eval is done."""
        cls._trace_config = None

    def __init__(
        self,
        env_config=None,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        if env_config is None:
            env_config = {}

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)

        # Task configuration from extras (set by dataset)
        self.task_key = extras.get("task_key")
        self.tasks_file = (
            env_config.get("tasks_file") if hasattr(env_config, "get") else None
        ) or extras.get("tasks_file")

        if not self.task_key:
            raise ValueError("task_key must be provided in extras (from dataset)")
        if not self.tasks_file:
            raise ValueError(
                "tasks_file must be provided in env_config or extras"
            )

        # Expand path
        self.tasks_file = os.path.expanduser(self.tasks_file)

        # Load task config from JSON
        tasks = load_tasks_from_json(self.tasks_file)
        self.task_config = tasks.get(self.task_key)
        if not self.task_config:
            available_keys = list(tasks.keys())[:5]
            raise ValueError(
                f"Task '{self.task_key}' not found in {self.tasks_file}. "
                f"Available keys (first 5): {available_keys}"
            )

        # API key
        self.api_key = (
            env_config.get("api_key") if hasattr(env_config, "get") else None
        ) or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FLEET_API_KEY must be set in env_config or environment"
            )

        # Logfire telemetry (no-op if LOGFIRE_TOKEN is not set)
        logfire_token = os.environ.get("LOGFIRE_TOKEN")
        if logfire_token:
            try:
                from envs.fleet_env import configure_fleet_telemetry

                configure_fleet_telemetry(token=logfire_token)
            except ImportError:
                pass

        # TTL for Fleet environment instances
        self.ttl_seconds = (
            env_config.get("ttl_seconds") if hasattr(env_config, "get") else None
        )

        # Partial reward: use verifier accumulator counts instead of binary 0/1
        self.partial_reward = (
            env_config.get("partial_reward", False)
            if hasattr(env_config, "get")
            else False
        )

        # Hint config
        self.enable_hints = (
            env_config.get("enable_hints", False)
            if hasattr(env_config, "get")
            else False
        )

        # Environment state (initialized on init())
        self.openenv_task_env = None
        self.chat_history: ConversationType = []
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.last_reward: Optional[float] = None
        self.tools: List[Dict[str, Any]] = []

        # Verifier feedback (captured at close time for hint generation)
        self._verifier_stdout: Optional[str] = None
        self._verifier_error: Optional[str] = None
        self._tool_error_messages: List[str] = []

        # Context management (uses OpenEnv's ContextManager)
        self.enable_context_tools = (
            env_config.get("enable_context_tools", False)
            if hasattr(env_config, "get")
            else False
        )
        self.context_manager = None
        if self.enable_context_tools:
            try:
                from envs.fleet_env import ContextManager

                logger.info(
                    "Enabling context management tools with "
                    f"max_output_chars={extras.get('max_output_chars', 10000)}"
                )
                self.context_manager = ContextManager(
                    max_output_chars=extras.get("max_output_chars", 10000)
                )
            except ImportError:
                logger.warning(
                    "ContextManager not available, disabling context tools"
                )

    def _adapt_computer_tool_for_qwen(self):
        """Adapt computer tool description for Qwen VL's [0, 1000] coordinate space.

        Qwen3-VL/3.5 output coordinates in a normalized [0, 1000] grid regardless
        of screen resolution. This rewrites tool descriptions to match, and
        _convert_qwen_coordinates() converts back to pixels before MCP execution.
        """
        for tool in self.tools:
            func = tool.get("function", {})
            if func.get("name") != "computer":
                continue

            desc = func.get("description", "")

            # Parse actual screen dimensions
            res_match = re.search(r"Screen resolution:\s*(\d+)x(\d+)", desc)
            if res_match:
                self.screen_width = int(res_match.group(1))
                self.screen_height = int(res_match.group(2))
            else:
                self.screen_width = 1366
                self.screen_height = 768

            w, h = self.screen_width, self.screen_height

            # Rewrite description for Qwen's [0, 1000] coordinate space
            desc = re.sub(
                r"Screen resolution:\s*\d+x\d+\s*pixels\s*(\([^)]*\))?",
                "Screen resolution: 1000x1000",
                desc,
            )
            desc = re.sub(
                r"\(0, 0\) is top-left,\s*\(\d+, \d+\) is bottom-right",
                "(0, 0) is top-left, (999, 999) is bottom-right",
                desc,
            )
            desc = re.sub(
                r"valid range: x=0-\d+, y=0-\d+",
                "valid range: x=0-999, y=0-999",
                desc,
            )
            desc = re.sub(
                r"JPEG format at \d+x\d+",
                "JPEG format at 1000x1000",
                desc,
            )
            func["description"] = desc

            logger.info(
                f"Adapted computer tool for Qwen VL: actual_screen={w}x{h}, "
                f"model coordinate space=[0, 1000]"
            )
            break

    def _convert_qwen_coordinates(self, tool_call: Dict[str, Any]):
        """Convert Qwen's [0, 1000] normalized coordinates to pixel coordinates.

        Modifies tool_call arguments in-place.
        """
        if not getattr(self, "screen_width", None) or not getattr(
            self, "screen_height", None
        ):
            return
        args = tool_call.get("arguments", {})
        if not args or tool_call.get("name") != "computer":
            return
        for field in ("coordinate", "start_coordinate"):
            coords = args.get(field)
            if (
                coords
                and isinstance(coords, (list, tuple))
                and len(coords) == 2
            ):
                args[field] = [
                    int(coords[0] / 1000 * self.screen_width),
                    int(coords[1] / 1000 * self.screen_height),
                ]

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

    async def init_async(
        self, prompt: ConversationType
    ) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the Fleet environment and return initial observation.

        Creates Fleet environment via OpenEnv's FleetTaskEnv and returns
        the task prompt with tool definitions.
        """
        from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv

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
                partial_reward=self.partial_reward,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create OpenEnv FleetTaskEnv: {e}"
            ) from e

        # Reset episode state (tools are already cached from __init__)
        obs = await self.openenv_task_env.reset_async()

        # Reset state
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.last_reward = None

        # Reset context manager if enabled
        if self.context_manager:
            self.context_manager.reset()

        # Get tools from observation
        self.tools = obs.get("tools", [])

        # Add context management tools if enabled
        if self.context_manager:
            self.tools = self.tools + self.context_manager.get_tools()
        if not self.tools:
            raise RuntimeError(
                f"Task {self.task_key}: no tools found. Fleet env requires tools."
            )

        # VL: adapt computer tool for Qwen's normalized coordinate space
        modality = self.task_config.get("task_modality", "tool_use")
        if modality == "computer_use":
            self._adapt_computer_tool_for_qwen()

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Inject hint from previous failed attempt if provided
        hint = self.extras.get("hint")
        if hint:
            task_prompt = (
                f"{task_prompt}\n\nHere is feedback from a previous attempt "
                f"to help you:\n{hint}"
            )

        # Build system prompt with tool definitions
        tools_json = json.dumps(self.tools, indent=2)
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Build environment context section from env_variables
        env_context = ""
        env_vars = self.task_config.get("env_variables", {})
        if env_vars:
            env_lines = []
            if "LOGGED_IN_USER" in env_vars:
                env_lines.append(
                    f"- Logged in user ID: {env_vars['LOGGED_IN_USER']}"
                )
            if "LOGGED_IN_NAME" in env_vars:
                env_lines.append(
                    f"- Logged in as: {env_vars['LOGGED_IN_NAME']}"
                )
            for key, value in env_vars.items():
                if key not in (
                    "LOGGED_IN_USER",
                    "LOGGED_IN_NAME",
                    "CURRENT_DATE",
                ):
                    env_lines.append(f"- {key}: {value}")
            if env_lines:
                env_context = (
                    "\n## Environment Context\n"
                    + "\n".join(env_lines)
                    + "\n"
                )

        # Add environment-specific hints
        env_key = self.task_config.get("env_key") or self.task_config.get(
            "env_id"
        )
        env_hints = ""
        if env_key == "fostgres":
            env_hints = (
                "\n## Database Exploration\n"
                "Before writing SQL queries, first explore the database schema:\n"
                "- List tables: SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'\n"
                "- List columns: SELECT column_name, data_type FROM "
                "information_schema.columns WHERE table_name = 'your_table'\n"
            )

        # Computer-use hints for VL models
        computer_use_hints = ""
        if modality == "computer_use":
            computer_use_hints = (
                "\n## Browser Interaction Strategy\n"
                "You are controlling a web browser via screenshots. Follow this loop:\n\n"
                "1. **Act**: Perform ONE action (click, type, scroll, etc.)\n"
                "2. **Observe**: Take a screenshot to see the result\n"
                "3. **Adapt**: If the screen hasn't changed, try a DIFFERENT action\n\n"
                "Key rules:\n"
                "- After clicking or typing, ALWAYS take a screenshot next to see what happened\n"
                "- NEVER repeat the same action more than twice. If it didn't work, try something different:\n"
                "  - Can't find an element by scrolling? Use the search bar or navigation menu instead\n"
                "  - Page not loading after a click? Try refreshing with key(\"F5\") or clicking a different element\n"
                "  - Form not submitting? Check if required fields are missing\n"
                "- Use wait() only ONCE after a page navigation, then screenshot to check. Do not wait repeatedly\n"
                "- When the task is fully complete, say <done>. Do not keep clicking after finishing\n"
            )

        tool_names = [
            t["function"]["name"] for t in self.tools if "function" in t
        ]
        tool_names_str = ", ".join(tool_names)

        system_content = (
            f"You are a helpful agent. Complete the task by calling tools.\n\n"
            f"## Current Date\n"
            f"Today's date is {current_date}. When dates are mentioned without "
            f"a year, assume the current year ({datetime.now().year}) or a "
            f"future date.\n"
            f"{env_context}{env_hints}{computer_use_hints}\n"
            f"## Available Tools\n{tools_json}\n\n"
            f"## Tool Call Format\n"
            f"Use the tools listed above by name ({tool_names_str}). "
            f"Format each call as:\n"
            f'<tool_call>{{"name": "<tool_name_from_above>", "arguments": '
            f"{{...}}}}</tool_call>\n\n"
            f"## Error Handling\n"
            f"If a tool call returns an error:\n"
            f"- Read the error message carefully\n"
            f"- Do NOT repeat the same call with identical arguments\n"
            f"- Change your approach: use different parameters, try a different "
            f"tool, or break the task into smaller steps\n\n"
            f"## Response Format\n"
            f"EVERY response MUST end with exactly ONE of:\n"
            f"1. A tool call: <tool_call>...</tool_call> - to perform an action\n"
            f"2. Done signal: <done> - ONLY when the task is fully complete\n\n"
            f"IMPORTANT: When the task is complete, first output your final "
            f"answer with the requested information, THEN say <done>. Do not "
            f"just say <done> without providing the answer.\n\n"
            f"NEVER respond with just a message. NEVER say \"feel free to ask\" "
            f"or offer further help.\n"
            f"If the task is complete, provide your answer then say <done>. "
            f"Otherwise, make a tool call."
        )

        system_message = {"role": "system", "content": system_content}

        # VL: include initial screenshot in multimodal user message
        initial_screenshot = obs.get("initial_screenshot")
        if initial_screenshot and isinstance(initial_screenshot, list):
            user_content = [{"type": "text", "text": task_prompt}]
            for item in initial_screenshot:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    user_content.append(item)
            user_message = {"role": "user", "content": user_content}
        else:
            user_message = {"role": "user", "content": task_prompt}

        self.chat_history = [system_message, user_message]

        metadata = {
            "task_key": self.task_key,
            "env_key": env_key,
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def init(
        self, prompt: ConversationType
    ) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the Fleet environment (sync wrapper).

        Uses asyncio.run() for sync contexts. For async contexts, the upstream
        generator's _run_in_executor_if_available will call this in a thread pool,
        where asyncio.run() is safe.
        """
        return asyncio.run(self.init_async(prompt))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step in the Fleet environment.

        Parses the action for tool calls, executes via OpenEnv's FleetTaskEnv,
        and returns observation. Reward is computed by the verifier on completion.

        For computer_use modality, observations may include multimodal content
        (image_url blocks with base64 screenshots). Upstream SkyRL's generator
        handles these via extract_images_from_conversation().
        """
        step_start = time.time()
        self.turns += 1
        assistant_msg = {"role": "assistant", "content": action}
        self.chat_history.append(assistant_msg)
        if self.context_manager:
            self.context_manager.track_message(assistant_msg)

        max_turns_reached = self.turns >= self.max_turns

        # Check if agent signals completion
        agent_done = "<done>" in action.lower() or "[done]" in action.lower()

        # Parse tool call from LLM response
        tool_call = parse_tool_call(action)

        tool_result = None
        error = None
        reward = 0.0
        mcp_time = 0.0

        # VL: catch done signal wrapped in a computer tool call
        if (
            not agent_done
            and tool_call
            and tool_call.get("arguments", {}).get("action") == "done"
        ):
            agent_done = True
            tool_call = None

        # VL: convert Qwen [0,1000] coordinates to pixel coordinates
        if tool_call and getattr(self, "screen_width", None):
            self._convert_qwen_coordinates(tool_call)

        # Handle context management tools locally (no MCP call)
        if (
            tool_call
            and self.context_manager
            and self.context_manager.is_context_tool(tool_call["name"])
        ):
            tool_result, self.chat_history = self.context_manager.execute_tool(
                tool_call["name"],
                tool_call.get("arguments", {}),
                self.chat_history,
            )
        # Execute tool call if present via OpenEnv
        elif tool_call and self.openenv_task_env:
            self.tool_calls += 1
            openenv_action = {
                "tool": tool_call["name"],
                "params": tool_call.get("arguments", {}),
                "done": agent_done,
            }

            try:
                mcp_start = time.time()
                obs, reward, done, info = (
                    await self.openenv_task_env.step_async(openenv_action)
                )
                mcp_time = time.time() - mcp_start
                tool_result = obs.get("observation")
                if "tool_error" in info:
                    error = info["tool_error"]

                # Truncate long outputs if context management is enabled
                if (
                    tool_result
                    and isinstance(tool_result, str)
                    and self.context_manager
                ):
                    tool_result = self.context_manager.truncate_output(
                        tool_result
                    )
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)
        elif agent_done and self.openenv_task_env:
            # Agent signaled done without tool call
            openenv_action = {"done": True}
            try:
                mcp_start = time.time()
                obs, reward, done, info = (
                    await self.openenv_task_env.step_async(openenv_action)
                )
                mcp_time = time.time() - mcp_start
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)

        # Detect error patterns in tool_result
        if not error and tool_result:
            result_str = (
                str(tool_result)
                if not isinstance(tool_result, str)
                else tool_result
            )
            if result_str.strip().startswith(
                "Error:"
            ) or result_str.strip().startswith("error:"):
                error = result_str
                tool_result = None
            elif isinstance(tool_result, dict) and tool_result.get("error"):
                error = tool_result["error"]
                tool_result = None

        episode_done = agent_done or max_turns_reached

        # Upload trace at episode end if trace config is set
        if episode_done and FleetTaskEnv._trace_config:
            try:
                from envs.fleet_env.trace import upload_trace

                inst_id = None
                orch = getattr(self.openenv_task_env, "_orch", None)
                if orch:
                    fleet_env = getattr(orch, "_fleet_env", None)
                    if fleet_env:
                        inst_id = getattr(fleet_env, "instance_id", None)
                await upload_trace(
                    api_key=self.api_key,
                    job_id=FleetTaskEnv._trace_config["job_id"],
                    task_key=self.task_key,
                    model=FleetTaskEnv._trace_config["model"],
                    chat_history=self.chat_history,
                    reward=reward,
                    instance_id=inst_id,
                    metadata={
                        "env_key": self.task_config.get("env_key"),
                        "turns": self.turns,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to upload trace for {self.task_key}: {e}"
                )

        # Build observation message
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={
                    "done_reason": "max_turns",
                    "task_key": self.task_key,
                },
            )

        # Build response observation
        if error:
            self.tool_errors += 1
            self._tool_error_messages.append(str(error)[:500])
            obs_content = f"Error: {error}"
        elif tool_result:
            # Handle multimodal results (list with image_url blocks)
            if isinstance(tool_result, list):
                # Multimodal: return as structured content for VL models
                new_obs = {"role": "user", "content": tool_result}
                self.chat_history.append(new_obs)
                if self.context_manager:
                    self.context_manager.track_message(new_obs)

                step_time = time.time() - step_start
                metadata = {
                    "task_key": self.task_key,
                    "turn": self.turns,
                    "tool_call": tool_call,
                    "error": None,
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
            elif isinstance(tool_result, dict):
                obs_content = (
                    f"Tool result:\n{json.dumps(tool_result, indent=2)}"
                )
            else:
                obs_content = f"Tool result:\n{tool_result}"
        elif agent_done:
            obs_content = "Task marked as complete."
        elif not tool_call:
            obs_content = (
                "No tool call found. Use "
                '<tool_call>{"name": "...", "arguments": {...}}</tool_call> '
                "format."
            )
        else:
            obs_content = "Action executed."

        new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)
        if self.context_manager:
            self.context_manager.track_message(new_obs)

        step_time = time.time() - step_start
        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": (
                tool_result[:200]
                if isinstance(tool_result, str) and len(tool_result) > 200
                else tool_result
            ),
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
            "step_time": step_time,
            "mcp_time": mcp_time,
        }

        # If context was modified, return full chat_history so the generator
        # can replace its copy (required for stepwise training).
        if (
            tool_call
            and self.context_manager
            and self.context_manager.is_context_tool(tool_call["name"])
        ):
            if tool_call["name"] == "manage_context":
                metadata["modified_chat_history"] = self.chat_history.copy()

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step in the Fleet environment (sync wrapper)."""
        return asyncio.run(self.step_async(action))

    def _capture_verifier_feedback(self):
        """Capture verifier feedback from OpenEnv before nulling the env."""
        if self.openenv_task_env:
            self._verifier_stdout = getattr(
                self.openenv_task_env, "verifier_stdout", None
            )
            self._verifier_error = getattr(
                self.openenv_task_env, "verifier_error", None
            )
            self._tool_error_messages = getattr(
                self.openenv_task_env, "tool_errors_list", []
            )

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.openenv_task_env:
            try:
                self.openenv_task_env.close()
                if self.openenv_task_env.final_reward is not None:
                    self.last_reward = self.openenv_task_env.final_reward
                self._capture_verifier_feedback()
            except Exception as e:
                logger.warning(f"Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    async def close_async(self):
        """Close the Fleet environment (async version).

        Runs verifier via OpenEnv's close_async() to get actual reward for
        orphaned rollouts (context overflow, early termination by SkyRL).
        """
        if self.openenv_task_env:
            try:
                await self.openenv_task_env.close_async()
                if self.openenv_task_env.final_reward is not None:
                    self.last_reward = self.openenv_task_env.final_reward
                self._capture_verifier_feedback()
            except Exception as e:
                logger.warning(f"Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics for this episode."""
        metrics = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key")
            or self.task_config.get("env_id"),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "is_hinted": bool(self.extras.get("hint")),
        }
        if self.last_reward is not None:
            metrics["final_reward"] = self.last_reward
        # Include verifier feedback for hint generation
        if self._verifier_stdout is not None:
            metrics["verifier_stdout"] = self._verifier_stdout
        if self._verifier_error is not None:
            metrics["verifier_error"] = self._verifier_error
        if self._tool_error_messages:
            metrics["tool_error_messages"] = self._tool_error_messages
        return metrics

    @staticmethod
    def build_hint_text(
        verifier_stdout: Optional[str],
        verifier_error: Optional[str],
        tool_error_messages: Optional[List[str]],
    ) -> str:
        """Build hint text from verifier feedback. No LLM call.

        Parses ERROR_ACCUMULATOR / SUCCESS_ACCUMULATOR from verifier stdout
        and formats tool errors into structured feedback for the next attempt.
        """
        parts = []

        if verifier_stdout:
            err_match = re.search(
                r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            suc_match = re.search(
                r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n"
                r"<<< SUCCESS_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            if err_match or suc_match:
                try:
                    errors = (
                        ast.literal_eval(err_match.group(1))
                        if err_match
                        else []
                    )
                    successes = (
                        ast.literal_eval(suc_match.group(1))
                        if suc_match
                        else []
                    )
                except Exception:
                    errors, successes = [], []
                if successes:
                    parts.append(
                        f"Checks passed ({len(successes)}): "
                        + ", ".join(
                            str(s)[:100] for s in successes[:5]
                        )
                    )
                if errors:
                    parts.append(
                        f"Checks failed ({len(errors)}): "
                        + ", ".join(str(e)[:100] for e in errors[:5])
                    )

        if verifier_error:
            parts.append(f"Verifier: {verifier_error}")

        if tool_error_messages:
            unique = list(dict.fromkeys(tool_error_messages))[:5]
            parts.append(
                "Tool errors: " + "; ".join(e[:200] for e in unique)
            )

        return (
            "\n".join(parts)
            if parts
            else "The previous attempt failed. Try a different approach."
        )

    @staticmethod
    def aggregate_metrics(
        metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate metrics across episodes with per-env breakdown."""
        if not metrics:
            return {}

        env_init_failures: Dict[str, int] = {}
        total_init_failures = 0

        env_data: Dict[str, Dict[str, List[int]]] = {}
        for m in metrics:
            # Check for init failure metrics
            for key, value in m.items():
                if key.startswith("env_init_failed/"):
                    env_key = key.split("/", 1)[1]
                    env_init_failures[env_key] = (
                        env_init_failures.get(env_key, 0) + int(value)
                    )
                    total_init_failures += int(value)

            env_key = m.get("env_key")
            if env_key:
                if env_key not in env_data:
                    env_data[env_key] = {
                        "turns": [],
                        "tool_calls": [],
                        "tool_errors": [],
                    }
                env_data[env_key]["turns"].append(m.get("turns", 0))
                env_data[env_key]["tool_calls"].append(
                    m.get("tool_calls", 0)
                )
                env_data[env_key]["tool_errors"].append(
                    m.get("tool_errors", 0)
                )

        result: Dict[str, Any] = {}
        total_turns = 0
        total_tool_calls = 0
        total_tool_errors = 0
        total_episodes = 0

        for env_key, data in env_data.items():
            turns_list = data["turns"]
            tool_calls_list = data["tool_calls"]
            tool_errors_list = data["tool_errors"]

            avg_turns = sum(turns_list) / len(turns_list)
            avg_tool_calls = sum(tool_calls_list) / len(tool_calls_list)
            avg_tool_errors = sum(tool_errors_list) / len(tool_errors_list)
            total_env_turns = sum(turns_list)
            total_env_tool_calls = sum(tool_calls_list)
            total_env_tool_errors = sum(tool_errors_list)
            tool_calls_per_turn = (
                total_env_tool_calls / total_env_turns
                if total_env_turns > 0
                else 0
            )
            tool_error_rate = (
                total_env_tool_errors / total_env_tool_calls
                if total_env_tool_calls > 0
                else 0
            )

            result[f"{env_key}/avg_turns"] = avg_turns
            result[f"{env_key}/min_turns"] = min(turns_list)
            result[f"{env_key}/max_turns"] = max(turns_list)
            result[f"{env_key}/avg_tool_calls"] = avg_tool_calls
            result[f"{env_key}/tool_calls_per_turn"] = tool_calls_per_turn
            result[f"{env_key}/avg_tool_errors"] = avg_tool_errors
            result[f"{env_key}/total_tool_errors"] = total_env_tool_errors
            result[f"{env_key}/tool_error_rate"] = tool_error_rate
            result[f"{env_key}/num_episodes"] = len(turns_list)

            total_turns += total_env_turns
            total_tool_calls += total_env_tool_calls
            total_tool_errors += total_env_tool_errors
            total_episodes += len(turns_list)

        result["avg_turns"] = (
            total_turns / total_episodes if total_episodes > 0 else 0
        )
        result["avg_tool_calls"] = (
            total_tool_calls / total_episodes if total_episodes > 0 else 0
        )
        result["tool_calls_per_turn"] = (
            total_tool_calls / total_turns if total_turns > 0 else 0
        )
        result["avg_tool_errors"] = (
            total_tool_errors / total_episodes if total_episodes > 0 else 0
        )
        result["total_tool_errors"] = total_tool_errors
        result["tool_error_rate"] = (
            total_tool_errors / total_tool_calls
            if total_tool_calls > 0
            else 0
        )
        result["total_episodes"] = total_episodes

        for env_key, failures in env_init_failures.items():
            result[f"{env_key}/env_init_failed"] = failures
        if total_init_failures > 0:
            result["total_env_init_failed"] = total_init_failures

        return result
