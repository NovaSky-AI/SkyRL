"""OpenReward environment adapter for SkyRL.

Wraps a remote OpenReward environment as a BaseTextEnv so it can be used
with the standard SkyRLGymGenerator agent_loop.

Expected env_extras (from the dataset prepared by prepare_tasks.py):
    - env_name: str       — OpenReward environment name, e.g. "GeneralReasoning/WhoDunit"
    - split: str          — task split, e.g. "train"
    - task_index: int     — task index within the split

Rollout upload is controlled by the OPENREWARD_UPLOAD_ROLLOUT environment variable.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from openreward import OpenReward
from openreward.api.rollouts.serializers.base import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # seconds, exponential backoff: 3, 6, 12, 24, 48


def _retry_on_server_error(fn, *args, **kwargs):
    """Retry a callable with exponential backoff on 5xx / connection errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            is_retryable = any(
                code in err_str for code in ("503", "502", "429", "Connection refused", "connection timeout")
            )
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(f"OpenReward API error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
            time.sleep(delay)


class OpenRewardEnv(BaseTextEnv):
    """BaseTextEnv adapter for OpenReward remote environments."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        self.env_name: str = extras["env_name"]
        self.split: str = extras["split"]
        self.task_index: int = int(extras["task_index"])
        self.max_turns = extras.get("max_turns", 10)
        self.upload_rollout: bool = os.environ.get("OPENREWARD_UPLOAD_ROLLOUT", "false").lower() == "true"

        # Accumulated rewards across turns
        self._rewards: List[float] = []

        # Session state (opened in init(), closed in close())
        self._client: Optional[OpenReward] = None
        self._env = None
        self._session = None
        self._session_ctx = None

        # Rollout tracking for OpenReward upload
        self._or_blocks: List[Dict[str, Any]] = []
        self._rollout_name: Optional[str] = None
        self._task_spec: Optional[Any] = None

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Open an OpenReward session for this task. Prompt is already pre-built by prepare_tasks.py."""
        # prompt may arrive as a JSON string from the Parquet dataset — parse it
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        self._client = OpenReward()
        self._env = self._client.environments.get(name=self.env_name)
        self._session_ctx = self._env.session(split=self.split, index=self.task_index)
        self._session = _retry_on_server_error(self._session_ctx.__enter__)

        # Initialize rollout tracking if enabled
        if self.upload_rollout:
            self._or_blocks = []
            self._rollout_name = f"{self.env_name.replace('/', '-')}-{self.split}-{self.task_index}"
            try:
                self._task_spec = self._session.task_spec
            except AttributeError:
                self._task_spec = None
            # Add system and user messages from prompt to rollout blocks
            for msg in prompt:
                if msg.get("role") == "system":
                    self._or_blocks.append({"message": SystemMessage(content=msg.get("content", ""))})
                elif msg.get("role") == "user":
                    self._or_blocks.append({"message": UserMessage(content=msg.get("content", ""))})

        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        # Parse tool call from model output
        tool_call = _parse_tool_call(action)

        # No tool call found — treat as final answer, end episode
        if tool_call is None:
            # No new tool call this turn, so reward = 0 (previous rewards already returned per-step)
            reward = 0.0
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"stop_reason": "no_tool_call"},
            )

        # Tool call parse error — give error feedback, continue
        if tool_call["type"] == "error":
            error_msg = f"Error parsing tool call: {tool_call['error']}"
            obs = [{"role": "user", "content": f"<tool_response>\n{error_msg}\n</tool_response>"}]
            return BaseTextEnvStepOutput(
                observations=obs,
                reward=0.0,
                done=self.turns >= self.max_turns,
                metadata={"tool_call_error": tool_call["error"]},
            )

        # Execute the tool call against OpenReward
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        # Track assistant message and tool call for rollout
        if self.upload_rollout:
            call_id = hashlib.md5(f"{self.turns}:{tool_name}".encode()).hexdigest()[:12]
            self._or_blocks.append(
                {
                    "message": AssistantMessage(content=action.strip()),
                }
            )
            self._or_blocks.append(
                {
                    "message": ToolCall(name=tool_name, content=json.dumps(tool_args), call_id=call_id),
                }
            )

        tool_output = None
        try:
            tool_output = _retry_on_server_error(self._session.call_tool, tool_name=tool_name, input=tool_args)
            output_text = "".join(b.text for b in tool_output.blocks if b.type == "text")
            finished = tool_output.finished
            reward = tool_output.reward if tool_output.reward is not None else 0.0
        except Exception as e:
            # Catch ToolCallError, network errors, 429s etc. gracefully
            logger.warning(f"OpenReward call_tool failed: {e}")
            output_text = f"Error: {e}"
            finished = False
            reward = 0.0

        # Track tool result for rollout
        if self.upload_rollout:
            call_id = hashlib.md5(f"{self.turns}:{tool_name}".encode()).hexdigest()[:12]
            self._or_blocks.append(
                {
                    "message": ToolResult(content=output_text, call_id=call_id),
                    "reward": tool_output.reward if tool_output is not None else None,
                    "is_finished": finished,
                    "metadata": getattr(tool_output, "metadata", None) if tool_output is not None else None,
                }
            )

        self._rewards.append(reward)
        done = finished or self.turns >= self.max_turns

        obs = [{"role": "user", "content": f"<tool_response>\n{output_text}\n</tool_response>"}]

        return BaseTextEnvStepOutput(
            observations=obs,
            reward=reward,
            done=done,
            metadata={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "finished": finished,
            },
        )

    def close(self):
        """Close the OpenReward session and client to avoid async cleanup issues."""
        # Upload rollout before closing if enabled
        if self.upload_rollout and self._or_blocks:
            self._upload_rollout()

        if self._session_ctx is not None:
            try:
                self._session_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
            self._session = None

        # Explicitly close the client to prevent async cleanup in GC
        if self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
            except Exception:
                pass
            self._client = None

    def _upload_rollout(self) -> None:
        """Upload rollout to OpenReward for logging/visualization (fire-and-forget)."""
        if not self._or_blocks or not self._client:
            return

        run_name = os.environ.get("OPENREWARD_RUN_NAME", "skyrl-openreward")

        def _sync_upload():
            try:
                rollout = self._client.rollout.create(
                    run_name=run_name,
                    rollout_name=self._rollout_name or "unknown",
                    environment=self.env_name,
                    split=self.split,
                    task_spec=self._task_spec,
                    metadata={
                        "stop_reason": "finished" if self._rewards else "no_tool_call",
                        "total_reward": sum(self._rewards),
                        "num_turns": self.turns,
                    },
                )
                for block in self._or_blocks:
                    rollout.log(
                        message=block["message"],
                        reward=block.get("reward"),
                        is_finished=block.get("is_finished", False),
                        metadata=block.get("metadata"),
                    )
            except Exception as e:
                logger.warning(f"Failed to upload rollout: {e}")

        # Upload synchronously — close() immediately cleans up the client after this,
        # so fire-and-forget would race with cleanup.
        try:
            _sync_upload()
        except Exception as e:
            logger.warning(f"Rollout upload failed: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "total_reward": sum(self._rewards),
            "num_rewards": len(self._rewards),
        }


def _parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse a <tool_call>...</tool_call> block from generated text.

    Returns:
        None if no tool call found.
        {"type": "success", "name": str, "arguments": dict} on success.
        {"type": "error", "error": str} on parse failure.
    """
    start_tag, end_tag = "<tool_call>", "</tool_call>"
    si = text.find(start_tag)
    if si == -1:
        return None

    ei = text.find(end_tag, si)
    json_str = text[si + len(start_tag) : ei].strip() if ei != -1 else text[si + len(start_tag) :].strip()

    try:
        data = json.loads(json_str)
        name = data.get("name")
        args = data.get("arguments", {})
        if not name:
            return {"type": "error", "error": "missing 'name' field"}
        if not isinstance(args, dict):
            return {"type": "error", "error": f"arguments is not a dict: {type(args).__name__}"}
        return {"type": "success", "name": name, "arguments": args}
    except (json.JSONDecodeError, KeyError) as e:
        return {"type": "error", "error": str(e)}
