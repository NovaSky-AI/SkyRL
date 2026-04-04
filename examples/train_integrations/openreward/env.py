"""OpenReward environment adapter for SkyRL.

Wraps a remote OpenReward environment as a BaseTextEnv so it can be used
with the standard SkyRLGymGenerator agent_loop.

Expected env_extras (from the dataset prepared by prepare_tasks.py):
    - env_name: str       — OpenReward environment name, e.g. "GeneralReasoning/WhoDunit"
    - split: str          — task split, e.g. "train"
    - task_index: int     — task index within the split
"""
import json
import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
            is_retryable = any(code in err_str for code in ("503", "502", "429", "Connection refused", "connection timeout"))
            if not is_retryable or attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(f"OpenReward API error (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
            time.sleep(delay)

logger = logging.getLogger(__name__)

# Lazy-loaded at first use
_OpenReward = None


def _get_openreward_client_class():
    """Lazy import openreward, adding system site-packages to sys.path if needed."""
    global _OpenReward
    if _OpenReward is not None:
        return _OpenReward

    try:
        from openreward import OpenReward
        _OpenReward = OpenReward
        return _OpenReward
    except ImportError:
        pass

    # openreward was pip-installed into system Python by run_openreward.sh
    # but Ray workers may not see it. Add system site-packages to sys.path.
    import glob
    for pattern in [
        "/home/ray/anaconda3/lib/python3.*/site-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/local/lib/python3.*/site-packages",
    ]:
        for sp in glob.glob(pattern):
            if sp not in sys.path:
                sys.path.insert(0, sp)

    from openreward import OpenReward
    _OpenReward = OpenReward
    return _OpenReward


class OpenRewardEnv(BaseTextEnv):
    """BaseTextEnv adapter for OpenReward remote environments."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()

        self.env_name: str = extras["env_name"]
        self.split: str = extras["split"]
        self.task_index: int = int(extras["task_index"])
        self.max_turns = extras.get("max_turns", 10)

        # Accumulated rewards across turns
        self._rewards: List[float] = []

        # Session state (opened in init(), closed in close())
        self._client: Optional[OpenReward] = None
        self._env = None
        self._session = None
        self._session_ctx = None

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Open an OpenReward session for this task. Prompt is already pre-built by prepare_tasks.py."""
        # prompt may arrive as a JSON string from the Parquet dataset — parse it
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        OpenReward = _get_openreward_client_class()
        self._client = OpenReward()
        self._env = self._client.environments.get(name=self.env_name)
        self._session_ctx = self._env.session(split=self.split, index=self.task_index)
        self._session = _retry_on_server_error(self._session_ctx.__enter__)
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        # Parse tool call from model output
        tool_call = _parse_tool_call(action)

        # No tool call found — treat as final answer, end episode
        if tool_call is None:
            reward = sum(self._rewards)
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

        try:
            tool_output = _retry_on_server_error(self._session.call_tool, tool_name=tool_name, input=tool_args)
            output_text = "".join(b.text for b in tool_output.blocks if b.type == "text")
            finished = tool_output.finished
            reward = tool_output.reward or 0.0
        except Exception as e:
            # Catch ToolCallError, network errors, 429s etc. gracefully
            logger.warning(f"OpenReward call_tool failed: {e}")
            output_text = f"Error: {e}"
            finished = False
            reward = 0.0

        self._rewards.append(reward)
        done = finished or self.turns >= self.max_turns

        obs = [{"role": "user", "content": f"<tool_response>\n{output_text}\n</tool_response>"}]

        # On done, return cumulative reward
        final_reward = sum(self._rewards) if done else reward

        return BaseTextEnvStepOutput(
            observations=obs,
            reward=final_reward,
            done=done,
            metadata={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "finished": finished,
            },
        )

    def close(self):
        """Close the OpenReward session."""
        if self._session_ctx is not None:
            try:
                self._session_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._session_ctx = None
            self._session = None

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
    json_str = text[si + len(start_tag):ei].strip() if ei != -1 else text[si + len(start_tag):].strip()

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
