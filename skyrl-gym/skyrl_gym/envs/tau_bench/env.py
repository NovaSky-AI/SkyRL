"""SkyRL-Gym environment wrapping Sierra's tau-bench (retail domain).

This environment reuses the vendored upstream retail machinery
(``tau_core``) verbatim for tool execution and reward computation, and only
replaces the LLM user simulator with a SkyRL-controlled one (HTTP-backed in
production, scripted in tests).

Action protocol (text, since the SkyRL rollout is tag-based rather than native
function-calling): each assistant turn is either

  - a tool call:   ``<tool_call>{"name": "<tool>", "arguments": {<json args>}}</tool_call>``
  - a message to the user: any plain text without a ``<tool_call>`` block.

The episode ends when the simulated user emits ``###STOP###``, a terminating tool
(``transfer_to_human_agents``) is called, or ``max_turns`` is reached. The reward
(0/1) is the upstream tau-bench retail reward: the agent's final database state
must match the gold trajectory's state and all required outputs must have been
communicated to the user.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.tau_bench.tau_core.retail.env import MockRetailDomainEnv
from skyrl_gym.envs.tau_bench.tau_core.types import Action, RESPOND_ACTION_NAME
from skyrl_gym.envs.tau_bench.tau_core.user import BaseUserSimulationEnv
from skyrl_gym.envs.tau_bench.user_simulator import HTTPUserSimulator

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*\})\s*</tool_call>", re.DOTALL)

_RESPONSE_FORMAT = """# Response format
You are a customer service agent. On each turn take exactly ONE action: either call a tool OR reply to the user.
- To call a tool, output exactly one block and nothing else:
<tool_call>{"name": "<tool_name>", "arguments": {<json arguments>}}</tool_call>
- To reply to the user, just write your message as plain text (do NOT include a <tool_call> block).
Never do both in the same turn. Use the tools to look up and modify data; only state facts you obtained from tools."""


@dataclass
class TauBenchEnvConfig:
    """Static (per-run) config for the tau-bench environment.

    Per-task data (which task, the gold trajectory) arrives through ``extras``;
    these fields configure the user simulator and domain.
    """

    user_simulator_endpoint: str = "http://127.0.0.1:8001/v1"
    user_simulator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    user_sim_max_tokens: int = 512
    user_sim_temperature: float = 0.0
    request_timeout: int = 120
    domain: str = "retail"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class TauBenchEnv(BaseTextEnv):
    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        extras = extras or {}

        if _cfg_get(env_config, "domain", "retail") != "retail":
            raise NotImplementedError("Only the 'retail' tau-bench domain is currently supported.")

        self.task_index: int = int(extras["task_index"])
        self.task_split: str = extras.get("task_split", "test")
        self.max_turns = extras.get("max_turns", 30)

        # SkyRL-controlled user simulator injected into the vendored retail env.
        self.user_sim: BaseUserSimulationEnv = HTTPUserSimulator(
            endpoint=_cfg_get(env_config, "user_simulator_endpoint", "http://127.0.0.1:8001/v1"),
            model=_cfg_get(env_config, "user_simulator_model", "Qwen/Qwen2.5-7B-Instruct"),
            max_tokens=_cfg_get(env_config, "user_sim_max_tokens", 1024),
            temperature=_cfg_get(env_config, "user_sim_temperature", 0.0),
            timeout=_cfg_get(env_config, "request_timeout", 120),
        )
        self.tau_env = MockRetailDomainEnv(
            user=self.user_sim,
            task_split=self.task_split,
            task_index=self.task_index,
        )

        self.last_reward: float = 0.0
        self.done: bool = False
        self.num_tool_errors: int = 0

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_system_prompt(self) -> str:
        tool_schemas = json.dumps([t["function"] for t in self.tau_env.tools_info], indent=2)
        rules = "\n".join(f"- {r}" for r in self.tau_env.rules)
        return (
            f"{self.tau_env.wiki}\n\n"
            f"# Additional rules\n{rules}\n\n"
            f"# Available tools\n{tool_schemas}\n\n"
            f"{_RESPONSE_FORMAT}"
        )

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Build the system prompt and obtain the opening user message.

        The dataset ``prompt`` is a placeholder; the real conversation is built here
        so the (large, static) retail wiki + tool schemas live in the env, not in
        every dataset row.
        """
        first_user_msg = self.tau_env.reset(task_index=self.task_index).observation
        conversation: ConversationType = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": first_user_msg},
        ]
        return conversation, {"task_index": self.task_index}

    # ------------------------------------------------------------------
    # Action parsing + step
    # ------------------------------------------------------------------
    def _parse_action(self, action: str) -> Tuple[Optional[Action], Optional[str]]:
        """Return (Action, None) on success or (None, error_message) on a malformed tool call."""
        match = _TOOL_CALL_RE.search(action)
        if match is None:
            # No tool call -> message to the user.
            return Action(name=RESPOND_ACTION_NAME, kwargs={"content": action.strip()}), None
        try:
            payload = json.loads(match.group(1))
            name = payload["name"]
            kwargs = payload.get("arguments", {}) or {}
            if not isinstance(kwargs, dict):
                raise ValueError("`arguments` must be a JSON object")
            return Action(name=name, kwargs=kwargs), None
        except Exception as e:
            return (
                None,
                f'Error: could not parse tool call ({e}). Expected <tool_call>{{"name": ..., "arguments": {{...}}}}</tool_call>.',
            )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        hit_turn_limit = self.turns >= self.max_turns

        tau_action, parse_error = self._parse_action(action)
        if parse_error is not None:
            self.num_tool_errors += 1
            done = hit_turn_limit
            self.done = done
            obs = [] if done else [{"role": "user", "content": parse_error}]
            return BaseTextEnvStepOutput(observations=obs, reward=0.0, done=done, metadata={"error": parse_error})

        env_resp = self.tau_env.step(tau_action)
        reward = float(env_resp.reward)
        done = bool(env_resp.done) or hit_turn_limit
        self.last_reward = reward
        self.done = done

        metadata = {"source": env_resp.info.source}
        if done:
            # No further turns; drop the trailing observation from the trajectory.
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": env_resp.observation}],
            reward=reward,
            done=False,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "success": float(self.last_reward > 0),
            "reward": float(self.last_reward),
            "num_turns": float(self.turns),
            "num_tool_errors": float(self.num_tool_errors),
        }
