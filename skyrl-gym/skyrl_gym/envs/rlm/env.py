import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from .repl import PersistentREPL, REPLResult, _iter_tool_entries


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_RLM_SYSTEM_PROMPT = """\
You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `SHOW_VARS()` function that returns all variables you have created in the REPL. Use this to check what variables exist before using FINAL_VAR.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.
{custom_tools_section}
When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier:
```repl
# your code here
```

Use variables as buffers to build up your final answer. Make sure to explicitly look through the context in the REPL before answering your query.

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer using one of:
1. FINAL(your final answer here) — provide the answer directly as text
2. FINAL_VAR(variable_name) — return a variable you have created in the REPL

WARNING: FINAL_VAR retrieves an EXISTING variable. You MUST create and assign the variable in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE response.

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this". Output to the REPL environment as much as possible.\
"""


# ---------------------------------------------------------------------------
# Per-turn user prompt injection (from rlm/rlm/utils/prompts.py)
# ---------------------------------------------------------------------------

_USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the original prompt: \"{root_prompt}\".\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)


def _build_user_prompt(root_prompt: Optional[str], iteration: int) -> Dict[str, str]:
    """Build the per-turn user message injected before every model call."""
    if iteration == 0:
        safeguard = (
            "You have not interacted with the REPL environment or seen your prompt / context yet. "
            "Your next action should be to look through and figure out how to answer the prompt, "
            "so don't just provide a final answer yet.\n\n"
        )
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = safeguard + body
    else:
        prefix = "The history before is your previous interactions with the REPL environment. "
        body = _USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else _USER_PROMPT
        content = prefix + body
    return {"role": "user", "content": content}


# ---------------------------------------------------------------------------
# Parsing helpers (from rlm/rlm/utils/parsing.py)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _find_code_block(text: str) -> Optional[str]:
    """Return the first ```repl ... ``` code block, or None."""
    text = _strip_thinking(text)
    match = re.search(r"```repl\s*\n(.*?)\n```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _find_final_answer(text: str, repl: Optional[PersistentREPL]) -> Optional[str]:
    """Parse FINAL_VAR(...) or FINAL(...) from the model's text response."""
    text = _strip_thinking(text)

    # FINAL_VAR — retrieves a variable from the REPL
    match = re.search(r"^\s*FINAL_VAR\((.*?)\)", text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if repl is not None:
            result = repl.execute(f"print(FINAL_VAR({variable_name!r}))")
            answer = result.stdout.strip()
            if answer == "":
                return None
            if "Variable '" in answer and "' not found" in answer and "FINAL_VAR" in answer:
                return None
            return answer
        return None

    # FINAL — inline literal
    match = re.search(r"^\s*FINAL\((.*)\)\s*$", text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _format_execution_result(result: REPLResult) -> str:
    """Format a REPLResult as a string for display in the conversation (from rlm/rlm/utils/parsing.py)."""
    parts = []
    if result.stdout:
        parts.append(f"\n{result.stdout}")
    if result.stderr:
        parts.append(f"\n{result.stderr}")
    important_vars = {
        k: ""
        for k, v in result.locals.items()
        if not k.startswith("_")
        and k not in ("__builtins__", "__name__", "__doc__")
        and isinstance(v, (str, int, float, bool, list, dict, tuple))
    }
    if important_vars:
        parts.append(f"REPL variables: {list(important_vars.keys())}\n")
    return "\n\n".join(parts) if parts else "No output"


_MAX_RESULT_LEN = 20_000


def _format_context_metadata(context_payload) -> str:
    """Build the model-facing 'your context is a ... with ... total characters' line."""
    if isinstance(context_payload, str):
        ctx_type, lengths = "str", [len(context_payload)]
    elif isinstance(context_payload, dict):
        ctx_type = "dict"
        lengths = []
        for chunk in context_payload.values():
            if isinstance(chunk, str):
                lengths.append(len(chunk))
            else:
                try:
                    lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    lengths.append(len(repr(chunk)))
    elif isinstance(context_payload, list):
        ctx_type, lengths = "list", [len(str(c)) for c in context_payload]
    else:
        ctx_type, lengths = type(context_payload).__name__, [len(repr(context_payload))]
    return (
        f"Your context is a {ctx_type} with {sum(lengths)} total characters, "
        f"and is broken up into chunks of char lengths: {lengths}."
    )


def _format_tools_for_prompt(custom_tools: Optional[Dict[str, Any]]) -> Optional[str]:
    """Format custom tools for inclusion in the system prompt."""
    lines = []
    for name, value, description in _iter_tool_entries(custom_tools):
        if callable(value):
            lines.append(f"- `{name}`: {description}" if description else f"- `{name}`: A custom function")
        else:
            lines.append(f"- `{name}`: {description}" if description else f"- `{name}`: A custom {type(value).__name__} value")
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLMEnvConfig:
    repl_timeout: float = 60.0
    parent_repl_timeout: float = 180.0  # timeout for parent REPL (with child RLM calls)
    custom_system_prompt: Optional[str] = None
    child_system_prompt: Optional[str] = None
    custom_tools: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------

class BaseRLMEnv(BaseTextEnv):
    """Base class for Recursive Language Model (RLM) environments.

    Provides REPL plumbing, parent/child rollout wiring, the multi-turn loop,
    and FINAL/FINAL_VAR parsing. Task-specific behavior — reward, system
    prompt, REPL tools — is supplied by subclasses via three override hooks:

      • ``_get_reward(final_answer)``   — score the final answer
      • ``_get_system_prompt()``        — return the system prompt template
      • ``_get_repl_tools()``           — return task-specific REPL helpers
                                          (called after ``self._context`` is set,
                                          so closures can capture it)

    See ``examples/train/rlm/envs/evidence_rlm_env.py`` for a worked example.

    init() returns:
        [system_msg, context_metadata_msg, turn_0_user_prompt_msg]

    step() returns observations:
        [repl_output_msg, turn_N_user_prompt_msg]

    The model always sees the per-turn user prompt as the last message before
    it generates, keeping root_prompt visible every turn.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        self.max_turns = extras.get("max_turns", 10)

        if isinstance(env_config, RLMEnvConfig):
            self.rlm_config = env_config
        elif isinstance(env_config, Mapping):
            self.rlm_config = RLMEnvConfig(**{k: v for k, v in env_config.items() if k in RLMEnvConfig.__dataclass_fields__})
        else:
            self.rlm_config = RLMEnvConfig()

        # Per-example custom tools merged on top of any static config-level tools
        if extras.get("custom_tools"):
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(extras["custom_tools"])
            self.rlm_config.custom_tools = merged

        # LM query callbacks — passed via extras to stay serialization-friendly
        self.lm_callback = extras.get("lm_callback", None)
        self.subcall_fn = extras.get("subcall_fn", None)

        self.step_wise = extras.get("step_wise", True)

        # Subclasses may set self.reward_fn in their __init__ if they want the
        # default _get_reward to delegate. Otherwise they should override _get_reward.
        self.reward_fn: Optional[Any] = None

        self.repl: Optional[PersistentREPL] = None
        self._context: Any = None
        self._final_answer: Optional[str] = None
        self._turn_index = 0
        self._last_repl_exec_s: float = 0.0

    # ------------------------------------------------------------------
    # Override hooks — subclasses customize task-specific behavior here
    # ------------------------------------------------------------------

    def _get_reward(self, final_answer: Optional[str]) -> float:
        """Score the final answer. Override or set ``self.reward_fn``.

        Default: delegates to ``self.reward_fn(final_answer)`` if set, else 0.0.
        """
        if final_answer is None:
            return 0.0
        if self.reward_fn is not None:
            return float(self.reward_fn(final_answer))
        return 0.0

    def _get_system_prompt(self) -> str:
        """Return the system prompt template. Override for custom prompts.

        The returned string may contain ``{custom_tools_section}`` which the
        base will replace with auto-rendered descriptions of LM-query tools
        (when ``lm_callback`` is set) and ``rlm_config.custom_tools``.
        """
        return DEFAULT_RLM_SYSTEM_PROMPT

    def _get_repl_tools(self) -> Dict[str, Any]:
        """Return task-specific REPL helpers as ``{name: callable_or_value}``.

        Called after ``self._context`` is populated, so subclasses can return
        closures that capture per-rollout context. Default: empty.
        """
        return {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        extra_info = self.extras.get("extra_info", {}) if hasattr(self, "extras") else {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        # root_prompt: the user question shown every turn
        # context_text: data payload loaded into the REPL `context` variable
        root_prompt = self._extract_prompt_text(prompt)
        context_payload = extra_info.get("context_text") or root_prompt
        if isinstance(context_payload, str):
            try:
                decoded = json.loads(context_payload)
                if isinstance(decoded, dict):
                    context_payload = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        self._root_prompt = root_prompt
        self._context = context_payload

        # Subclass-supplied REPL tools (closures may capture self._context)
        subclass_tools = self._get_repl_tools()
        if subclass_tools:
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(subclass_tools)
            self.rlm_config.custom_tools = merged

        repl_timeout = self.rlm_config.parent_repl_timeout if self.subcall_fn is not None else self.rlm_config.repl_timeout
        self.repl = PersistentREPL(
            timeout=repl_timeout,
            custom_tools=self.rlm_config.custom_tools or {},
            lm_callback=self.lm_callback,
            subcall_fn=self.subcall_fn,
        )
        self.repl.add_context(context_payload, context_index=0)

        # Compute context metadata for the first user message
        metadata_text = _format_context_metadata(context_payload)

        system_content = self._build_system_prompt()

        # Turn-0 user prompt injection
        self._turn_index = 0
        turn0_prompt = _build_user_prompt(root_prompt, iteration=0)

        init_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": metadata_text},
        ]

        if self.step_wise:
            return init_messages, {"next_user_message": turn0_prompt}
        else:
            init_messages.append(turn0_prompt)

        return init_messages, {}

    def _build_system_prompt(self) -> str:
        template = self._get_system_prompt()

        custom_tools_section = ""
        if self.lm_callback is not None:
            custom_tools_section += (
                "\n4. LM query tools available in the REPL:\n"
                "- `llm_query(prompt)` — make a direct LLM call, returns str\n"
                "- `llm_query_batched(prompts)` — batch LLM calls, returns list[str]\n"
                "- `rlm_query(prompt)` — recursive LM call that spawns a child agent with its own REPL, returns str\n"
                "- `rlm_query_batched(prompts)` — batch recursive calls in parallel, returns list[str]"
            )
        if self.rlm_config.custom_tools:
            tools_formatted = _format_tools_for_prompt(self.rlm_config.custom_tools)
            if tools_formatted:
                section_num = 5 if self.lm_callback is not None else 4
                custom_tools_section += f"\n{section_num}. Custom tools and data available in the REPL:\n{tools_formatted}"

        return template.replace("{custom_tools_section}", custom_tools_section)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._turn_index += 1

        done = self.turns >= self.max_turns
        code = _find_code_block(action)

        if code is None:
            obs_text = "[No ```repl``` code block found. Wrap your code in ```repl\\n...\\n``` blocks.]"
            reward = self._get_reward(None) if done else 0.0
            if not done:
                next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
                if self.step_wise:
                    return BaseTextEnvStepOutput(
                        observations=[{"role": "user", "content": obs_text}],
                        next_user_message=next_prompt,
                        reward=reward,
                        done=done,
                        metadata={},
                    )
                else:
                    return BaseTextEnvStepOutput(
                        observations=[{"role": "user", "content": obs_text}, next_prompt],
                        next_user_message=None,
                        reward=reward,
                        done=done,
                        metadata={},
                    )

            return BaseTextEnvStepOutput(
                observations=[], next_user_message=None, reward=reward, done=done, metadata={}
            )

        _t_repl = time.perf_counter()
        result = self.repl.execute(code)
        self._last_repl_exec_s = time.perf_counter() - _t_repl

        # Two-stage final answer detection
        final_answer = result.final_answer  # set by FINAL_VAR() callable during execution
        if final_answer is None:
            final_answer = _find_final_answer(action, self.repl)
        if final_answer is not None:
            self._final_answer = final_answer
            done = True

        reward = self._get_reward(final_answer) if done else 0.0

        if done:
            return BaseTextEnvStepOutput(
                observations=[], next_user_message=None, reward=reward, done=True, metadata=self._build_metadata()
            )

        # Format REPL output observation
        result_str = _format_execution_result(result)
        if len(result_str) > _MAX_RESULT_LEN:
            result_str = result_str[:_MAX_RESULT_LEN] + f"... + [{len(result_str) - _MAX_RESULT_LEN} chars...]"
        repl_obs_text = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}"

        next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
        if self.step_wise:
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": repl_obs_text}],
                next_user_message=next_prompt,
                reward=reward,
                done=False,
                metadata=self._build_metadata(),
            )
        else:
            return BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": repl_obs_text}, next_prompt],
                next_user_message=None,
                reward=reward,
                done=False,
                metadata=self._build_metadata(),
            )

    def _extract_prompt_text(self, prompt: ConversationType) -> str:
        parts = [msg["content"] for msg in prompt if msg.get("content")]
        return "\n".join(parts)

    def _build_metadata(self) -> Dict[str, Any]:
        return {"turns": self.turns, "repl_exec_s": self._last_repl_exec_s}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns_used": self.turns,
            "final_value_set": self._final_answer is not None,
            "final_answer": self._final_answer,
            "reward": self._get_reward(self._final_answer),
        }

    def close(self):
        if self.repl is not None:
            self.repl.cleanup()
            self.repl = None
