import json
import re

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
# Per-turn user prompt injection
# ---------------------------------------------------------------------------

_USER_PROMPT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the prompt.\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "by writing to a ```repl``` tag, and determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    'to answer the original prompt: "{root_prompt}".\n\n'
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

# Matches: ```repl\n<code>\n```
_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
# Matches: FINAL_VAR(<expr>)  — non-greedy, first closing paren wins
_FINAL_VAR_RE = re.compile(r"^\s*FINAL_VAR\((.*?)\)", re.MULTILINE | re.DOTALL)
# Matches: FINAL(<anything>)  — greedy, entire rest of line
_FINAL_RE = re.compile(r"^\s*FINAL\((.*)\)\s*$", re.MULTILINE | re.DOTALL)


def _find_code_block(text: str) -> Optional[str]:
    """Return the LAST ```repl ... ``` code block in the response, or None."""
    matches = _REPL_BLOCK_RE.findall(text)
    return matches[-1].strip() if matches else None


def _find_final_answer(text: str, repl: Optional[PersistentREPL]) -> Optional[str]:
    """Parse FINAL_VAR(...) or FINAL(...) from the model's text response.

    Takes the LAST occurrence of either marker (FINAL_VAR preferred over FINAL
    when both are present).
    """
    var_matches = _FINAL_VAR_RE.findall(text)
    if var_matches:
        variable_name = var_matches[-1].strip().strip('"').strip("'")
        if repl is not None:
            result = repl.execute(f"print(FINAL_VAR({variable_name!r}))")
            answer = result.stdout.strip()
            if answer == "":
                return None
            if "Variable '" in answer and "' not found" in answer and "FINAL_VAR" in answer:
                return None
            return answer
        return None

    final_matches = _FINAL_RE.findall(text)
    if final_matches:
        return final_matches[-1].strip()

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
            lines.append(
                f"- `{name}`: {description}" if description else f"- `{name}`: A custom {type(value).__name__} value"
            )
    return "\n".join(lines) if lines else None


# ---------------------------------------------------------------------------
# Base environment
# ---------------------------------------------------------------------------


class BaseRLMEnv(BaseTextEnv):
    """Base class for Recursive Language Model (RLM) environments.

    Provides REPL plumbing, parent/child rollout wiring, the multi-turn loop,
    and FINAL/FINAL_VAR parsing. Task-specific behavior — reward, system
    prompt, REPL tools — is supplied by subclasses via three override hooks:

      • ``_get_reward(final_answer)``  — score the final answer (default: 0.0)
      • ``_get_system_prompt()``       — return the system prompt template
      • ``_get_repl_tools()``          — return task-specific REPL helpers
                                         (called after ``self._context`` is set,
                                         so closures can capture it)

    See ``examples/train/rlm/multi_paper_env/evidence_rlm_env.py`` for a worked example.

    All per-rollout knobs come through ``extras``:
      • ``repl_timeout`` — REPL execution timeout in seconds (default 180)
      • ``max_turns`` — turn budget (default 10)
      • ``lm_callback`` / ``subcall_fn`` — LM query callbacks injected by the generator
      • ``depth`` — rollout depth in a parent/child tree (default 0)

    Ephemeral user-prompt mechanism
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Each turn the model should see a per-turn instruction prompt (e.g.
    "Think step-by-step …") as the last user message, but previous turns'
    prompts must NOT accumulate in the chat history.

    This is achieved by mutating the generator's ``chat_history`` list
    through a stashed reference (``self._chat_history_ref``):

      • ``init()`` appends ``turn0_prompt`` to the returned list and stores
        the reference.
      • ``step()`` pops the stale prompt off the tail (it is still there
        because ``step`` runs *before* the generator appends new messages),
        then includes the next prompt as the last element of observations
        so the generator appends it at the tail for the next turn.

    NOTE: this relies on the generator never copying/rebinding
    ``chat_history`` after ``init()`` — it must remain the same list
    object.  The upstream ``SkyRLGymGenerator`` satisfies this.
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        self.max_turns = extras.get("max_turns", 10)

        self.lm_callback = extras.get("lm_callback", None)
        self.subcall_fn = extras.get("subcall_fn", None)

        self.repl: Optional[PersistentREPL] = None
        self._context: Any = None
        self._tools: Dict[str, Any] = {}
        self._final_answer: Optional[str] = None
        self._reward: float = 0.0
        self._turn_index = 0

        # Shared reference to the generator's chat_history list (set in init).
        self._chat_history_ref: Optional[ConversationType] = None

    # ------------------------------------------------------------------
    # Override hooks — subclasses customize task-specific behavior here
    # ------------------------------------------------------------------

    def _get_reward(self, final_answer: str) -> float:
        """Score the final answer. **Subclasses should override.**

        Only called when the rollout produced a final answer; turn-limit
        timeouts and other no-answer terminations score 0 without invoking
        this method.
        """
        return 1.0 if final_answer else 0.0

    def _get_system_prompt(self) -> str:
        """Return the system prompt template. Override for custom prompts.

        The returned string may contain ``{custom_tools_section}`` which the
        base will replace with auto-rendered descriptions of LM-query tools
        (when ``lm_callback`` is set) and the tools from ``_get_repl_tools()``.
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
        root_prompt = "\n".join(msg["content"] for msg in prompt if msg.get("content"))
        context_payload = self.extras.get("extra_info", {}).get("context_text") or root_prompt
        if isinstance(context_payload, str):
            try:
                decoded = json.loads(context_payload)
                if isinstance(decoded, dict):
                    context_payload = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        self._root_prompt = root_prompt
        self._context = context_payload

        self._tools = self._get_repl_tools() or {}

        self.repl = PersistentREPL(
            timeout=self.extras.get("repl_timeout", 180.0),
            custom_tools=self._tools,
            lm_callback=self.lm_callback,
            subcall_fn=self.subcall_fn,
        )
        self.repl.add_context(context_payload, context_index=0)

        metadata_text = _format_context_metadata(context_payload)
        system_content = self._build_system_prompt()

        self._turn_index = 0
        turn0_prompt = _build_user_prompt(root_prompt, iteration=0)

        init_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": metadata_text},
            turn0_prompt,
        ]
        # Stash reference — the generator will use this same list object as
        # chat_history throughout the episode, so we can pop the ephemeral
        # prompt off it in step().
        self._chat_history_ref = init_messages
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
        if self._tools:
            tools_formatted = _format_tools_for_prompt(self._tools)
            if tools_formatted:
                section_num = 5 if self.lm_callback is not None else 4
                custom_tools_section += (
                    f"\n{section_num}. Custom tools and data available in the REPL:\n{tools_formatted}"
                )

        return template.replace("{custom_tools_section}", custom_tools_section)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._turn_index += 1

        # Pop the previous turn's ephemeral user prompt from chat_history.
        # step() runs before the generator appends the new assistant+obs
        # messages, so the stale prompt is still the last element.
        self._chat_history_ref.pop()

        done = self.turns >= self.max_turns
        code = _find_code_block(action)

        # Branch 1: model didn't produce a repl block.
        if code is None:
            obs_text = "[No ```repl``` code block found. Wrap your code in ```repl\\n...\\n``` blocks.]"
            return self._make_step_output([{"role": "user", "content": obs_text}], done=done)

        # Branch 2: execute the repl block.
        result = self.repl.execute(code)

        # Two-stage final answer detection: FINAL_VAR() inside exec, or text-parsed FINAL/FINAL_VAR.
        final_answer = result.final_answer or _find_final_answer(action, self.repl)
        if final_answer is not None:
            self._final_answer = final_answer
            self._reward = self._get_reward(final_answer)
            return self._make_step_output([], done=True)

        # Hit max_turns without an answer: terminate with reward 0 (the default _reward).
        if done:
            return self._make_step_output([], done=True)

        # Otherwise emit the REPL output and continue.
        result_str = _format_execution_result(result)
        _MAX_RESULT_LEN = 20_000
        if len(result_str) > _MAX_RESULT_LEN:
            result_str = result_str[:_MAX_RESULT_LEN] + f"... + [{len(result_str) - _MAX_RESULT_LEN} chars...]"
        obs_text = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{result_str}"
        return self._make_step_output([{"role": "user", "content": obs_text}], done=False)

    def _make_step_output(self, observations: List[Dict[str, str]], done: bool) -> BaseTextEnvStepOutput:
        """Build a step output.

        When not done, appends the next turn's ephemeral user prompt to
        observations so the generator places it at the tail of chat_history.
        It will be popped at the start of the next step().
        """
        if not done:
            next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
            observations = observations + [next_prompt]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=self._reward if done else 0.0,
            done=done,
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns_used": self.turns,
            "final_value_set": self._final_answer is not None,
            "final_answer": self._final_answer,
            "reward": self._reward,
        }

    def close(self):
        if self.repl is not None:
            self.repl.cleanup()
            self.repl = None
