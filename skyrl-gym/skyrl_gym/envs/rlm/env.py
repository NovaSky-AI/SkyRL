import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.tools.repl import PersistentREPL, REPLResult


# ---------------------------------------------------------------------------
# System prompt (official RLM prompt from rlm/rlm/utils/prompts.py)
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
    "and determine your answer. Your next action:"
)
_USER_PROMPT_WITH_ROOT = (
    "Think step-by-step on what to do using the REPL environment (which contains the context) "
    "to answer the original prompt: \"{root_prompt}\".\n\n"
    "Continue using the REPL environment, which has the `context` variable, "
    "and determine your answer. Your next action:"
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


# ---------------------------------------------------------------------------
# QueryMetadata (from rlm/rlm/core/types.py)
# ---------------------------------------------------------------------------

class _QueryMetadata:
    def __init__(self, context_payload):
        if isinstance(context_payload, str):
            self.context_lengths = [len(context_payload)]
            self.context_type = "str"
        elif isinstance(context_payload, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in context_payload.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                else:
                    try:
                        self.context_lengths.append(len(json.dumps(chunk, default=str)))
                    except Exception:
                        self.context_lengths.append(len(repr(chunk)))
        elif isinstance(context_payload, list):
            self.context_type = "list"
            self.context_lengths = [len(str(c)) for c in context_payload]
        else:
            self.context_type = type(context_payload).__name__
            self.context_lengths = [len(repr(context_payload))]
        self.context_total_length = sum(self.context_lengths)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RLMEnvConfig:
    repl_timeout: float = 15.0
    custom_system_prompt: Optional[str] = None
    custom_tools: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RLMEnv(BaseTextEnv):
    """
    Recursive Language Model environment.

    init() returns:
        [system_msg, context_metadata_msg, turn_0_user_prompt_msg]

    step() returns observations:
        [repl_output_msg, turn_N_user_prompt_msg]

    The model always sees the per-turn user prompt as the last message before
    it generates, keeping root_prompt visible every turn.

    Context is loaded into the REPL via a temp file (add_context) so it
    appears as a genuine REPL variable in SHOW_VARS() and format_execution_result().

    Final answer is detected via:
      1. REPLResult.final_answer — set when FINAL_VAR() is called inside a repl block
      2. Text parsing of FINAL(...) / FINAL_VAR(...) in the model's response
    """

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = None):
        super().__init__()
        extras = extras or {}
        self.extras = extras

        assert "reward_spec" in extras, "reward_spec field is required"
        self.ground_truth = extras["reward_spec"].get("ground_truth")
        # reward_fn: optional callable(final_answer: str) -> float
        # Can be passed directly or will be built lazily in init() from reward_spec data.
        self.reward_fn = extras["reward_spec"].get("reward_fn")
        self.max_turns = extras.get("max_turns", 10)

        if isinstance(env_config, RLMEnvConfig):
            self.rlm_config = env_config
        elif isinstance(env_config, Mapping):
            self.rlm_config = RLMEnvConfig(**{k: v for k, v in env_config.items() if k in RLMEnvConfig.__dataclass_fields__})
        else:
            self.rlm_config = RLMEnvConfig()

        if extras.get("custom_system_prompt"):
            self.rlm_config.custom_system_prompt = extras["custom_system_prompt"]

        # Per-example custom tools (e.g. search/extract_section that close over context)
        # merged on top of any static config-level tools
        if extras.get("custom_tools"):
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(extras["custom_tools"])
            self.rlm_config.custom_tools = merged

        self.repl: Optional[PersistentREPL] = None
        self._final_answer: Optional[str] = None
        self._turn_index = 0  # iteration counter for build_user_prompt

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        extra_info = self.extras.get("extra_info", {}) if hasattr(self, "extras") else {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        # root_prompt: the user question shown every turn
        # context_text: data payload loaded into the REPL `context` variable
        root_prompt = self._extract_prompt_text(prompt)
        context_payload = extra_info.get("context_text") or root_prompt
        self._root_prompt = root_prompt

        # If reward_spec carries evidence spans but no reward_fn, build both the reward_fn
        # and per-example custom tools (search/extract_section) from the serializable data.
        # This keeps the dataset Parquet-serializable while still supporting callable-based reward.
        reward_spec = self.extras.get("reward_spec", {})
        if self.reward_fn is None and reward_spec.get("evidence") is not None:
            from skyrl_gym.envs.rlm.evidence_tools import make_reward_fn, make_tools
            evidence = reward_spec["evidence"]
            self.reward_fn = make_reward_fn(context_payload, evidence)
            tools = make_tools(context_payload)
            merged = dict(self.rlm_config.custom_tools or {})
            merged.update(tools)
            self.rlm_config.custom_tools = merged

        self.repl = PersistentREPL(
            timeout=self.rlm_config.repl_timeout,
            custom_tools=self.rlm_config.custom_tools or {},
        )
        self.repl.add_context(context_payload, context_index=0)

        # Compute context metadata for the first user message
        meta = _QueryMetadata(context_payload)
        metadata_text = (
            f"Your context is a {meta.context_type} with {meta.context_total_length} total characters, "
            f"and is broken up into chunks of char lengths: {meta.context_lengths}."
        )

        system_content = self._build_system_prompt()

        # Turn-0 user prompt injection
        self._turn_index = 0
        turn0_prompt = _build_user_prompt(root_prompt, iteration=0)

        init_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": metadata_text},
            turn0_prompt,
        ]
        return init_messages, {}

    def _build_system_prompt(self) -> str:
        template = self.rlm_config.custom_system_prompt or DEFAULT_RLM_SYSTEM_PROMPT

        custom_tools_section = ""
        if self.rlm_config.custom_tools:
            from skyrl_gym.tools.repl import format_tools_for_prompt
            tools_formatted = format_tools_for_prompt(self.rlm_config.custom_tools)
            if tools_formatted:
                custom_tools_section = f"\n4. Custom tools and data available in the REPL:\n{tools_formatted}"

        return template.format(custom_tools_section=custom_tools_section)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self._turn_index += 1

        done = self.turns >= self.max_turns
        code = _find_code_block(action)

        if code is None:
            obs_text = "[No ```repl``` code block found. Wrap your code in ```repl\\n...\\n``` blocks.]"
            if not done:
                next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
                obs = [
                    {"role": "user", "content": obs_text},
                    next_prompt,
                ]
            else:
                obs = []
            return BaseTextEnvStepOutput(
                observations=obs, reward=self._get_reward(done, None), done=done, metadata={}
            )

        result = self.repl.execute(code)

        # Two-stage final answer detection
        final_answer = result.final_answer  # set by FINAL_VAR() callable during execution
        if final_answer is None:
            final_answer = _find_final_answer(action, self.repl)
        if final_answer is not None:
            self._final_answer = final_answer
            done = True

        reward = self._get_reward(done, final_answer)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=True, metadata=self._build_metadata()
            )

        # Format REPL output observation
        result_str = _format_execution_result(result)
        if len(result_str) > _MAX_RESULT_LEN:
            result_str = result_str[:_MAX_RESULT_LEN] + f"... + [{len(result_str) - _MAX_RESULT_LEN} chars...]"
        repl_obs_text = f"Code executed:\n```repl\n{code}\n```\n\nREPL output:\n{result_str}"

        next_prompt = _build_user_prompt(self._root_prompt, self._turn_index)
        obs = [
            {"role": "user", "content": repl_obs_text},
            next_prompt,
        ]
        return BaseTextEnvStepOutput(
            observations=obs, reward=reward, done=False, metadata=self._build_metadata()
        )

    def _extract_prompt_text(self, prompt: ConversationType) -> str:
        parts = [msg["content"] for msg in prompt if msg.get("content")]
        return "\n".join(parts)

    def _get_reward(self, done: bool, final_answer: Optional[str]) -> float:
        if not done:
            return 0.0
        if final_answer is None:
            return 0.0

        if self.reward_fn is not None:
            return float(self.reward_fn(final_answer))

        final_str = str(final_answer).strip()
        gt_str = str(self.ground_truth).strip()

        if final_str == gt_str:
            return 1.0
        try:
            if abs(float(final_str) - float(gt_str)) < 1e-6:
                return 1.0
        except (ValueError, TypeError):
            pass
        if gt_str.lower() in final_str.lower():
            return 0.5
        return 0.0

    def _build_metadata(self) -> Dict[str, Any]:
        return {"turns": self.turns}

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "turns_used": self.turns,
            "final_value_set": self._final_answer is not None,
            "reward": self._get_reward(True, self._final_answer),
        }

    def close(self):
        if self.repl is not None:
            self.repl.cleanup()
            self.repl = None


if __name__ == "__main__":
    direct_prompt = "What is the capital of France? Extract it from the context."
    context_text = (
        "The population of France is approximately 67 million people. "
        "The capital city is Paris. France is known for the Eiffel Tower, "
        "fine cuisine, and its contributions to art and philosophy."
    )

    env = RLMEnv(
        env_config=RLMEnvConfig(repl_timeout=30.0),
        extras={
            "reward_spec": {"ground_truth": "Paris"},
            "max_turns": 5,
            "extra_info": {"context_text": context_text},
        },
    )

    prompt = [{"role": "user", "content": direct_prompt}]
    init_messages, info = env.init(prompt)

    print("=== Init Messages ===")
    for msg in init_messages:
        preview = msg["content"][:300].replace("\n", "\\n")
        print(f"  [{msg['role']}] {preview}...")

    action_1 = (
        "Let me inspect the context.\n\n"
        "```repl\nprint(context[:100])\n```"
    )
    step_out = env.step(action_1)
    print("\n=== Turn 1 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")
    for obs in step_out["observations"]:
        preview = obs["content"][:200].replace("\n", "\\n")
        print(f"  [{obs['role']}] {preview}")

    action_2 = (
        "I can see the answer is Paris.\n\n"
        "```repl\nmy_answer = \"Paris\"\nprint(my_answer)\n```\n\n"
        "FINAL_VAR(my_answer)"
    )
    step_out = env.step(action_2)
    print("\n=== Turn 2 ===")
    print(f"  Reward: {step_out['reward']}, Done: {step_out['done']}")

    print("\n=== Metrics ===")
    for k, v in env.get_metrics().items():
        print(f"  {k}: {v}")

    env.close()
