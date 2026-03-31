"""
Task Generation Environment for SkyRL.

Multi-turn BaseTextEnv where the LLM can explore the seed database via
``describe_db`` / ``query_db`` meta-tools before generating a task.

When ``max_turns > 1`` (the default), the model explores the DB first
and then produces a ``<task>`` block.  When ``max_turns == 1`` it
behaves identically to the original single-turn variant.

Reward:

    R(task) = base_quality + llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    base_quality:     Small reward for passing sandbox+judge (default 0.1)
    llm_validity:     Binary 0/1 from LLM-as-a-judge (is the task well-formed?)
    var(raw_scores):  Variance of k raw evaluator rollouts (difficulty calibration)
    p_hint - p_raw:   Hint gap — solvable with hints but not without (learnability)
    alpha:            Weight balancing variance vs hint gap (default 0.5)
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call, parse_tool_calls
from skyrl_gym.envs.task_gen.verifier_sandbox import (
    VerifierSandbox,
    parse_task_output,
)

logger = logging.getLogger(__name__)

# Meta-tools the model can call to explore the seed database.
_META_TOOLS = {"describe_db", "query_db"}

# All callable tools = meta-tools + any MCP env tools discovered at init time.
# Populated per-instance in init_async().


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Supports multi-turn: the model can explore the seed DB via ``describe_db``
    and ``query_db`` meta-tools before outputting a ``<task>`` block.

    Reward = llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    Evaluation uses Fleet harness jobs (POST /v1/jobs) to run an LLM agent
    against the generated task, rather than a stub evaluator.

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        max_turns: Max turns before forced termination (default 10)
        judge_model: Model ID for LLM-as-a-judge gate
        k_rollouts: Number of rollouts per condition (raw/hinted, default 4)
        max_eval_steps: Max agent steps per evaluator rollout (default 30)
        evaluator_model: Fleet harness model for task evaluation (default anthropic/claude-sonnet-4.5)
        base_quality_reward: Small reward for passing sandbox+judge (default 0.1).
            Prevents GRPO zero-signal deadlock when all harness evals fail.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Configurable multi-turn (default 10; set to 1 for single-turn)
        self.max_turns = int(env_config.get("max_turns", 10)) if env_config else 10

        # Fleet orchestrator for DB exploration (set in init_async)
        self.orch = None
        # MCP tools client for calling env tools (set in init_async)
        self.mcp_tools = None
        # Set of all callable tool names (meta-tools + MCP tools)
        self.callable_tools = set(_META_TOOLS)
        # Exploration sequence tracking (reset in init_async)
        self.called_describe_db = False
        self.called_query_db = False

        # Environment context from dataset (extras)
        self.env_key = extras.get("env_key") or extras.get("data_source", "unknown")
        self.env_version = extras.get("env_version", "")
        self.data_key = extras.get("data_key", "")
        self.data_version = extras.get("data_version", "")

        # Parse env_tools_schema (full tool schemas for prompt building)
        env_tools_schema_raw = extras.get("env_tools_schema", "[]")
        if isinstance(env_tools_schema_raw, str):
            try:
                self.env_tools_schema: List[Dict[str, Any]] = json.loads(env_tools_schema_raw)
            except json.JSONDecodeError:
                self.env_tools_schema: List[Dict[str, Any]] = []
        else:
            self.env_tools_schema: List[Dict[str, Any]] = env_tools_schema_raw or []

        # Parse env_tools (tool name list for sandbox validation)
        env_tools_raw = extras.get("env_tools", [])
        if isinstance(env_tools_raw, str):
            try:
                self.env_tools: List[str] = json.loads(env_tools_raw)
            except json.JSONDecodeError:
                self.env_tools: List[str] = []
        else:
            self.env_tools: List[str] = env_tools_raw or []

        # If env_tools is empty but we have schemas, extract names from schemas
        if not self.env_tools and self.env_tools_schema:
            self.env_tools = [
                t["function"]["name"] for t in self.env_tools_schema if "function" in t and "name" in t["function"]
            ]

        # Parse env_variable_keys (available context variables for this env)
        env_var_keys_raw = extras.get("env_variable_keys", "[]")
        if isinstance(env_var_keys_raw, str):
            try:
                self.env_variable_keys: List[str] = json.loads(env_var_keys_raw)
            except json.JSONDecodeError:
                self.env_variable_keys: List[str] = []
        else:
            self.env_variable_keys: List[str] = env_var_keys_raw or []

        # Parse env_variables (actual values for harness evaluation)
        env_vars_raw = extras.get("env_variables", "{}")
        if isinstance(env_vars_raw, str):
            try:
                self.env_variables: Dict[str, Any] = json.loads(env_vars_raw)
            except json.JSONDecodeError:
                self.env_variables: Dict[str, Any] = {}
        else:
            self.env_variables: Dict[str, Any] = env_vars_raw or {}

        # Parse env_schema (compact DB schema: table→columns)
        self.env_schema: str = extras.get("env_schema", "") or ""

        # Verifier sandbox — filters out CUA-only tool "computer" from available tools
        api_tools = set(self.env_tools) - {"computer"} if self.env_tools else None
        self.sandbox = VerifierSandbox(available_tools=api_tools if api_tools else None)

        # Judge config (from Hydra env_config)
        self.judge_model = str(env_config.get("judge_model", "")) if env_config else ""

        # Evaluator config (from Hydra env_config)
        self.k_rollouts = int(env_config.get("k_rollouts", 4)) if env_config else 4
        self.max_eval_steps = int(env_config.get("max_eval_steps", 30)) if env_config else 30
        self.evaluator_model = (
            str(env_config.get("evaluator_model", "anthropic/claude-sonnet-4.5"))
            if env_config
            else "anthropic/claude-sonnet-4.5"
        )

        # API keys from environment variables (set by SkyPilot YAML)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.fleet_api_key = os.environ.get("FLEET_API_KEY", "")

        # Eval mode: k=8 raw only (no hints)
        self.is_eval = extras.get("training_phase") == "eval"
        self.eval_k_rollouts = int(env_config.get("eval_k_rollouts", 8)) if env_config else 8
        # Whether to run hinted evaluation jobs (2nd harness job with verifier feedback).
        # Default off — hints were net negative in iter#11 (verifier code dump confused evaluator).
        self.enable_hints = bool(env_config.get("enable_hints", False)) if env_config else False

        # Lazy-init Fleet SDK client for harness evaluation
        self._fleet_client = None

        # Rollout dump directory (full prompt/verifier/scores per eval)
        default_rollout_dir = os.path.join(os.path.expanduser("~"), "reward_rollouts")
        self._rollout_dir = os.environ.get("REWARD_ROLLOUT_DIR", default_rollout_dir)
        os.makedirs(self._rollout_dir, exist_ok=True)

        # Base quality reward for tasks passing sandbox + judge gate.
        # Provides GRPO gradient signal even when all harness evals return 0.
        self.base_quality_reward = float(env_config.get("base_quality_reward", 0.1)) if env_config else 0.1

        # Small per-tool-call reward to incentivize DB exploration (describe_db, query_db).
        # Default 0.0 = off (no behavior change for existing runs).
        self.tool_call_reward_per_call = float(env_config.get("tool_call_reward_per_call", 0.0)) if env_config else 0.0

        logger.info(
            f"TaskGenEnv: env={self.env_key}, max_turns={self.max_turns}, "
            f"judge={self.judge_model or 'none'}, "
            f"tools={len(self.env_tools)}, k={self.k_rollouts}, eval_k={self.eval_k_rollouts}, "
            f"evaluator={self.evaluator_model}, is_eval={self.is_eval}, "
            f"base_quality={self.base_quality_reward}, tool_call_reward={self.tool_call_reward_per_call}"
        )

    def _format_tool_schema(self, tool: Dict[str, Any]) -> str:
        """Format a single tool schema for the system prompt."""
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        lines = [f"**{name}**: {desc}"]
        if properties:
            lines.append("  Parameters:")
            for pname, pschema in properties.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req_marker = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req_marker}): {pdesc}")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with environment context and priors."""
        parts = []

        parts.append(f'You are a task designer for the "{self.env_key}" environment.')

        # --- Date context (critical for date-sensitive environments) ---
        current_date = self.env_variables.get("CURRENT_DATE", "")
        if current_date:
            parts.append(
                f"\n**IMPORTANT — Current Date: {current_date}**\n"
                f"The environment's current date is {current_date}. "
                "All dates in generated tasks MUST be on or after this date. "
                "Do NOT use past dates — the environment will reject them "
                "(e.g., check-in dates, event dates, appointment dates must be in the future)."
            )

        # --- A. Environment context (from tool discovery) ---
        parts.append(f"\n## Environment: {self.env_key}")
        parts.append("\n### Available Tools")

        # Filter out CUA-only "computer" tool — task-gen is for tool-use APIs
        api_schemas = [t for t in self.env_tools_schema if t.get("function", {}).get("name") != "computer"]
        api_tool_names = [t for t in self.env_tools if t != "computer"]

        if api_schemas:
            # Compact format: name + description only (no parameter schemas)
            # Full schemas make the prompt too long for envs with many tools
            for tool in api_schemas:
                func = tool.get("function", {})
                name = func.get("name", "unknown")
                desc = func.get("description", "")
                parts.append(f"- **{name}**: {desc}")
        elif api_tool_names:
            parts.append("\n".join(f"- {t}" for t in api_tool_names))
        else:
            parts.append("No tools discovered for this environment.")

        # Environment variables (user context available at task runtime)
        if self.env_variables:
            parts.append("\n### Environment Variables (embed as constants)")
            parts.append(
                "These variables describe the user/session context. "
                "**Embed them directly as string constants** in your verifier code. "
                "Do NOT use `env.env_variables` — it is not available at verifier runtime."
            )
            for var_key, var_val in self.env_variables.items():
                parts.append(f'- `{var_key}` = `"{var_val}"`')
            parts.append(
                "\nExample usage in verifier:\n"
                "```python\n"
                f'LOGGED_IN_USER = "{self.env_variables.get("LOGGED_IN_USER", "user@example.com")}"\n'
                f'# Use as: rows = current.table("users").eq("email", LOGGED_IN_USER).all()\n'
                "```"
            )
        elif self.env_variable_keys:
            parts.append("\n### Environment Variables")
            parts.append(
                "These variables parameterize each environment instance. "
                "Look up values from the database instead of using env.env_variables."
            )
            for var_key in self.env_variable_keys:
                parts.append(f"- `{var_key}`")

        # Database schema (table names and columns)
        if self.env_schema:
            parts.append("\n### Database Schema")
            parts.append(
                "Use these exact table and column names in verifiers "
                '(e.g., `current.table("bookings").eq("guest_email", val).all()`):'
            )
            parts.append(f"```\n{self.env_schema}\n```")

        # --- B. Priors (concise, static, same for all envs) ---
        # Date awareness guidance (prevents past-date failures in booking/ticketmaster)
        if current_date:
            date_guidance = (
                f"### Date Awareness\n"
                f"The environment's current date is **{current_date}**. "
                f"ALL dates in your task MUST be on or after {current_date}. "
                "Tasks with past dates will always fail because the environment "
                "rejects them (e.g., 'checkIn date cannot be in the past'). "
                "Use `query_db` to check what date ranges exist in the data, "
                "and always generate future dates."
            )
        else:
            date_guidance = (
                "### Date Awareness\n"
                "If the environment works with dates, verify what date ranges "
                "are valid before generating tasks. Use `query_db` to check."
            )

        # NOTE: env.env_variables is NOT available at verifier runtime (Fleet harness bug).
        # Model is instructed to embed env var values as constants instead.

        parts.append(
            f"""
## Verifier Guidelines

The verifier checks whether the agent completed the task by inspecting database state changes.

Signature: `def validate_task(env: Environment, final_answer: str | None = None) -> int`

**IMPORTANT**: The function MUST be named `validate_task` and return `TASK_FAILED_SCORE` (0) or `TASK_SUCCESSFUL_SCORE` (1).

### Verifier API
```python
env.instance.load()              # Load current state (call first)
seed = env.db("seed")            # Original DB before agent acted
current = env.db("current")      # Current DB after agent acted

# Query tables — ALL results are Python dicts, use row["column"] NOT row.column:
rows = current.table("table_name").eq("column", value).all()   # -> List[dict]
row = current.table("table_name").eq("column", value).first()  # -> dict or None
rows = current.table("table_name").neq("column", value).all()  # -> List[dict]
count = current.table("table_name").eq("column", value).count() # -> int
rows = current.table("table_name").select("col1", "col2").all() # -> List[dict]
# Access fields: row["id"], row["name"], row["email"] — NEVER row.id or row.name
# Only methods: .table(), .eq(), .neq(), .select(), .all(), .first(), .count()
# NO .like(), .gt(), .lt(), .contains(), .in_() — use Python filtering instead

# Compare seed vs current to detect NEW entries:
def find_new_entries(seed, current, table_name, id_field="id", filter_conditions=None):
    before_query = seed.table(table_name)
    after_query = current.table(table_name)
    if filter_conditions:
        for key, value in filter_conditions.items():
            before_query = before_query.eq(key, value)
            after_query = after_query.eq(key, value)
    before_ids = {{entry[id_field] for entry in before_query.select(id_field).all()}}
    return [e for e in after_query.all() if e[id_field] not in before_ids]
```

### Error Tracking (REQUIRED)
Every verifier MUST track errors and successes using accumulator lists, and print them
before returning. This enables automated feedback for hint-based evaluation.

```python
error_accumulator = []
success_accumulator = []

# ... check conditions ...
if condition_met:
    success_accumulator.append("[C] Booking was created")
else:
    error_accumulator.append("[X] Expected booking not found")

# ALWAYS print accumulators before returning:
if error_accumulator:
    print(">>> ERROR_ACCUMULATOR >>>")
    print(error_accumulator)
    print("<<< ERROR_ACCUMULATOR <<<")
if success_accumulator:
    print(">>> SUCCESS_ACCUMULATOR >>>")
    print(success_accumulator)
    print("<<< SUCCESS_ACCUMULATOR <<<")
```

### Verifier Template (follow this structure)
```python
def validate_task(env: Environment, final_answer: str | None = None) -> int:
    error_accumulator = []
    success_accumulator = []
    env.instance.load()
    seed = env.db("seed")
    current = env.db("current")

    def find_new_entries(table_name, id_field="id", filter_conditions=None):
        before_query = seed.table(table_name)
        after_query = current.table(table_name)
        if filter_conditions:
            for key, value in filter_conditions.items():
                before_query = before_query.eq(key, value)
                after_query = after_query.eq(key, value)
        before_ids = set(entry[id_field] for entry in before_query.select(id_field).all())
        return [e for e in after_query.all() if e[id_field] not in before_ids]

    # Check conditions...
    # On early failure:
    if critical_failure:
        error_accumulator.append("[X] Critical check failed")
        print(">>> ERROR_ACCUMULATOR >>>")
        print(error_accumulator)
        print("<<< ERROR_ACCUMULATOR <<<")
        return TASK_FAILED_SCORE

    # Final result:
    if error_accumulator:
        print(">>> ERROR_ACCUMULATOR >>>")
        print(error_accumulator)
        print("<<< ERROR_ACCUMULATOR <<<")
        return TASK_FAILED_SCORE
    print(">>> SUCCESS_ACCUMULATOR >>>")
    print(success_accumulator)
    print("<<< SUCCESS_ACCUMULATOR <<<")
    return TASK_SUCCESSFUL_SCORE
```

### Rules
- **NEVER hardcode database IDs** (user_id, hotel_id, etc.) — always query the DB to find them
- **NEVER use `env.env_variables`** — it is not available at runtime. Embed env var values as string constants at the top of your verifier (e.g., `LOGGED_IN_USER = "riley3318"`)
- **DB rows are dicts** — use `row["id"]`, `row["name"]`, NOT `row.id`, `row.name`. Using dot notation will crash with `AttributeError: 'dict' object has no attribute 'id'`
- **Only use supported query methods**: `.eq()`, `.neq()`, `.select()`, `.all()`, `.first()`, `.count()`. NO `.like()`, `.gt()`, `.lt()`, `.order()`, `.limit()`, `.contains()`, `.in_()` — filter and sort in Python instead (e.g., `sorted([r for r in rows if r["score"] > 8.0], key=lambda r: r["score"], reverse=True)[:5]`)
- **`.eq()` takes exactly 2 args**: `.eq(column, value)`. NO operator arg like `.eq("rating", ">", 8)` — use Python: `[r for r in rows if r["rating"] > 8]`
- **Use timezone-tolerant comparisons** for datetimes — the DB may store `"2025-08-08T14:00:00Z"` while you expect `"2025-08-08T14:00:00"`. Use `.startswith()` or strip the trailing `"Z"` before comparing
- **If you use `.select()`, only access the selected columns** — accessing other columns raises `KeyError`. Prefer `.all()` without `.select()` unless you specifically need to limit columns
- **Define `find_new_entries` inside your verifier function** — it is NOT a built-in. Copy it from the template above into your `validate_task()` function body. Do NOT call `find_new_entries()` without defining it first
- **List comprehensions produce tuples if you use tuple syntax** — `[(a, b) for ...]` creates tuples, not dicts. If you need dict-like access later, keep the original dicts: `[row for row in rows if condition]`
- **NEVER hardcode expected values the agent must create** — e.g., don't check for a specific phone number or email the agent would need to invent. Instead, check that the field was changed from its original value: `current_val != seed_val`
- Look up the logged-in user by name/email from the users table, don't assume an ID
- Compare `seed` (before) vs `current` (after) to detect what the agent did
- Must return `TASK_FAILED_SCORE` on a fresh environment (before agent acts)
- Use `final_answer` for tasks that require the agent to report a value
- Reference actual tool names from this environment

## Task Design Guidelines

Design tasks that maximize learnability: an ideal task is one that a capable agent can solve with effort, but not trivially. Tasks that are too easy (always solved) or too hard (never solved) produce no learning signal.

{date_guidance}

### Realism
Write prompts as a real user would — natural language, concrete parameters, plausible intent. The task should sound like something a person would actually ask, not a test case.

BAD:  "Call get_user with id=5, then call update_user to set email to test@example.com"
GOOD: "Update the email address for Jamie Chen to jamie.chen@newdomain.com"

### Avoiding Underspecification
A prompt is underspecified when multiple valid solutions exist but the verifier only accepts one. This creates false negatives — the agent solves the task correctly but gets reward 0.

BAD prompt:  "Find a designer in Mexico" (3 designers exist, verifier checks for one specific one)
FIX option 1: Make the prompt specific: "Find the designer in Mexico City who joined after 2023"
FIX option 2: Make the verifier accept all valid answers: check that ANY designer in Mexico is returned

Use `describe_db`/`query_db` to check the actual data before writing the prompt. If a query returns multiple rows, either narrow the prompt or widen the verifier. Always verify your assumptions by querying — don't guess. You MUST call all three of `describe_db`, `query_db`, and at least one environment API tool before writing the task — your task will be rejected otherwise.

### Avoiding Overspecification
A prompt is overspecified when it dictates HOW to accomplish the task rather than WHAT outcome is needed. This makes the task trivially easy (no learning signal) and doesn't test real problem-solving.

BAD:  "First call list_tables, then call get_bookings with check_in_date='2024-03-15', then count the results and call submit_answer with the count"
GOOD: "How many bookings have a check-in date of March 15, 2024?"

The prompt should specify the desired outcome. The agent should figure out which tools to use and in what order.

### Complexity
Aim for tasks solvable in 2-8 tool calls. Tasks requiring 1 tool call are too easy (no signal). Tasks requiring 15+ calls are too hard (agent gives up). The sweet spot is 3-6 calls with some reasoning required.

### Diversity
Vary tasks across multiple dimensions:
- Operations: reads (lookup, search, aggregate) AND writes (create, update, delete)
- Complexity: simple (2-3 tool calls) through moderate (4-8 tool calls with dependencies)
- Reasoning: some tasks need multi-step logic (find X, use X to look up Y, modify Y based on Z)
- Data entities: use different tables, columns, and relationships in the schema

### Verifier-Prompt Consistency
The verifier must check exactly what the prompt asks — no more, no less. Before writing, verify:
1. Is there exactly one correct outcome for this prompt? (If not, widen the verifier or narrow the prompt)
2. Does the verifier return 0.0 on a fresh environment? (It must — the agent hasn't acted yet)
3. Does the verifier avoid hardcoded values? (Query the DB instead)
4. Could a different valid approach fool the verifier? (If so, fix the verifier to accept it)"""
        )

        # --- C. Exploration tools (multi-turn only) ---
        if self.max_turns > 1:
            parts.append(
                """
## Exploration Tools

Before generating a task, explore the environment to understand the actual data and API behavior.

### Database Tools
<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>
Returns the full schema: table names, columns, types.

<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT * FROM table_name LIMIT 5"}}</tool_call>
Runs a read-only SQL query against the seed database.

### Environment Tools
You MUST call at least one of the environment's API tools listed above to understand their input/output formats.

**REQUIRED before generating a task:** You must call ALL THREE of: (1) `describe_db`, (2) `query_db`, and (3) at least one environment API tool. Your task will be rejected if any are missing.

<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
Calls the tool and returns its result. Use this to understand input/output formats.

### Workflow
1. **Explore**: Call `describe_db` to see all tables and columns.
2. **Inspect data**: Call `query_db` with SELECT queries to inspect real data (values, ranges, row counts, patterns).
3. **Try tools**: Call at least one environment API tool to understand its behavior, input/output formats, and edge cases.
4. **Draft a task idea**: Think about what prompt + verifier you could write based on the data you've seen.
5. **Validate your draft**: Before outputting the task, run `query_db` to verify your assumptions:
   - Does the data your prompt references actually exist? (e.g., "Update Jamie's email" — is there a Jamie?)
   - Will the verifier return 0.0 on a fresh DB? (Check seed state)
   - Are there edge cases? (e.g., multiple matches, null values, empty tables)
6. **Iterate**: If your queries reveal problems (wrong assumptions, ambiguous data, too many/few matches), revise your task idea and verify again. Do NOT output the task until you've confirmed the data supports it.
7. **Output**: Only when confident, output the final task in the format below."""
            )

        # --- D. Few-shot examples removed ---
        # Few-shot examples were removed because they anchored the model to
        # generate near-copies of the examples (especially booking/wishlist tasks),
        # causing mode collapse and zero reward signal. The verifier template +
        # guidelines above provide enough structure for the model to generate
        # diverse tasks from the actual DB schema and tools.

        # --- E. Output format ---
        parts.append(
            """
## Output Format

Generate exactly ONE task. Output it in this format:

<task>
<prompt>
[Natural language task instruction for the agent. Be specific about what needs to be done.]
</prompt>
<verifier>
[Python function: def validate_task(env, final_answer=None) -> int]
</verifier>
</task>"""
        )

        return "\n".join(parts)

    def _judge_task(self, prompt: str, verifier: str) -> float:
        """LLM classifier gate: returns 0.0 (reject) or 1.0 (accept).

        Predicts whether the (prompt, verifier) pair will produce meaningful
        evaluation signal. Optimized for very low false positive rate — only
        rejects tasks that are near-certain to waste harness compute.

        Checks:
            1. Phantom tables: verifier references tables not in env schema
            2. Undefined references: calls to functions/constants not defined
            3. Vacuous checks: verifier only checks user existence or len>0
        """
        if not self.judge_model or not self.openrouter_api_key:
            return 1.0  # No judge configured, pass through

        # Build context for the classifier
        tool_names = [t for t in self.env_tools if t != "computer"]
        tools_str = ", ".join(tool_names[:20]) if tool_names else "none discovered"

        schema_block = self.env_schema if self.env_schema else "Schema not available."

        judge_prompt = (
            "You are a pre-filter that decides whether to run an expensive evaluation harness "
            "on a generated (prompt, verifier) pair. Your goal: reject tasks that will CERTAINLY "
            "produce zero useful signal, while accepting anything that MIGHT work.\n\n"
            "ACCEPT unless you find a clear, concrete defect from this list:\n\n"
            "1. PHANTOM TABLES — The verifier calls .table(\"X\") where X does not exist in the "
            "schema below. If the table doesn't exist, the verifier crashes → zero signal.\n\n"
            "2. UNDEFINED REFERENCES — The verifier calls functions (e.g., find_new_entries()) or "
            "uses constants (e.g., TASK_FAILED_SCORE) that are not defined in the code and are "
            "not Python builtins. These cause NameError → zero signal.\n\n"
            "3. VACUOUS CHECKS — The verifier's ONLY checks are whether a user/account exists or "
            "whether any table has rows (len > 0), without validating any task-specific outcome. "
            "These always pass regardless of agent behavior → zero variance → zero signal.\n\n"
            "IMPORTANT: When in doubt, ACCEPT. A false reject wastes a potentially good task. "
            "A false accept only wastes one harness call. Err heavily toward ACCEPT.\n\n"
            f'--- Environment: "{self.env_key}" ---\n\n'
            f"Available tools: {tools_str}\n\n"
            f"Database schema (these are the ONLY valid tables):\n```\n{schema_block}\n```\n\n"
            f"--- Generated task ---\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Verifier:\n```python\n{verifier}\n```\n\n"
            "Answer with exactly one word: ACCEPT or REJECT"
        )

        try:
            import litellm

            response = litellm.completion(
                model=f"openrouter/{self.judge_model}",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10,
                api_key=self.openrouter_api_key,
            )
            answer = response.choices[0].message.content.strip().upper()
            accepted = "ACCEPT" in answer and "REJECT" not in answer
            logger.info(
                f"LLM classifier [{self.env_key}]: {answer} -> "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            )
            return 1.0 if accepted else 0.0
        except Exception as e:
            logger.warning(f"LLM classifier failed, defaulting to accept: {e}")
            return 1.0

    @staticmethod
    def _build_hint_text(
        verifier_stdout: Optional[str],
        verifier_error: Optional[str],
        tool_error_messages: Optional[List[str]],
    ) -> str:
        """Build hint text from verifier feedback. No LLM call.

        Parses ERROR_ACCUMULATOR / SUCCESS_ACCUMULATOR from verifier stdout
        and formats tool errors into structured feedback for hinted rollouts.
        """
        parts: List[str] = []

        if verifier_stdout:
            err_match = re.search(
                r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            suc_match = re.search(
                r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n<<< SUCCESS_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            if err_match or suc_match:
                try:
                    errors = ast.literal_eval(err_match.group(1)) if err_match else []
                    successes = ast.literal_eval(suc_match.group(1)) if suc_match else []
                except Exception:
                    errors, successes = [], []
                if successes:
                    parts.append(f"Checks passed ({len(successes)}): " + ", ".join(str(s)[:100] for s in successes[:5]))
                if errors:
                    parts.append(f"Checks failed ({len(errors)}): " + ", ".join(str(e)[:100] for e in errors[:5]))

        if verifier_error:
            parts.append(f"Verifier: {verifier_error}")

        if tool_error_messages:
            unique = list(dict.fromkeys(tool_error_messages))[:5]
            parts.append("Tool errors: " + "; ".join(e[:200] for e in unique))

        return "\n".join(parts) if parts else "The previous attempt failed. Try a different approach."

    def _get_fleet_client(self):
        """Lazy-init Fleet SDK client."""
        if self._fleet_client is None:
            from fleet import Fleet

            self._fleet_client = Fleet(api_key=self.fleet_api_key)
        return self._fleet_client

    async def _poll_job(self, fleet, job_id: str, poll_interval: int = 10, timeout: int = 600) -> str:
        """Poll Fleet job until completion or timeout.

        Returns:
            Final job status string.
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                job = fleet.get_job(job_id)
                status = job.status
                if status in ("completed", "cancelled", "errored"):
                    return status
            except Exception as e:
                logger.warning(f"Error polling job {job_id}: {e}")
            await asyncio.sleep(poll_interval)

        logger.error(f"Job {job_id} timed out after {timeout}s")
        return "timeout"

    def _query_supabase_scores(self, job_id: str) -> Dict[str, float]:
        """Query Supabase for session verifier scores as fallback.

        When Fleet backend doesn't populate verifier_execution FK (regression
        since 2026-03-23), the score is still available in session metadata.

        Returns:
            Dict mapping session_id -> verifier_score.
        """
        supabase_url = os.environ.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if not supabase_url or not supabase_key:
            return {}
        try:
            import httpx

            resp = httpx.get(
                f"{supabase_url}/rest/v1/sessions",
                params={"job_id": f"eq.{job_id}", "select": "id,metadata"},
                headers={
                    "apikey": supabase_key,
                    "Authorization": f"Bearer {supabase_key}",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"Supabase query failed: {resp.status_code}")
                return {}
            scores = {}
            for row in resp.json():
                meta = row.get("metadata") or {}
                sid = row.get("id")
                v_score = meta.get("verifier_score")
                if sid and v_score is not None:
                    scores[sid] = float(v_score)
            return scores
        except Exception as e:
            logger.warning(f"Supabase fallback failed: {e}")
            return {}

    def _extract_job_results(self, fleet, job_id: str) -> List[Tuple[float, Optional[str], Optional[str]]]:
        """Extract (score, verifier_stdout, verifier_error) from completed job sessions.

        Primary path: read from session.verifier_execution (Fleet SDK).
        Fallback: query Supabase for metadata.verifier_score when VE is null
        (Fleet backend regression since 2026-03-23 stopped populating VE FK).

        Returns:
            List of (score, stdout, error) tuples per session.
        """
        results: List[Tuple[float, Optional[str], Optional[str]]] = []
        sessions_response = fleet.list_job_sessions(job_id)

        # Check if any session has verifier_execution populated
        all_ve_null = all(s.verifier_execution is None for tg in sessions_response.tasks for s in tg.sessions)

        # Fallback: query Supabase only when needed
        supabase_scores: Dict[str, float] = {}
        if all_ve_null:
            supabase_scores = self._query_supabase_scores(job_id)
            if supabase_scores:
                logger.info(f"[{job_id[:8]}] Using Supabase fallback for {len(supabase_scores)} session scores")

        for task_group in sessions_response.tasks:
            for session in task_group.sessions:
                score = 0.0
                stdout = None
                error = None
                if session.verifier_execution:
                    if session.verifier_execution.score is not None:
                        score = float(session.verifier_execution.score)
                    elif session.verifier_execution.success:
                        score = 1.0
                    stdout = getattr(session.verifier_execution, "stdout", None)
                    # Capture error from verifier crashes — error is nested in result.error
                    ve_result = getattr(session.verifier_execution, "result", None)
                    if ve_result:
                        ve_error = (
                            ve_result.get("error") if isinstance(ve_result, dict) else getattr(ve_result, "error", None)
                        )
                        if ve_error:
                            error = (
                                ve_error.get("message", "")
                                if isinstance(ve_error, dict)
                                else getattr(ve_error, "message", "")
                            )
                            traceback_str = (
                                ve_error.get("traceback", "")
                                if isinstance(ve_error, dict)
                                else getattr(ve_error, "traceback", "")
                            )
                            if traceback_str:
                                # Extract just the last line of traceback (the actual error)
                                error = traceback_str.strip().split("\n")[-1] if traceback_str else error
                elif session.session_id in supabase_scores:
                    # Fallback: use Supabase metadata.verifier_score
                    score = supabase_scores[session.session_id]
                results.append((score, stdout, error))
        return results

    async def _run_harness_job(
        self, prompt: str, verifier: str, k: int
    ) -> List[Tuple[float, Optional[str], Optional[str]]]:
        """Run a single Fleet harness job and return per-session results + job ID.

        1. Import task to Fleet
        2. Create harness job with pass_k=k
        3. Poll until completion
        4. Extract results

        Returns:
            Tuple of (job_id, results) where results is a list of
            (score, verifier_stdout, verifier_error) tuples.
            job_id is None on failure.
        """
        from fleet.tasks import Task

        fleet = self._get_fleet_client()
        task_key = f"taskgen_{uuid.uuid4().hex[:12]}"

        task = Task(
            key=task_key,
            prompt=prompt,
            env_id=self.env_key,
            version=self.env_version or None,
            verifier_func=verifier,
            data_id=self.data_key or None,
            data_version=self.data_version or None,
            env_variables=self.env_variables or {},
        )

        try:
            import_response = fleet.import_single_task(task)
        except Exception as e:
            logger.error(f"[{task_key}] Failed to import task to Fleet: {e}")
            return (None, [(0.0, None, None)] * k)
        if import_response is None:
            logger.error(f"[{task_key}] Failed to import task to Fleet (returned None, api_key set: {bool(self.fleet_api_key)})")
            return (None, [(0.0, None, None)] * k)

        job_response = fleet.create_job(
            models=[self.evaluator_model],
            task_keys=[task_key],
            pass_k=k,
            max_steps=self.max_eval_steps,
            mode="tool-use",
            name=f"taskgen-eval-{task_key}",
        )
        job_id = job_response.job_id
        logger.info(f"[{task_key}] Harness job created: {job_id} (model={self.evaluator_model}, k={k})")

        status = await self._poll_job(fleet, job_id)
        if status != "completed":
            logger.warning(f"[{task_key}] Job {job_id} ended with status: {status}")
            return (job_id, [(0.0, None, None)] * k)

        return (job_id, self._extract_job_results(fleet, job_id))

    async def _evaluate_task(self, prompt: str, verifier: str) -> Dict[str, float]:
        """Run hint-based evaluation via Fleet harness jobs.

        1. Raw job: k rollouts without hints
        2. Build hint from first failing session's verifier stdout
        3. Hinted job: k rollouts with hint appended to prompt
        4. Compute reward via compute_task_reward()

        Returns:
            Reward breakdown dict from compute_task_reward.
        """
        from integrations.fleet.task_gen_reward import compute_task_reward

        zero_result = compute_task_reward([], [], validity=1.0)

        if not self.fleet_api_key:
            return zero_result

        task_id = f"taskgen_{uuid.uuid4().hex[:8]}"
        start = time.time()

        try:
            # Eval: k=eval_k_rollouts for pass rate; Train: k=k_rollouts
            eval_k = self.eval_k_rollouts if self.is_eval else self.k_rollouts

            # 1. Raw job: k rollouts without hints
            raw_job_id, raw_results = await self._run_harness_job(prompt, verifier, k=eval_k)
            raw_scores = [r[0] for r in raw_results]

            if self.enable_hints and not self.is_eval:
                # Hinted training: k raw + k hinted for hint_gap signal
                # 2. Build hint from first failing session's stdout/error
                hint_stdout = None
                hint_error = None
                for score, stdout, error in raw_results:
                    if score < 1.0:
                        if stdout:
                            hint_stdout = stdout
                        if error:
                            hint_error = error
                        if hint_stdout or hint_error:
                            break
                hint_text = self._build_hint_text(hint_stdout, hint_error, None)

                # Fallback: if hint is generic (no VE stdout due to backend regression),
                # use the verifier source code as the hint. This tells the hinted agent
                # exactly what checks to satisfy, creating hint_gap signal.
                if hint_text == "The previous attempt failed. Try a different approach.":
                    # Truncate verifier to avoid blowing up prompt length
                    verifier_hint = verifier[:2000]
                    hint_text = (
                        "Here is the verification function that will be used to check your work. "
                        "Make sure your actions satisfy all the checks:\n\n"
                        f"```python\n{verifier_hint}\n```"
                    )

                # 3. Hinted job: k rollouts with hint
                hinted_prompt = f"{prompt}\n\nHere is feedback from a previous attempt to help you:\n{hint_text}"
                hinted_job_id, hinted_results = await self._run_harness_job(hinted_prompt, verifier, k=self.k_rollouts)
                hinted_scores = [r[0] for r in hinted_results]

                # 4. Compute reward
                result = compute_task_reward(raw_scores, hinted_scores, validity=1.0)
            else:
                # No hints — reward based on raw variance only
                hinted_scores = []
                hinted_job_id = None
                hint_text = ""
                result = compute_task_reward(raw_scores, raw_scores, validity=1.0)

            duration = time.time() - start

            # --- Iron-clad eval logging ---
            # Truncate prompt/verifier for log readability
            prompt_log = prompt[:300].replace("\n", " ")
            verifier_log = verifier[:200].replace("\n", " ")
            hint_log = hint_text[:200].replace("\n", " ")
            logger.info(
                f"[{task_id}] EVAL | "
                f"raw_job={raw_job_id} hinted_job={hinted_job_id} | "
                f"raw={raw_scores} hinted={hinted_scores} | "
                f"var={result['var_raw']:.4f} gap={result['hint_gap']:.4f} total={result['total']:.4f} | "
                f"time={duration:.0f}s | "
                f"prompt={prompt_log} | "
                f"verifier={verifier_log} | "
                f"hint={hint_log}"
            )

            # Save full rollout to local JSONL
            self._save_rollout(
                task_id=task_id,
                env_key=self.env_key,
                data_key=self.data_key,
                prompt=prompt,
                verifier=verifier,
                hint=hint_text,
                raw_scores=raw_scores,
                hinted_scores=hinted_scores,
                raw_job_id=raw_job_id,
                hinted_job_id=hinted_job_id,
                result=result,
                duration=duration,
            )

            return result

        except Exception as e:
            logger.error(f"[{task_id}] Evaluation failed: {e}")
            return zero_result

    def _save_rollout(
        self,
        task_id,
        env_key,
        data_key,
        prompt,
        verifier,
        hint,
        raw_scores,
        hinted_scores,
        raw_job_id,
        hinted_job_id,
        result,
        duration,
    ):
        """Append full rollout data to a local JSONL file."""
        try:
            run_name = os.environ.get("RUN_NAME", "unknown")
            path = os.path.join(self._rollout_dir, f"{run_name}.jsonl")
            record = {
                "task_id": task_id,
                "env_key": env_key,
                "data_key": data_key,
                "prompt": prompt,
                "verifier": verifier,
                "hint": hint,
                "raw_scores": raw_scores,
                "hinted_scores": hinted_scores,
                "raw_job_id": raw_job_id,
                "hinted_job_id": hinted_job_id,
                "var_raw": result["var_raw"],
                "hint_gap": result["hint_gap"],
                "total": result["total"],
                "duration": duration,
                "timestamp": time.time(),
            }
            with open(path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to save rollout: {e}")

    async def _handle_task_generation(self, action: str) -> BaseTextEnvStepOutput:
        """Evaluate a generated task through the full pipeline.

        Pipeline:
            1. Parse <task> output -> fail = reward 0
            2. Sandbox validation -> fail = reward 0
            3. LLM-as-a-judge -> gate (0/1), fail = reward 0
            4. Hint-based evaluation via Fleet harness (k raw + k hinted rollouts)
            5. R = base_quality + judge_gate * compute_task_reward(raw, hinted)

        base_quality (default 0.1) rewards structural validity (sandbox+judge pass),
        providing GRPO gradient signal even when harness evals return all zeros.
        """
        metadata: Dict[str, Any] = {"env_key": self.env_key, "turn": self.turns}

        # 1. Parse
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

        # 2. Sandbox validation
        validation = self.sandbox.validate(verifier, prompt)
        metadata["validation"] = {
            "valid": validation.valid,
            "passed": validation.checks_passed,
            "failed": validation.checks_failed,
            "error": validation.error,
        }
        if not validation.valid:
            metadata["reward_breakdown"] = {"sandbox": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 3. LLM-as-a-judge gate
        judge_gate = self._judge_task(prompt, verifier)
        metadata["judge_gate"] = judge_gate

        if judge_gate == 0.0:
            metadata["reward_breakdown"] = {"sandbox": 1.0, "judge": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 4. Hint-based evaluation via Fleet harness
        eval_result = await self._evaluate_task(prompt, verifier)

        # 5. R = base_quality + tool_call_reward + eval_signal
        # base_quality: small reward for passing sandbox+judge (structural validity)
        # tool_call_reward: incentivize DB exploration (describe_db, query_db)
        # eval_signal: judge_gate * compute_task_reward (harness-based quality)
        # This prevents GRPO zero-signal deadlock when all harness evals fail.
        base_quality = self.base_quality_reward
        tool_call_reward = self.meta_tool_calls * self.tool_call_reward_per_call
        eval_signal = judge_gate * eval_result["total"]
        reward = base_quality + tool_call_reward + eval_signal

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "base_quality": base_quality,
            "tool_call_reward": tool_call_reward,
            "meta_tool_calls": self.meta_tool_calls,
            "eval_signal": eval_signal,
            **eval_result,
            "total": reward,
        }

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Sync wrapper for step_async."""
        return asyncio.run(self.step_async(action))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step — tool call, task generation, or nudge.

        Multi-turn flow:
            1. <task> block detected  → evaluation pipeline (done=True)
            2. <tool_call> detected   → execute describe_db/query_db (done=False)
            3. Neither                → nudge observation (done=False)
            4. max_turns reached      → done=True, reward=0
        """
        self.turns += 1
        max_turns_reached = self.turns >= self.max_turns

        # 1. Check for <task> block → evaluation pipeline
        if "<task>" in action:
            return await self._handle_task_generation(action)

        # 2. Check for tool calls → execute all via Fleet orchestrator or MCP
        tool_calls = parse_tool_calls(action)
        tool_calls = [tc for tc in tool_calls if tc["name"] in self.callable_tools]
        if tool_calls:
            results = []
            for tc in tool_calls:
                if tc["name"] in _META_TOOLS:
                    self.meta_tool_calls += 1
                    if tc["name"] == "describe_db":
                        self.called_describe_db = True
                    elif tc["name"] == "query_db":
                        self.called_query_db = True
                    result = await self._execute_meta_tool(tc)
                else:
                    self.mcp_tool_calls += 1
                    result = await self._execute_mcp_tool(tc)
                results.append(f"[{tc['name']}] {result}")

            if max_turns_reached:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={"env_key": self.env_key, "turn": self.turns, "done_reason": "max_turns"},
                )

            obs_content = "\n\n".join(results)
            observation = {"role": "user", "content": obs_content}
            return BaseTextEnvStepOutput(
                observations=[observation],
                reward=0.0,
                done=False,
                metadata={"env_key": self.env_key, "turn": self.turns, "tool_calls": [tc["name"] for tc in tool_calls]},
            )

        # 3. Neither task nor tool call → nudge
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={
                    "env_key": self.env_key,
                    "turn": self.turns,
                    "done_reason": "max_turns",
                },
            )

        remaining = self.max_turns - self.turns
        if self.max_turns == 1:
            nudge = "No <task> block found. Output your task in <task>...</task> format."
        elif remaining <= 2:
            nudge = (
                f"You have {remaining} turn(s) left. Output your <task> block NOW or you will "
                "get reward 0. Stop exploring and generate the task."
            )
        else:
            nudge = "Use <tool_call> to explore the database or call environment tools, then generate a <task> block."
        observation = {"role": "user", "content": nudge}
        return BaseTextEnvStepOutput(
            observations=[observation],
            reward=0.0,
            done=False,
            metadata={"env_key": self.env_key, "turn": self.turns},
        )

    async def _execute_meta_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a describe_db or query_db meta-tool call via the Fleet orchestrator."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.orch is None:
            return "Error: Fleet environment not provisioned. Generate a <task> directly."

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if name == "describe_db":
                    result = await self.orch.describe_db_async(db_name=args.get("db_name", "seed"))
                elif name == "query_db":
                    sql = args.get("sql", "")
                    if not sql:
                        return "Error: query_db requires a 'sql' argument."
                    result = await self.orch.query_db_async(sql=sql, db_name=args.get("db_name", "seed"))
                else:
                    return f"Error: Unknown meta-tool '{name}'."

                if isinstance(result, dict):
                    return f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
                return f"Tool result:\n{result}"
            except Exception as e:
                if attempt < max_retries - 1 and ("closed" in str(e).lower() or "transport" in str(e).lower() or "connection" in str(e).lower()):
                    await asyncio.sleep(1)
                    continue
                return f"Error: {e}"

    async def _execute_mcp_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute an MCP tool call via FleetMCPTools."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.mcp_tools is None:
            return "Error: MCP tools not available. Use describe_db/query_db or generate a <task>."

        try:
            result = await self.mcp_tools.call_tool(name, args)
            if isinstance(result, dict):
                return f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
            return f"Tool result:\n{result}"
        except Exception as e:
            return f"Error calling {name}: {e}"

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the environment, optionally provisioning a Fleet env for DB exploration.

        When ``max_turns > 1``, provisions a Fleet environment via
        ``FleetEnvClient.from_fleet_async`` so the model can call
        ``describe_db`` / ``query_db`` during exploration turns.
        Falls back to single-turn if provisioning fails.
        """
        self.turns = 0
        self.meta_tool_calls = 0
        self.mcp_tool_calls = 0
        self.called_describe_db = False
        self.called_query_db = False
        self.orch = None
        self.mcp_tools = None
        self.callable_tools = set(_META_TOOLS)

        # Provision Fleet env for multi-turn exploration (DB + MCP tools)
        if self.max_turns > 1 and self.fleet_api_key and self.data_key:
            try:
                from envs.fleet_env import FleetEnvClient

                self.orch, self.mcp_tools = await FleetEnvClient.from_fleet_async(
                    api_key=self.fleet_api_key,
                    env_key=self.env_key,
                    data_key=self.data_key,
                    data_version=self.data_version,
                    image_type="standard",
                    ttl_seconds=900,
                )
                # Load instance resources so db("seed") works
                # instance.load() is async — must await directly, not via to_thread
                await self.orch._fleet_env.instance.load()
                logger.info(f"TaskGenEnv [{self.env_key}]: Fleet env provisioned for DB + tool exploration")

                # Discover MCP tools so the model can call them
                if self.mcp_tools:
                    try:
                        tools_action = await self.mcp_tools.list_tools()
                        mcp_tools = [
                            t for t in tools_action.tools if "function" in t and t["function"].get("name") != "computer"
                        ]
                        mcp_tool_names = {t["function"]["name"] for t in mcp_tools}
                        self.callable_tools = set(_META_TOOLS) | mcp_tool_names
                        # Update tool schemas for system prompt if dataset didn't have them
                        if not self.env_tools_schema:
                            self.env_tools_schema = mcp_tools
                            self.env_tools = [t["function"]["name"] for t in mcp_tools]
                        logger.info(f"TaskGenEnv [{self.env_key}]: {len(mcp_tool_names)} MCP tools available")
                    except Exception as e:
                        logger.warning(f"TaskGenEnv [{self.env_key}]: Failed to list MCP tools: {e}")
            except Exception as e:
                logger.warning(
                    f"TaskGenEnv [{self.env_key}]: Fleet provisioning failed, " f"falling back to single-turn: {e}"
                )
                self.max_turns = 1

        system_prompt = self._build_system_prompt()

        user_content = (
            f"Generate a task for the {self.env_key} environment. "
            "First explore the database to understand the data, then draft a prompt and verifier. "
            "Before outputting, query the DB to verify your assumptions are correct — "
            "iterate on your draft until you're confident the data supports it."
            if self.max_turns > 1
            else f"Generate a task for the {self.env_key} environment."
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        metadata = {
            "env_key": self.env_key,
            "env_version": self.env_version,
            "num_tools": len(self.env_tools),
            "multi_turn": self.max_turns > 1,
        }

        return conversation, metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Sync wrapper for init_async."""
        return asyncio.run(self.init_async(prompt))

    def close(self):
        """Close the Fleet orchestrator if provisioned."""
        if self.orch is not None:
            try:
                self.orch.close()
            except Exception:
                pass
            self.orch = None

    async def close_async(self):
        """Async close — release Fleet orchestrator resources."""
        if self.orch is not None:
            try:
                await self.orch.close_async()
            except Exception:
                pass
            self.orch = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return per-episode metrics."""
        return {
            "env_key": self.env_key,
            "turns": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        # Group by env_key
        env_counts: Dict[str, int] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        result = {"total_episodes": len(metrics)}
        for env_key, count in env_counts.items():
            result[f"{env_key}/episodes"] = count

        return result
