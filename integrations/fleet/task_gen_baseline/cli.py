"""CLI for task-generation baselines and solver rollouts.

This module intentionally stays outside the RL trainer. It reuses TaskGenEnv's
prompt construction, exploration tools, and Fleet harness solver path so a
frontier model can be compared against the trained Qwen generator.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import random
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
SKYRL_GYM_SRC = REPO_ROOT / "skyrl-gym"
if SKYRL_GYM_SRC.exists() and str(SKYRL_GYM_SRC) not in sys.path:
    sys.path.insert(0, str(SKYRL_GYM_SRC))


DEFAULT_EVALUATOR_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_JUDGE_MODEL = DEFAULT_EVALUATOR_MODEL
DEFAULT_RANDOM_ENV_KEY = "booking"
SEED_STATE_DRYRUN_ERROR = (
    "Verifier returned 1 on the unmodified database — it passes even when no agent has acted. "
    "Your verifier must return 0 on seed state. Check that your task involves a write/mutation action "
    "and your verifier checks for that mutation (e.g., find_new_entries)."
)
UNSET = object()


os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")


def log_generate(message: str) -> None:
    print(f"[task-gen generate] {message}", file=sys.stderr, flush=True)


def patch_fleet_env_client_close_async() -> None:
    try:
        from envs.fleet_env.client import FleetEnvClient
    except ImportError:
        return

    async def close_async(self):
        try:
            close_result = vars(self)[chr(95) + "fleet_env"].close()
        except KeyError:
            close_result = None
        if inspect.isawaitable(close_result):
            await close_result
        super(FleetEnvClient, self).close()

    FleetEnvClient.close_async = close_async


def make_task_gen_env(env_config: Dict[str, Any], extras: Dict[str, Any]):
    patch_fleet_env_client_close_async()

    from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv

    return TaskGenEnv(env_config=env_config, extras=extras)


def load_json_arg(value: Optional[str], path: Optional[str], default: Any) -> Any:
    if value and path:
        raise ValueError("Pass either an inline JSON value or a JSON file, not both.")
    if path:
        with open(path) as f:
            return json.load(f)
    if value:
        return json.loads(value)
    return default


def load_task_key_strings(path: str) -> List[str]:
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Fleet task key file must be a JSON array of task key strings.")

    task_keys = [task_key.strip() for task_key in data if isinstance(task_key, str) and task_key.strip()]
    if len(task_keys) != len(data):
        raise ValueError("Fleet task key file must contain only non-empty string values.")
    if not task_keys:
        raise ValueError("Fleet task key file is empty.")
    return task_keys


def has_manual_snapshot(args: argparse.Namespace) -> bool:
    return bool(args.env_key and args.data_key and args.data_version)


def has_any_manual_snapshot_arg(args: argparse.Namespace) -> bool:
    return bool(args.env_key or args.data_key or args.data_version)


def is_live_task_list_selector(args: argparse.Namespace) -> bool:
    return bool(
        args.command == "generate"
        and args.allow_live_task_list
        and args.env_key
        and not args.data_key
        and not args.data_version
        and not args.fleet_task_key
        and not args.fleet_task_key_file
    )


def has_partial_manual_snapshot(args: argparse.Namespace) -> bool:
    return has_any_manual_snapshot_arg(args) and not has_manual_snapshot(args)


def validate_context_arguments(args: argparse.Namespace) -> None:
    if args.fleet_task_key and args.fleet_task_key_file:
        raise ValueError("Pass either --fleet-task-key or --fleet-task-key-file, not both.")
    if (args.fleet_task_key or args.fleet_task_key_file) and has_any_manual_snapshot_arg(args):
        raise ValueError("Pass either a Fleet task key source or manual snapshot args, not both.")
    if has_partial_manual_snapshot(args) and not is_live_task_list_selector(args):
        raise ValueError("Pass --env-key, --data-key, and --data-version together, or omit all three.")
    if args.fleet_base_url:
        raise ValueError(
            "--fleet-base-url is not supported for baseline generation/solving because TaskGenEnv "
            "creates its own Fleet client. Use the default Fleet backend for controlled runs."
        )
    if (
        args.command == "generate"
        and not args.fleet_task_key
        and not has_manual_snapshot(args)
        and not args.fleet_task_key_file
        and not args.allow_live_task_list
    ):
        raise ValueError(
            "Pass --fleet-task-key, --fleet-task-key-file, or manual --env-key/--data-key/--data-version. "
            "Use --allow-live-task-list only for ad hoc, non-controlled sampling from Fleet."
        )
    if (
        args.command == "solve"
        and not args.fleet_task_key
        and not has_manual_snapshot(args)
        and not args.fleet_task_key_file
    ):
        raise ValueError(
            "Solve requires environment context. Use a generated JSON file with env_key/data_key/data_version, "
            "or pass --fleet-task-key, --fleet-task-key-file, or manual --env-key/--data-key/--data-version."
        )
    if args.command == "solve" and args.judge_model and not (
        args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    ):
        raise RuntimeError("OPENROUTER_API_KEY is required when --judge-model is set for solve.")


def apply_runtime_environment(args: argparse.Namespace) -> None:
    if args.fleet_api_key:
        os.environ["FLEET_API_KEY"] = args.fleet_api_key
    if args.command == "solve" and args.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_api_key


def make_fleet_client(args: argparse.Namespace, purpose: str):
    try:
        from fleet import Fleet
    except ImportError as exc:
        raise RuntimeError(f"fleet-python is required for {purpose}.") from exc

    api_key = args.fleet_api_key or os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        raise RuntimeError("FLEET_API_KEY is required. Pass --fleet-api-key or set the env var.")

    return Fleet(
        api_key=api_key,
        base_url=args.fleet_base_url or None,
        timeout=args.fleet_timeout,
    )


def fetch_fleet_task_key_candidates(args: argparse.Namespace, env_key: str) -> List[str]:
    start = time.time()
    log_generate(f"Listing Fleet task keys for env_key={env_key!r}")
    fleet = make_fleet_client(args, "listing Fleet task keys")
    response = fleet.client.request(
        "GET",
        "/v1/tasks",
        params={"env_key": env_key},
        timeout=args.fleet_timeout,
    )
    response_data = response.json()
    tasks = response_data.get("tasks", []) if isinstance(response_data, dict) else []
    task_keys = sorted(
        {
            task.get("key") or ""
            for task in tasks
            if isinstance(task, dict) and isinstance(task.get("key"), str) and task.get("key")
        }
    )
    if not task_keys:
        raise ValueError(f"No Fleet task keys found for env_key={env_key!r}.")
    log_generate(f"Found {len(task_keys)} Fleet task key(s) for env_key={env_key!r} in {time.time() - start:.1f}s")
    return task_keys


def select_random_fleet_task_key(args: argparse.Namespace) -> str:
    if args.fleet_task_key_file:
        candidates = load_task_key_strings(args.fleet_task_key_file)
        source = args.fleet_task_key_file
        log_generate(f"Loaded {len(candidates)} Fleet task key(s) from {source}")
    else:
        env_key = args.env_key or DEFAULT_RANDOM_ENV_KEY
        candidates = fetch_fleet_task_key_candidates(args, env_key)
        source = f"Fleet env_key={env_key!r}"

    rng = random.Random(args.random_seed) if args.random_seed is not None else random.SystemRandom()
    task_key = rng.choice(candidates)
    log_generate(f"Using random Fleet task key from {source}: {task_key}")
    args.fleet_task_key_source = source
    args.fleet_task_key_candidate_count = len(candidates)
    return task_key


def print_selected_fleet_task_key(task_key: str, source: str) -> None:
    if task_key:
        log_generate(f"Using Fleet task key from {source}: {task_key}")


def load_generated_file_data(file_text: str) -> Dict[str, Any]:
    try:
        data = json.loads(file_text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def apply_fleet_context_from_generated_file(args: argparse.Namespace, generated_data: Dict[str, Any]) -> None:
    if args.command != "solve":
        return
    if args.fleet_task_key or args.fleet_task_key_file or has_any_manual_snapshot_arg(args):
        return

    fleet_task_key = generated_data.get("fleet_task_key")
    if isinstance(fleet_task_key, str) and fleet_task_key:
        args.fleet_task_key = fleet_task_key
        args.fleet_task_key_source = "generated file"
        return

    env_key = generated_data.get("env_key")
    data_key = generated_data.get("data_key")
    data_version = generated_data.get("data_version")
    if isinstance(env_key, str) and isinstance(data_key, str) and isinstance(data_version, str):
        if env_key and data_key and data_version:
            args.env_key = env_key
            args.data_key = data_key
            args.data_version = data_version
            args.env_version = generated_data.get("env_version") or args.env_version


def load_fleet_context(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if args.fleet_task_key:
        print_selected_fleet_task_key(args.fleet_task_key, args.fleet_task_key_source)
        return load_fleet_task_context(args)
    if args.command == "generate" and not has_manual_snapshot(args):
        args.fleet_task_key = select_random_fleet_task_key(args)
        return load_fleet_task_context(args)
    if args.command == "solve" and args.fleet_task_key_file:
        args.fleet_task_key = select_random_fleet_task_key(args)
        return load_fleet_task_context(args)
    return None


def load_fleet_task_context(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    log_generate(f"Fetching Fleet context for task_key={args.fleet_task_key}")
    fleet = make_fleet_client(args, "--fleet-task-key")
    response = fleet.client.request(
        "GET",
        "/v1/tasks",
        params={"task_keys": [args.fleet_task_key]},
        timeout=args.fleet_timeout,
    )
    tasks = response.json().get("tasks", [])
    exact_matches = [task for task in tasks if task.get("key") == args.fleet_task_key]
    if not exact_matches:
        raise ValueError(f"No Fleet task found for key {args.fleet_task_key!r}.")

    task = exact_matches[0]
    task_context = {
        "task_key": task.get("key") or "",
        "prompt": task.get("prompt") or "",
        "env_key": task.get("environment_id") or task.get("env_id") or task.get("env_key") or "",
        "env_version": task.get("version") or "",
        "data_key": task.get("data_id") or task.get("data_key") or "",
        "data_version": task.get("data_version") or "",
        "verifier_code": task.get("verifier_func") or "",
        "env_variables": task.get("env_variables") or {},
    }
    log_generate(
        "Fetched Fleet context "
        f"env={task_context['env_key']} data_key={task_context['data_key']} "
        f"data_version={task_context['data_version']} in {time.time() - start:.1f}s"
    )
    return task_context


def fleet_context_extras(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if context is None:
        return {}
    return {
        "env_key": context["env_key"],
        "data_source": context["env_key"],
        "env_version": context["env_version"],
        "data_key": context["data_key"],
        "data_version": context["data_version"],
        "env_variables": context["env_variables"],
        "env_variable_keys": [],
        "env_tools_schema": [],
        "env_tools": [],
        "env_schema": "",
        "task_key": context["task_key"],
    }


def read_optional_text(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text()


def openrouter_model(model: str) -> str:
    if model.startswith("openrouter/"):
        return model
    return f"openrouter/{model}"


def make_env_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "max_turns": args.max_turns,
        "judge_model": args.judge_model,
        "k_rollouts": args.k_rollouts,
        "max_eval_steps": args.max_eval_steps,
        "evaluator_model": args.evaluator_model,
        "base_quality_reward": args.base_quality_reward,
        "eval_k_rollouts": args.eval_k_rollouts,
        "enable_hints": args.enable_hints,
        "tool_call_reward_per_call": args.tool_call_reward_per_call,
        "verifier_min_ast_nodes": args.verifier_min_ast_nodes,
        "verifier_max_ast_nodes": args.verifier_max_ast_nodes,
    }


def make_env_extras(args: argparse.Namespace) -> Dict[str, Any]:
    fleet_context = load_fleet_context(args)
    fleet_extras = fleet_context_extras(fleet_context)

    env_variables = load_json_arg(args.env_variables_json, args.env_variables_file, UNSET)
    env_variable_keys = load_json_arg(args.env_variable_keys_json, args.env_variable_keys_file, UNSET)
    env_tools_schema = load_json_arg(args.env_tools_schema_json, args.env_tools_schema_file, UNSET)
    env_tools = load_json_arg(args.env_tools_json, args.env_tools_file, UNSET)
    env_schema_from_file = read_optional_text(args.env_schema_file)

    env_key = args.env_key or fleet_extras.get("env_key") or ""
    if not env_key:
        raise ValueError("Pass --env-key/--data-key/--data-version or pass a Fleet task key.")

    extras = {
        "env_key": env_key,
        "data_source": env_key,
        "env_version": args.env_version or fleet_extras.get("env_version", ""),
        "data_key": args.data_key or fleet_extras.get("data_key", ""),
        "data_version": args.data_version or fleet_extras.get("data_version", ""),
        "env_variables": fleet_extras.get("env_variables", {}),
        "env_variable_keys": fleet_extras.get("env_variable_keys", []),
        "env_tools_schema": fleet_extras.get("env_tools_schema", []),
        "env_tools": fleet_extras.get("env_tools", []),
        "env_schema": env_schema_from_file or fleet_extras.get("env_schema", ""),
    }
    if fleet_extras.get("task_key"):
        extras["fleet_task_key"] = fleet_extras["task_key"]
    if env_variables is not UNSET:
        extras["env_variables"] = env_variables
    if env_variable_keys is not UNSET:
        extras["env_variable_keys"] = env_variable_keys
    if env_tools_schema is not UNSET:
        extras["env_tools_schema"] = env_tools_schema
    if env_tools is not UNSET:
        extras["env_tools"] = env_tools

    if args.training_phase == "eval":
        extras["training_phase"] = "eval"

    return extras


def write_json(data: Dict[str, Any], output: Optional[str]) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    if output:
        Path(output).write_text(rendered + "\n")
    else:
        print(rendered)


def extract_task_from_text(text: str) -> Tuple[str, str, str]:
    """Extract prompt and verifier from raw XML-ish text or a JSON output file."""
    raw_text = text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict):
        if isinstance(data.get("prompt"), str) and isinstance(data.get("verifier"), str):
            task_xml = format_task_xml(data["prompt"], data["verifier"])
            return data["prompt"], data["verifier"], task_xml
        for key in ("task_xml", "generated_text", "final_text", "output"):
            if isinstance(data.get(key), str):
                raw_text = data[key]
                break

    parsed = parse_task_output(raw_text)
    if parsed is None:
        raise ValueError("Could not find <prompt>...</prompt> and <verifier>...</verifier> in input file.")
    return parsed["prompt"], parsed["verifier"], raw_text


def parse_task_output(action: str) -> Optional[dict]:
    prompt_match = re.search(r"<prompt>(.*?)</prompt>", action, re.DOTALL)
    verifier_match = re.search(r"<verifier>(.*?)</verifier>", action, re.DOTALL)
    if not prompt_match or not verifier_match:
        return None
    return {
        "prompt": prompt_match.group(1).strip(),
        "verifier": verifier_match.group(1).strip(),
    }


def format_task_xml(prompt: str, verifier: str) -> str:
    return f"<task>\n<prompt>\n{prompt}\n</prompt>\n<verifier>\n{verifier}\n</verifier>\n</task>"


def format_dryrun_feedback(dryrun_error: str, turns_remaining: int) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"⚠️ Verifier dry-run FAILED: {dryrun_error}\n\n"
            f"Fix your verifier and resubmit. {turns_remaining} turn(s) left."
        ),
    }


def format_parse_feedback(turns_remaining: int) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            "No complete <task> block with both <prompt> and <verifier> was found. "
            f"Fix the format and resubmit. {turns_remaining} turn(s) left."
        ),
    }


def format_sandbox_feedback(validation: Dict[str, Any], turns_remaining: int) -> Dict[str, str]:
    failed = ", ".join(validation["failed"]) if validation["failed"] else "unknown validation check"
    return {
        "role": "user",
        "content": f"Sandbox rejected your verifier: {failed}. Fix and resubmit. {turns_remaining} turn(s) left.",
    }


def format_exploration_feedback(turns_remaining: int) -> Dict[str, str]:
    return {
        "role": "user",
        "content": (
            "You must explore the database with `query_db` before submitting a task. "
            "Use SELECT queries to inspect actual data — table contents, value ranges, "
            f"row counts — so your task and verifier are grounded in real data. "
            f"You have {turns_remaining} turn(s) remaining."
        ),
    }


def validation_to_dict(validation: Any) -> Dict[str, Any]:
    return {
        "valid": validation.valid,
        "passed": validation.checks_passed,
        "failed": validation.checks_failed,
        "error": validation.error,
    }


async def run_verifier_dryrun_async(env: Any, verifier: str) -> Tuple[bool, str]:
    if env.orch is None:
        return False, "Fleet environment was not provisioned, so verifier dry-run could not run."

    try:
        return await env.dryrun_verifier(verifier)
    except Exception as exc:
        error = str(exc)
        if len(error) > 500:
            error = error[:500] + "..."
        return False, f"Verifier crashed on seed DB: {error}"


def should_retry_generation(result: Dict[str, Any]) -> bool:
    if result.get("parse_error"):
        return True
    validation = result.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is False:
        return True
    if result.get("dryrun_error"):
        return True
    return result.get("done_reason") in {"max_turns", "missing_query_db", "task_not_generated"}


def generation_failure_message(result: Dict[str, Any]) -> str:
    if result.get("parse_error"):
        return result["parse_error"]
    validation = result.get("validation")
    if isinstance(validation, dict) and validation.get("valid") is False:
        return validation.get("error") or f"Sandbox failed: {validation.get('failed', [])}"
    if result.get("dryrun_error"):
        return result["dryrun_error"]
    return f"Generation ended with done_reason={result.get('done_reason', 'unknown')}"


def retry_budget_allows(args: argparse.Namespace, attempt: int) -> bool:
    return args.retry < 0 or attempt <= args.retry


def native_query_db_tool_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Run a read-only SQL query against the seed database for the current Fleet task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT query to run against the seed database.",
                    },
                    "db_name": {
                        "type": "string",
                        "description": "Database name. Use seed unless explicitly instructed otherwise.",
                        "default": "seed",
                    },
                },
                "required": ["sql"],
            },
        },
    }


def build_native_tool_call_instructions() -> str:
    return """
## Exploration Tools

You are solving the outer task: write a prompt and verifier for an inner solver agent.
The Available Tools section above describes tools the inner solver agent can use while solving your generated task.
You cannot call those inner solver tools during task generation.

Your only outer-task tool is the native `query_db` tool. Use it to run read-only SQL against the seed database. Inspect real rows, value ranges, IDs, dates, and row counts before writing the task.

### Workflow
1. **Inspect data**: Call `query_db` to inspect real data.
2. **Draft a task idea**: Base it on observed data and the inner solver tools described above.
3. **Validate**: Make sure the referenced data exists, the task is achievable with the inner solver tools, and the verifier will return 0 on the unmodified database.
4. **Output**: Only when confident, output the final task in the textual `<task>` format below."""


def build_native_generation_messages(env: Any, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from skyrl_gym.envs.task_gen.task_gen_env import (
        build_task_gen_db_schema_prompt,
        build_task_gen_output_format_instructions,
        build_task_gen_task_verifier_instructions,
    )

    system_prompt = "\n".join(
        [
            build_task_gen_db_schema_prompt(
                env_key=env.env_key,
                env_tools_schema=env.env_tools_schema,
                env_tools=env.env_tools,
                env_variables=env.env_variables,
                env_variable_keys=env.env_variable_keys,
                env_schema=env.env_schema,
            ),
            build_task_gen_task_verifier_instructions(env.env_variables.get("CURRENT_DATE", "")),
            build_native_tool_call_instructions(),
            build_task_gen_output_format_instructions(),
        ]
    )
    return [{"role": "system", "content": system_prompt}, conversation[1]]


def native_tools_for_env(env: Any) -> List[Dict[str, Any]]:
    return [native_query_db_tool_schema()]


def jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return jsonable_value(value.model_dump())
    except AttributeError:
        pass
    try:
        return jsonable_value(dict(value))
    except (TypeError, ValueError):
        pass
    try:
        return jsonable_value(vars(value))
    except TypeError:
        return str(value)


def parse_native_tool_arguments(raw_arguments: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(raw_arguments, dict):
        return raw_arguments, ""
    if raw_arguments in (None, ""):
        return {}, ""
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, f"Could not parse tool arguments as JSON: {exc}"
        if isinstance(parsed, dict):
            return parsed, ""
        return {}, "Tool arguments must decode to a JSON object."
    return {}, f"Unsupported tool arguments type: {type(raw_arguments).__name__}"


def normalize_native_tool_call(tool_call: Any, index: int) -> Dict[str, Any]:
    tool_call_data = jsonable_value(tool_call)
    if not isinstance(tool_call_data, dict):
        tool_call_data = {}
    function_data = tool_call_data.get("function", {})
    if not isinstance(function_data, dict):
        function_data = jsonable_value(function_data)
    if not isinstance(function_data, dict):
        function_data = {}
    raw_arguments = function_data.get("arguments", "{}")
    arguments, argument_error = parse_native_tool_arguments(raw_arguments)
    argument_text = raw_arguments if isinstance(raw_arguments, str) else json.dumps(arguments)
    call_id = str(tool_call_data.get("id") or f"call_{index}")
    name = str(function_data.get("name", ""))
    return {
        "id": call_id,
        "name": name,
        "arguments": arguments,
        "argument_error": argument_error,
        "assistant_tool_call": {
            "id": call_id,
            "type": tool_call_data.get("type") or "function",
            "function": {
                "name": name,
                "arguments": argument_text,
            },
        },
    }


def format_xml_tool_call_from_native(native_tool_call: Dict[str, Any]) -> str:
    payload = {"name": native_tool_call["name"], "arguments": native_tool_call["arguments"]}
    return f"<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>"


def observation_text(observations: List[Dict[str, Any]]) -> str:
    contents = [message.get("content", "") for message in observations]
    return "\n\n".join(content for content in contents if content)


async def execute_native_tool_calls_async(
    env: Any, native_tool_calls: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tool_messages = []
    last_step_output = {"observations": [], "reward": 0.0, "done": False, "metadata": {"tool_calls": []}}

    for native_tool_call in native_tool_calls:
        if native_tool_call["argument_error"]:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": native_tool_call["id"],
                    "name": native_tool_call["name"],
                    "content": f"Error: {native_tool_call['argument_error']}",
                }
            )
        elif native_tool_call["name"] not in env.callable_tools:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": native_tool_call["id"],
                    "name": native_tool_call["name"],
                    "content": f"Error: Unknown tool {native_tool_call['name']!r}.",
                }
            )
        else:
            xml_action = format_xml_tool_call_from_native(native_tool_call)
            last_step_output = await env.step_async(xml_action)
            content = observation_text(last_step_output["observations"]) or "Tool call completed with no output."
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": native_tool_call["id"],
                    "name": native_tool_call["name"],
                    "content": content,
                }
            )
            if last_step_output["done"]:
                break

    if last_step_output["done"] and len(tool_messages) < len(native_tool_calls):
        for native_tool_call in native_tool_calls[len(tool_messages) :]:
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": native_tool_call["id"],
                    "name": native_tool_call["name"],
                    "content": "Error: Environment ended before this tool call could be executed.",
                }
            )

    return last_step_output, tool_messages


def validate_generated_task(env: Any, prompt: str, verifier: str) -> Dict[str, Any]:
    validation = env.sandbox.validate(verifier, prompt)
    return validation_to_dict(validation)


async def validate_generated_task_on_fresh_env_async(
    args: argparse.Namespace, env_extras: Dict[str, Any], prompt: str, verifier: str
) -> Tuple[Dict[str, Any], bool, str]:
    check_env = make_task_gen_env(env_config=make_env_config(args), extras=env_extras)
    try:
        await check_env.init_async([])
        require_generation_db_ready(args, check_env, env_extras)
        validation = validate_generated_task(check_env, prompt, verifier)
        if not validation["valid"]:
            return validation, False, ""
        dryrun_ok, dryrun_error = await run_verifier_dryrun_async(check_env, verifier)
        return validation, dryrun_ok, dryrun_error
    finally:
        await check_env.close_async()


def require_generation_db_ready(args: argparse.Namespace, env: Any, env_extras: Dict[str, Any]) -> None:
    if args.allow_missing_db:
        return
    if args.max_turns <= 1:
        return
    if not env_extras.get("data_key"):
        return
    if env.orch is None:
        raise RuntimeError(
            "TaskGenEnv did not provision a Fleet database for query_db. "
            "Use a Fleet task key or complete snapshot args, or pass --allow-missing-db for prompt-only debugging."
        )


def require_solve_db_ready(env: Any, env_extras: Dict[str, Any]) -> None:
    if not env_extras.get("data_key") or not env_extras.get("data_version"):
        raise RuntimeError("Solve requires data_key and data_version so the seed-state verifier dry-run is meaningful.")
    if env.orch is None:
        raise RuntimeError(
            "TaskGenEnv did not provision a Fleet database for solve. "
            "Refusing to skip the seed-state verifier dry-run."
        )


def config_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload = {
        "max_turns": args.max_turns,
        "temperature": args.temperature if args.command == "generate" else None,
        "top_p": args.top_p if args.command == "generate" else None,
        "max_tokens": args.max_tokens if args.command == "generate" else None,
        "random_seed": args.random_seed,
        "k_rollouts": args.k_rollouts,
        "eval_k_rollouts": args.eval_k_rollouts,
        "max_eval_steps": args.max_eval_steps,
        "training_phase": args.training_phase,
        "judge_model": args.judge_model,
        "verifier_min_ast_nodes": args.verifier_min_ast_nodes,
        "verifier_max_ast_nodes": args.verifier_max_ast_nodes,
    }
    return {key: value for key, value in payload.items() if value is not None}


def solver_metrics_from_scores(scores: List[float]) -> Dict[str, Any]:
    pass_count = sum(1 for score in scores if score >= 1.0)
    total = len(scores)
    return {
        "solver_scores": scores,
        "solver_rollouts": total,
        "solver_pass_count": pass_count,
        "solver_pass_rate": pass_count / total if total else 0.0,
        "solver_pass_at_k": pass_count > 0,
    }


def read_rollout_record(rollout_dir: str, run_name: str) -> Optional[Dict[str, Any]]:
    path = Path(rollout_dir) / f"{run_name}.jsonl"
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    data = json.loads(lines[-1])
    return data if isinstance(data, dict) else None


def gates_reached_solver_run(metadata: Dict[str, Any]) -> bool:
    breakdown = metadata.get("reward_breakdown", {})
    if not isinstance(breakdown, dict):
        return False
    return breakdown.get("sandbox") == 1.0 and breakdown.get("dryrun") == 1.0 and breakdown.get("judge") != 0.0


def attach_rollout_metrics(result: Dict[str, Any], rollout_record: Optional[Dict[str, Any]]) -> None:
    if rollout_record is None:
        result.update(solver_metrics_from_scores([]))
        return

    raw_scores = rollout_record.get("raw_scores", [])
    scores = [float(score) for score in raw_scores if isinstance(score, (int, float, bool))]
    result.update(solver_metrics_from_scores(scores))
    result["raw_job_id"] = rollout_record.get("raw_job_id")
    result["hinted_job_id"] = rollout_record.get("hinted_job_id")
    result["hinted_scores"] = rollout_record.get("hinted_scores", [])
    result["rollout_log"] = rollout_record

    if rollout_record.get("raw_job_id") is None:
        raise RuntimeError("Fleet harness import failed before a raw solver job was created.")

    expected = result["eval_k_rollouts"] if result["training_phase"] == "eval" else result["k_rollouts"]
    if result["solver_rollouts"] != expected:
        raise RuntimeError(
            f"Expected {expected} solver rollout score(s), got {result['solver_rollouts']} from Fleet."
        )


def call_openrouter(
    args: argparse.Namespace,
    messages: List[Dict[str, Any]],
    attempt: int,
    turn: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    try:
        import litellm
    except ImportError as exc:
        raise RuntimeError(
            "litellm is required for generation. Install it or run in the task-gen environment."
        ) from exc

    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the env var.")

    start = time.time()
    prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
    tool_count = len(tools or [])
    log_generate(
        f"Calling OpenRouter attempt={attempt} turn={turn} model={args.model} "
        f"messages={len(messages)} prompt_chars={prompt_chars} tools={tool_count} max_tokens={args.max_tokens}"
    )
    completion_kwargs = {
        "model": openrouter_model(args.model),
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "api_key": api_key,
    }
    if tools:
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = "auto"

    response = litellm.completion(**completion_kwargs)
    message = response.choices[0].message
    content = message.content or ""
    try:
        raw_tool_calls = message.tool_calls or []
    except AttributeError:
        raw_tool_calls = []
    native_tool_calls = [
        normalize_native_tool_call(tool_call, index=index) for index, tool_call in enumerate(raw_tool_calls)
    ]
    if not content and not native_tool_calls:
        raise RuntimeError("OpenRouter returned an empty generation.")
    assistant_message = {"role": "assistant", "content": content}
    if native_tool_calls:
        assistant_message["tool_calls"] = [tool_call["assistant_tool_call"] for tool_call in native_tool_calls]
    log_generate(
        f"OpenRouter returned attempt={attempt} turn={turn} chars={len(content)} "
        f"contains_task={'<task>' in content} contains_tool_call={'<tool_call>' in content} "
        f"native_tool_calls={len(native_tool_calls)} duration={time.time() - start:.1f}s"
    )
    return {
        "content": content,
        "assistant_message": assistant_message,
        "native_tool_calls": native_tool_calls,
    }


async def generate_attempt_async(args: argparse.Namespace, attempt: int) -> Dict[str, Any]:
    attempt_start = time.time()
    log_generate(
        f"Starting generation attempt={attempt} model={args.model} "
        f"max_turns={args.max_turns} retry_budget={args.retry}"
    )
    env_extras = make_env_extras(args)
    log_generate(
        "Resolved generation context "
        f"env={env_extras['env_key']} data_key={env_extras.get('data_key', '')} "
        f"data_version={env_extras.get('data_version', '')} "
        f"fleet_task_key={env_extras.get('fleet_task_key', '')}"
    )
    env = make_task_gen_env(env_config=make_env_config(args), extras=env_extras)
    log_generate("Initializing TaskGenEnv")
    try:
        conversation, metadata = await env.init_async([])
        require_generation_db_ready(args, env, env_extras)
    except Exception:
        await env.close_async()
        log_generate("Closed TaskGenEnv after initialization failure")
        raise
    log_generate(
        f"TaskGenEnv initialized messages={len(conversation)} "
        f"metadata_keys={sorted(metadata.keys()) if isinstance(metadata, dict) else []}"
    )
    native_tools: List[Dict[str, Any]] = []
    if args.tool_mode == "native":
        conversation = build_native_generation_messages(env, conversation)
        native_tools = native_tools_for_env(env)
        env.callable_tools = {"query_db"}
        log_generate(f"Using native tool mode with {len(native_tools)} tool(s)")

    transcript: List[Dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in conversation]
    final_text = ""
    done_reason = "max_turns"
    dryrun_error = ""
    validation_result: Optional[Dict[str, Any]] = None

    try:
        turns_remaining = args.max_turns
        while turns_remaining > 0:
            turn_number = args.max_turns - turns_remaining + 1
            turns_remaining -= 1
            log_generate(f"Starting turn={turn_number} turns_remaining_after_this={turns_remaining}")
            model_output = call_openrouter(
                args,
                conversation,
                attempt=attempt,
                turn=turn_number,
                tools=native_tools if args.tool_mode == "native" else None,
            )
            final_text = model_output["content"]

            if args.tool_mode == "native" and model_output["native_tool_calls"]:
                conversation.append(model_output["assistant_message"])
                transcript.append(model_output["assistant_message"])
                tool_names = [tool_call["name"] for tool_call in model_output["native_tool_calls"]]
                log_generate(
                    f"Executing native tool call(s) through TaskGenEnv tool path on turn={turn_number}: {tool_names}"
                )
                step_output, tool_messages = await execute_native_tool_calls_async(
                    env, model_output["native_tool_calls"]
                )
                conversation.extend(tool_messages)
                transcript.extend(tool_messages)
                log_generate(
                    f"Native tool step returned done={step_output['done']} tool_messages={len(tool_messages)} "
                    f"called_query_db={env.called_query_db}"
                )
                if step_output["done"]:
                    done_reason = step_output.get("metadata", {}).get("done_reason", "done")
                    log_generate(f"Environment ended generation with done_reason={done_reason}")
                    break
                continue

            conversation.append(model_output["assistant_message"])
            transcript.append(model_output["assistant_message"])

            if "<task>" in final_text:
                log_generate(f"Detected <task> block on attempt={attempt} turn={turn_number}")
                if args.enforce_exploration_gate and env.max_turns > 1 and not env.called_query_db:
                    log_generate("Task arrived before query_db; adding exploration feedback without running solver rollouts")
                    if turns_remaining > 0:
                        observations = [format_exploration_feedback(turns_remaining)]
                        conversation.extend(observations)
                        transcript.extend(observations)
                        continue
                    done_reason = "missing_query_db"
                    continue
                parsed_task = parse_task_output(final_text)
                if parsed_task is None:
                    log_generate("Task block was incomplete or malformed")
                    if turns_remaining > 0:
                        observations = [format_parse_feedback(turns_remaining)]
                        conversation.extend(observations)
                        transcript.extend(observations)
                        continue
                    done_reason = "parse_failed"
                    break

                log_generate(
                    "Running verifier sandbox and dry-run on a fresh seed environment "
                    f"prompt_chars={len(parsed_task['prompt'])} verifier_chars={len(parsed_task['verifier'])}"
                )
                validation_result, dryrun_ok, dryrun_error = await validate_generated_task_on_fresh_env_async(
                    args, env_extras, parsed_task["prompt"], parsed_task["verifier"]
                )
                if not validation_result["valid"]:
                    log_generate(f"Verifier sandbox failed: {validation_result}")
                    if turns_remaining > 0:
                        observations = [format_sandbox_feedback(validation_result, turns_remaining)]
                        conversation.extend(observations)
                        transcript.extend(observations)
                        continue
                    done_reason = "validation_failed"
                    break

                if not dryrun_ok:
                    log_generate(f"Verifier dry-run failed: {dryrun_error}")
                    if turns_remaining > 0:
                        observations = [format_dryrun_feedback(dryrun_error, turns_remaining)]
                        conversation.extend(observations)
                        transcript.extend(observations)
                        log_generate(f"Added dry-run feedback to conversation; turns_remaining={turns_remaining}")
                        continue
                    done_reason = "dryrun_failed"
                    break
                dryrun_error = ""
                log_generate("Verifier sandbox and dry-run passed")
                done_reason = "task_generated"
                break

            log_generate(f"No <task> block on turn={turn_number}; executing environment step")
            step_output = await env.step_async(final_text)
            observations = step_output["observations"]
            conversation.extend(observations)
            transcript.extend(observations)
            log_generate(
                f"Environment step returned done={step_output['done']} observations={len(observations)} "
                f"called_query_db={env.called_query_db}"
            )
            if step_output["done"]:
                done_reason = step_output.get("metadata", {}).get("done_reason", "done")
                log_generate(f"Environment ended generation with done_reason={done_reason}")
                break
    finally:
        await env.close_async()
        log_generate("Closed TaskGenEnv")

    parsed = parse_task_output(final_text)
    result: Dict[str, Any] = {
        "env_key": env_extras["env_key"],
        "data_key": env_extras.get("data_key", ""),
        "data_version": env_extras.get("data_version", ""),
        "fleet_task_key": env_extras.get("fleet_task_key", ""),
        "model": args.model,
        "tool_mode": args.tool_mode,
        "task_key_source": args.fleet_task_key_source,
        "task_key_candidate_count": args.fleet_task_key_candidate_count,
        "generation_config": config_payload(args),
        "attempt": attempt,
        "done_reason": done_reason,
        "called_query_db": env.called_query_db,
        "metadata": metadata,
        "generated_text": final_text,
    }
    if parsed is not None:
        result.update(
            {
                "prompt": parsed["prompt"],
                "verifier": parsed["verifier"],
                "task_xml": format_task_xml(parsed["prompt"], parsed["verifier"]),
            }
        )
    else:
        result["parse_error"] = "No complete <task> block found in final model response."

    if validation_result is not None:
        result["validation"] = validation_result

    if dryrun_error:
        result["dryrun_error"] = dryrun_error

    if args.include_transcript:
        result["transcript"] = transcript

    log_generate(
        f"Finished generation attempt={attempt} done_reason={done_reason} "
        f"parsed_task={parsed is not None} dryrun_error={bool(dryrun_error)} "
        f"duration={time.time() - attempt_start:.1f}s"
    )
    return result


async def print_generate_prompt_async(args: argparse.Namespace) -> None:
    log_generate("Prompt mode started; building initial generator messages")
    env_extras = make_env_extras(args)
    log_generate(
        "Resolved prompt context "
        f"env={env_extras['env_key']} data_key={env_extras.get('data_key', '')} "
        f"data_version={env_extras.get('data_version', '')} "
        f"fleet_task_key={env_extras.get('fleet_task_key', '')}"
    )
    env = make_task_gen_env(env_config=make_env_config(args), extras=env_extras)
    try:
        log_generate("Initializing TaskGenEnv for prompt rendering")
        conversation, metadata = await env.init_async([])
        native_tools: List[Dict[str, Any]] = []
        if args.tool_mode == "native":
            conversation = build_native_generation_messages(env, conversation)
            native_tools = native_tools_for_env(env)
            log_generate(f"Rendered native tool schema with {len(native_tools)} tool(s)")
        log_generate(f"Rendered {len(conversation)} initial generator message(s)")
    finally:
        await env.close_async()
        log_generate("Closed TaskGenEnv")

    payload = {
        "model": args.model,
        "env_key": env_extras["env_key"],
        "data_key": env_extras.get("data_key", ""),
        "data_version": env_extras.get("data_version", ""),
        "fleet_task_key": env_extras.get("fleet_task_key", ""),
        "tool_mode": args.tool_mode,
        "messages": conversation,
        "tools": native_tools if args.tool_mode == "native" else [],
        "metadata": metadata,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    log_generate("Prompt mode completed without calling OpenRouter")


async def generate_async(args: argparse.Namespace) -> None:
    if args.prompt:
        await print_generate_prompt_async(args)
        return

    log_generate(
        f"Generate command started model={args.model} output={args.output or 'stdout'} "
        f"retry={args.retry} max_turns={args.max_turns}"
    )
    attempt = 1
    while True:
        result = await generate_attempt_async(args, attempt)
        if not should_retry_generation(result):
            log_generate(f"Writing generation result to {args.output or 'stdout'}")
            write_json(result, args.output)
            log_generate("Generate command completed")
            return
        if not retry_budget_allows(args, attempt):
            raise RuntimeError(f"Generation failed after {attempt} attempt(s): {generation_failure_message(result)}")

        log_generate(f"Retrying generation after attempt={attempt}: {generation_failure_message(result)}")
        attempt += 1


def handle_generate(args: argparse.Namespace) -> None:
    validate_context_arguments(args)
    apply_runtime_environment(args)
    asyncio.run(generate_async(args))


async def solve_async(args: argparse.Namespace) -> None:
    file_text = Path(args.file).read_text()
    generated_data = load_generated_file_data(file_text)
    apply_fleet_context_from_generated_file(args, generated_data)
    validate_context_arguments(args)
    prompt, verifier, task_xml = extract_task_from_text(file_text)

    env_extras = make_env_extras(args)
    run_name = f"task_gen_baseline_solve_{uuid.uuid4().hex}"
    previous_run_name = os.environ.get("RUN_NAME")
    previous_rollout_dir = os.environ.get("REWARD_ROLLOUT_DIR")
    rollout_record = None

    with tempfile.TemporaryDirectory() as rollout_dir:
        os.environ["RUN_NAME"] = run_name
        os.environ["REWARD_ROLLOUT_DIR"] = rollout_dir
        env = make_task_gen_env(env_config=make_env_config(args), extras=env_extras)
        try:
            await env.init_async([])
            require_solve_db_ready(env, env_extras)
            env.called_query_db = True
            step_output = await env.step_async(task_xml)
            rollout_record = read_rollout_record(rollout_dir, run_name)
            result = {
                "mode": "gates_and_solve",
                "primary_metric": "solver_pass_rate",
                "task_gen_reward": step_output["reward"],
                "done": step_output["done"],
                "observations": step_output["observations"],
                "metadata": step_output["metadata"],
            }
        finally:
            await env.close_async()
            if previous_run_name is None:
                os.environ.pop("RUN_NAME", None)
            else:
                os.environ["RUN_NAME"] = previous_run_name
            if previous_rollout_dir is None:
                os.environ.pop("REWARD_ROLLOUT_DIR", None)
            else:
                os.environ["REWARD_ROLLOUT_DIR"] = previous_rollout_dir

    result.update(
        {
            "env_key": env_extras["env_key"],
            "env_version": env_extras.get("env_version", ""),
            "data_key": env_extras.get("data_key", ""),
            "data_version": env_extras.get("data_version", ""),
            "fleet_task_key": env_extras.get("fleet_task_key", ""),
            "evaluator_model": args.evaluator_model,
            "k_rollouts": args.k_rollouts,
            "eval_k_rollouts": args.eval_k_rollouts,
            "rollout_count_used": args.eval_k_rollouts if args.training_phase == "eval" else args.k_rollouts,
            "max_eval_steps": args.max_eval_steps,
            "training_phase": args.training_phase,
            "judge_model": args.judge_model,
            "base_quality_reward": args.base_quality_reward,
            "enable_hints": args.enable_hints,
            "verifier_min_ast_nodes": args.verifier_min_ast_nodes,
            "verifier_max_ast_nodes": args.verifier_max_ast_nodes,
            "config": config_payload(args),
            "prompt": prompt,
            "verifier": verifier,
            "task_xml": task_xml,
        }
    )

    attach_rollout_metrics(result, rollout_record)

    if gates_reached_solver_run(result["metadata"]) and rollout_record is None:
        raise RuntimeError("Solver run produced no rollout log; refusing to write an unauditable result.")

    write_json(result, args.output)


def handle_solve(args: argparse.Namespace) -> None:
    apply_runtime_environment(args)
    asyncio.run(solve_async(args))


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fleet-task-key",
        help="Fetch environment snapshot metadata directly from Fleet by task key.",
    )
    parser.add_argument(
        "--fleet-task-key-file",
        help=(
            "Optional local JSON file containing an array of Fleet task key strings. "
            "Build it by querying Fleet for task keys from the desired environment "
            "and writing the key values, e.g. "
            '`["task_abc", "task_def"]`. Used only for random task-key selection.'
        ),
    )
    parser.add_argument(
        "--allow-live-task-list",
        action="store_true",
        help="Allow ad hoc random task-key selection from the live Fleet /v1/tasks listing.",
    )
    parser.add_argument("--fleet-api-key", default="", help="Fleet API key. Defaults to FLEET_API_KEY.")
    parser.add_argument(
        "--fleet-base-url",
        default="",
        help="Unsupported for baseline runs; TaskGenEnv must use the default Fleet backend.",
    )
    parser.add_argument(
        "--fleet-timeout",
        type=float,
        default=20.0,
        help="Fleet API request timeout in seconds for task lookup/listing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Optional RNG seed for reproducible default Fleet task-key selection.",
    )
    parser.add_argument(
        "--env-key",
        help="Fleet/OpenEnv environment key, e.g. booking. Required only when passing snapshot metadata directly.",
    )
    parser.add_argument("--data-key", default="", help="Fleet data key/seed id to use for environment provisioning.")
    parser.add_argument("--data-version", default="", help="Fleet data version for the seed.")
    parser.add_argument("--env-version", default="", help="Optional Fleet environment version.")
    parser.add_argument("--env-schema-file", help="Optional compact schema text file. Auto-populated when omitted.")
    parser.add_argument("--env-variables-json", help="Inline JSON object with environment variables.")
    parser.add_argument("--env-variables-file", help="JSON file with environment variables.")
    parser.add_argument("--env-variable-keys-json", help="Inline JSON list with environment variable keys.")
    parser.add_argument("--env-variable-keys-file", help="JSON file with environment variable keys.")
    parser.add_argument("--env-tools-json", help="Inline JSON list of tool names.")
    parser.add_argument("--env-tools-file", help="JSON file with tool names.")
    parser.add_argument("--env-tools-schema-json", help="Inline JSON list of OpenAI-style tool schemas.")
    parser.add_argument("--env-tools-schema-file", help="JSON file with OpenAI-style tool schemas.")


def add_solve_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator-model", default=DEFAULT_EVALUATOR_MODEL)
    parser.add_argument("--openrouter-api-key", default="", help="OpenRouter API key for the judge model.")
    parser.add_argument("--k-rollouts", type=int, default=4)
    parser.add_argument("--eval-k-rollouts", type=int, default=8)
    parser.add_argument("--max-eval-steps", type=int, default=20)
    parser.add_argument("--base-quality-reward", type=float, default=0.0)
    parser.add_argument("--enable-hints", action="store_true")
    parser.add_argument("--training-phase", choices=["train", "eval"], default="eval")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model for solve gates.")
    parser.add_argument(
        "--verifier-min-ast-nodes",
        type=int,
        default=None,
        help="Override the verifier sandbox minimum AST node count. Defaults to the sandbox constant.",
    )
    parser.add_argument(
        "--verifier-max-ast-nodes",
        type=int,
        default=None,
        help="Override the verifier sandbox maximum AST node count. Defaults to the sandbox constant.",
    )


def generate_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Fleet task-generation prompts and solve them without RL training."
    )
    parser.set_defaults(func=lambda unused_args: parser.print_help())
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate a task prompt/verifier with a frontier model via OpenRouter.",
    )
    add_env_arguments(generate_parser)
    generate_parser.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-5.5.")
    generate_parser.add_argument("--openrouter-api-key", default="")
    generate_parser.add_argument("--max-turns", type=int, default=10)
    generate_parser.add_argument("--max-tokens", type=int, default=4096)
    generate_parser.add_argument("--temperature", type=float, default=0.95)
    generate_parser.add_argument("--top-p", type=float, default=0.95)
    generate_parser.add_argument(
        "--prompt",
        action="store_true",
        help="Print the initial generator chat messages as JSON to stdout and exit before the model call.",
    )
    generate_parser.add_argument(
        "--tool-mode",
        choices=["xml", "native"],
        default="xml",
        help="Use RL-style XML tool tags or native OpenRouter tool calls during generation.",
    )
    generate_parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Retry generation after verifier dry-run failures. Use -1 to retry forever.",
    )
    generate_parser.add_argument("--tool-call-reward-per-call", type=float, default=0.0)
    generate_parser.add_argument(
        "--no-enforce-exploration-gate",
        dest="enforce_exploration_gate",
        action="store_false",
        help="Allow a <task> response before query_db has been called.",
    )
    generate_parser.set_defaults(enforce_exploration_gate=True)
    generate_parser.add_argument(
        "--allow-missing-db",
        action="store_true",
        help="Allow generation to continue if Fleet DB provisioning fails. Intended only for prompt debugging.",
    )
    generate_parser.add_argument("--include-transcript", action="store_true")
    generate_parser.add_argument("-o", "--output", help="Write JSON output to this file. Defaults to stdout.")
    generate_parser.set_defaults(
        base_quality_reward=0.0,
        enable_hints=False,
        eval_k_rollouts=8,
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        fleet_task_key_source="argument",
        fleet_task_key_candidate_count=None,
        judge_model="",
        k_rollouts=4,
        max_eval_steps=20,
        training_phase="train",
        verifier_max_ast_nodes=None,
        verifier_min_ast_nodes=None,
    )
    generate_parser.set_defaults(func=handle_generate)

    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a generated task file with the current Fleet harness path.",
    )
    add_env_arguments(solve_parser)
    add_solve_config_arguments(solve_parser)
    solve_parser.add_argument("--max-turns", type=int, default=10)
    solve_parser.add_argument("--file", required=True, help="Generated JSON or raw <task> file to solve.")
    solve_parser.add_argument("-o", "--output", help="Write JSON output to this file. Defaults to stdout.")
    solve_parser.set_defaults(fleet_task_key_source="argument", tool_call_reward_per_call=0.0)
    solve_parser.set_defaults(allow_missing_db=False, fleet_task_key_candidate_count=None)
    solve_parser.set_defaults(func=handle_solve)

    return parser


if __name__ == "__main__":
    parser = generate_cli()
    args = parser.parse_args()
    args.func(args)
