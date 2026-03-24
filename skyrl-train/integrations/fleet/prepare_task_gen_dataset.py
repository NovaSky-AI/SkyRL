"""
Prepare dataset for task generation training.

Supports two modes:
    --mode sft:  (env_context, task_spec) pairs for supervised fine-tuning
    --mode grpo: Prompt-only records (env contexts) for GRPO reinforcement learning

GRPO mode discovers tools from live Fleet environments via OpenEnv's
FleetEnvClient + FleetMCPTools, so the system prompt contains real tool
schemas instead of empty placeholders.

Usage:
    # GRPO with tool discovery (requires FLEET_API_KEY)
    python -m integrations.fleet.prepare_task_gen_dataset \
        --tasks-json ~/data/fleet/all_tool_use.json \
        --output-dir ./data/task_gen --mode grpo

    # GRPO without tool discovery (local testing)
    python -m integrations.fleet.prepare_task_gen_dataset \
        --tasks-json ~/data/fleet/all_tool_use.json \
        --output-dir ./data/task_gen --mode grpo --no-discover-tools

    # SFT: supervised pairs with response
    python -m integrations.fleet.prepare_task_gen_dataset \
        --tasks-json ~/data/fleet/all_tool_use.json \
        --output-dir ./data/task_gen --mode sft
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_tasks(json_path: str) -> List[Dict[str, Any]]:
    """Load tasks from Fleet export JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    else:
        raise ValueError("Invalid JSON format")


def load_env_context(env_context_dir: str, env_key: str) -> Optional[Dict[str, Any]]:
    """Load environment context from YAML/JSON config.

    Returns dict with 'tools', 'schema', 'description' keys if available.
    """
    for ext in [".yaml", ".yml", ".json"]:
        path = os.path.join(env_context_dir, f"{env_key}{ext}")
        if os.path.exists(path):
            if ext == ".json":
                with open(path) as f:
                    return json.load(f)
            else:
                try:
                    import yaml

                    with open(path) as f:
                        return yaml.safe_load(f)
                except ImportError:
                    pass
    return None


def format_task_spec(task: Dict[str, Any]) -> str:
    """Format a task as the expected output format for SFT.

    Returns the <task><prompt>...</prompt><verifier>...</verifier></task> string.
    """
    prompt = task.get("prompt", "")
    verifier = task.get("verifier_func") or task.get("verifier_code", "")

    return f"""<task>
<prompt>
{prompt}
</prompt>
<verifier>
{verifier}
</verifier>
</task>"""


def format_env_context_prompt(
    env_key: str,
    tools: List[str],
    schema: str = "",
    example_tasks: List[Dict[str, str]] = [],
) -> str:
    """Build the system prompt for task generation from env context."""
    tools_str = "\n".join(f"- {t}" for t in tools) if tools else "No tools listed."

    examples_str = ""
    if example_tasks:
        for i, ex in enumerate(example_tasks[:3], 1):
            examples_str += f"\n### Example {i}\n"
            examples_str += f"Prompt: {ex.get('prompt', '')}\n"

    schema_str = schema if schema else "No schema available."

    return f"""You are a task designer for the "{env_key}" environment.

## Environment: {env_key}

### Available Tools
{tools_str}

### Data Schema
{schema_str}

### Example Tasks
{examples_str}

## Your Output Format

Generate exactly ONE task with a prompt and verifier function.

<task>
<prompt>[Task instruction]</prompt>
<verifier>[Python async verify function]</verifier>
</task>"""


# ---------------------------------------------------------------------------
# Tool discovery via OpenEnv (Fleet provisioning)
# ---------------------------------------------------------------------------


async def _discover_env_tools_async(
    env_key: str,
    api_key: str,
    data_key: Optional[str] = None,
    data_version: Optional[str] = None,
    ttl_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """Provision a short-lived Fleet environment and list its tools.

    Uses OpenEnv's FleetEnvClient.from_fleet() (sync provisioning) then
    FleetMCPTools.list_tools() (async MCP call) to get tool schemas in
    OpenAI format.

    Passes data_key/data_version so the environment is provisioned with the
    correct data seed (required for some envs to expose their tools).

    Returns list of tool dicts: [{"type": "function", "function": {...}}, ...]
    """
    from envs.fleet_env.client import FleetEnvClient

    orch, tools_client = FleetEnvClient.from_fleet(
        api_key=api_key,
        env_key=env_key,
        data_key=data_key,
        data_version=data_version,
        image_type="standard",
        ttl_seconds=ttl_seconds,
    )
    try:
        result = await tools_client.list_tools()
        return result.tools
    finally:
        try:
            orch.close()
        except Exception as e:
            logger.warning(f"[{env_key}] Error closing environment: {e}")


def discover_env_tools(
    env_key: str,
    api_key: str,
    data_key: Optional[str] = None,
    data_version: Optional[str] = None,
    ttl_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """Sync wrapper: provision Fleet env → list_tools() → destroy.

    Returns tool schemas in OpenAI format, or empty list on failure.
    """
    try:
        return asyncio.run(_discover_env_tools_async(env_key, api_key, data_key, data_version, ttl_seconds))
    except Exception as e:
        if data_version:
            logger.warning(
                f"[{env_key}] Tool discovery failed with data_version={data_version}, "
                f"retrying without version (tools don't change across versions): {e}"
            )
            try:
                return asyncio.run(_discover_env_tools_async(env_key, api_key, data_key, None, ttl_seconds))
            except Exception as e2:
                logger.error(f"[{env_key}] Tool discovery failed on retry without version: {e2}")
                return []
        logger.error(f"[{env_key}] Tool discovery failed: {e}")
        return []


def _collect_env_metadata(
    tasks_by_env: Dict[str, List[Dict]],
) -> Dict[str, Dict[str, Any]]:
    """Collect per-environment metadata from tasks.

    For each environment, extracts:
    - data_key / data_version (from first task, same for all tasks in env)
    - env_variable_keys: sorted list of unique env_variable keys across all tasks

    Returns dict mapping env_key -> {"data_key": ..., "data_version": ..., "env_variable_keys": [...]}
    """
    result: Dict[str, Dict[str, Any]] = {}
    for env_key, env_tasks in tasks_by_env.items():
        # data_key/data_version are env-level (same across tasks)
        first_task = env_tasks[0]
        data_key = first_task.get("data_key")
        data_version = first_task.get("data_version")

        # Collect env_variable keys and representative values from first task
        all_var_keys: set = set()
        representative_env_vars: Dict[str, Any] = {}
        for t in env_tasks:
            env_vars = t.get("env_variables") or {}
            if isinstance(env_vars, str):
                try:
                    env_vars = json.loads(env_vars)
                except json.JSONDecodeError:
                    env_vars = {}
            all_var_keys.update(env_vars.keys())
            # Use first task's values as representative (same env config)
            if not representative_env_vars and env_vars:
                representative_env_vars = dict(env_vars)

        result[env_key] = {
            "data_key": data_key,
            "data_version": data_version,
            "env_variable_keys": sorted(all_var_keys),
            "env_variables": representative_env_vars,
        }
    return result


def discover_all_env_tools(
    env_keys: List[str],
    api_key: str,
    env_metadata: Dict[str, Dict[str, Any]],
    cache_path: Optional[str] = None,
    ttl_seconds: int = 300,
) -> Dict[str, List[Dict[str, Any]]]:
    """Discover tools for all unique env_keys, with optional JSON cache.

    Args:
        env_keys: List of environment keys (duplicates are deduplicated)
        api_key: Fleet API key
        env_metadata: Per-env metadata with data_key/data_version
        cache_path: If set, load/save discovered tools to this JSON file
        ttl_seconds: TTL for provisioned Fleet instances

    Returns:
        Dict mapping env_key -> list of OpenAI-format tool dicts
    """
    unique_keys = sorted(set(env_keys))

    # Load cache if available
    cached: Dict[str, List[Dict[str, Any]]] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded tool cache with {len(cached)} environments from {cache_path}")

    result: Dict[str, List[Dict[str, Any]]] = {}
    to_discover = []

    for key in unique_keys:
        if key in cached:
            result[key] = cached[key]
        else:
            to_discover.append(key)

    if to_discover:
        print(f"Discovering tools for {len(to_discover)} environments...")
        for i, key in enumerate(to_discover, 1):
            meta = env_metadata.get(key, {})
            dk = meta.get("data_key")
            dv = meta.get("data_version")
            print(f"  [{i}/{len(to_discover)}] {key} (data={dk}:{dv})...", end=" ", flush=True)
            tools = discover_env_tools(key, api_key, dk, dv, ttl_seconds)
            result[key] = tools
            tool_names = [t["function"]["name"] for t in tools if "function" in t]
            print(f"{len(tools)} tools: {tool_names[:5]}{'...' if len(tool_names) > 5 else ''}")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved tool cache to {cache_path}")

    return result


# ---------------------------------------------------------------------------
# DB schema discovery via Supabase + S3
# ---------------------------------------------------------------------------

# Tables to exclude from schema (internal/system tables)
_SYSTEM_TABLES = {
    "sqlite_sequence",
    "generation_checkpoints",
    "seed_progress",
    "__drizzle_migrations",
    "_imported_comment_ids",
    "_imported_post_ids",
    "_litestream_lock",
    "_litestream_seq",
}

# SQL keywords that aren't column names
_SQL_KEYWORDS = {
    "PRIMARY",
    "FOREIGN",
    "UNIQUE",
    "CHECK",
    "CONSTRAINT",
    "INDEX",
    "CREATE",
    "TABLE",
    "IF",
    "NOT",
    "EXISTS",
    "OR",
    "AND",
    "ON",
    "SET",
    "DEFAULT",
    "NULL",
    "REFERENCES",
    "CASCADE",
    "AUTOINCREMENT",
}


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments (-- line comments and /* block comments */)."""
    # Remove line comments
    sql = re.sub(r"--[^\n]*", "", sql)
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def parse_sql_schema(sql: str) -> Dict[str, List[Dict[str, str]]]:
    """Parse CREATE TABLE SQL into compact table→columns mapping.

    Returns dict: {table_name: [{"name": col, "type": type}, ...]}
    """
    sql = _strip_sql_comments(sql)
    tables: Dict[str, List[Dict[str, str]]] = {}
    # Match CREATE TABLE statements
    for match in re.finditer(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\);",
        sql,
        re.DOTALL | re.IGNORECASE,
    ):
        table_name = match.group(1)
        if table_name.lower() in _SYSTEM_TABLES:
            continue

        body = match.group(2)
        columns = []
        for line in body.split(","):
            line = line.strip()
            # Skip constraints, foreign keys, etc.
            if re.match(r"(PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK|CONSTRAINT|INDEX)", line, re.IGNORECASE):
                continue
            # Parse: [``"]col_name["``] TYPE ...
            col_match = re.match(r"[`\"]?(\w+)[`\"]?\s+(\w+)", line)
            if col_match:
                col_name = col_match.group(1)
                col_type = col_match.group(2).upper()
                # Skip if it looks like a keyword, not a column
                if col_name.upper() in _SQL_KEYWORDS:
                    continue
                columns.append({"name": col_name, "type": col_type})

        if columns:
            tables[table_name] = columns
    return tables


def format_compact_schema(tables: Dict[str, List[Dict[str, str]]]) -> str:
    """Format parsed schema as compact text for prompt injection.

    Example output:
        bookings: id (INTEGER), guest_first_name (TEXT), hotel_id (INTEGER), ...
        hotels: id (INTEGER), name (TEXT), city (TEXT), ...
    """
    lines = []
    for table_name in sorted(tables.keys()):
        cols = tables[table_name]
        col_strs = [f"{c['name']} ({c['type']})" for c in cols]
        lines.append(f"{table_name}: {', '.join(col_strs)}")
    return "\n".join(lines)


def discover_env_schemas(
    env_metadata: Dict[str, Dict[str, Any]],
    cache_path: Optional[str] = None,
) -> Dict[str, str]:
    """Discover DB schemas for environments via Supabase seed_versions → S3.

    Looks up each env_key's image_repo_name in the environments table, then
    finds the schema_s3_url in seed_versions, downloads the SQL, and parses
    it into a compact format.

    Args:
        env_metadata: Per-env metadata with data_key/data_version
        cache_path: Optional JSON cache file for schemas

    Returns:
        Dict mapping env_key -> compact schema string
    """
    supabase_url = os.environ.get("SUPABASE_URL", "https://ehefoavidbttssbleuyv.supabase.co")
    supabase_key = os.environ.get("SUPABASE_KEY", "")

    # Load cache
    cached: Dict[str, str] = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded schema cache with {len(cached)} environments from {cache_path}")

    # Return cache if all envs are covered
    to_discover = [k for k in env_metadata if k not in cached]
    if not to_discover:
        return cached

    if not supabase_key:
        print("WARNING: No SUPABASE_KEY set, skipping schema discovery")
        return cached

    try:
        from supabase import create_client

        sb = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"WARNING: Could not connect to Supabase: {e}")
        return cached

    # Get env_key → image_repo_name mapping
    envs_result = sb.table("environments").select("env_key,image_repo_name").execute()
    env_to_repo = {}
    for r in envs_result.data:
        if r.get("image_repo_name"):
            repo = r["image_repo_name"].replace("theseus/", "")
            env_to_repo[r["env_key"]] = repo

    result = dict(cached)
    print(f"Discovering schemas for {len(to_discover)} environments...")

    for env_key in sorted(to_discover):
        repo_name = env_to_repo.get(env_key, env_key)
        meta = env_metadata.get(env_key, {})
        data_key = meta.get("data_key")

        # Find schema_s3_url from seed_versions
        query = sb.table("seed_versions").select("schema_s3_url,data_key,version")
        query = query.eq("env_key", repo_name)
        if data_key:
            query = query.eq("data_key", data_key)
        query = query.not_.is_("schema_s3_url", "null")
        query = query.order("created_at", desc=True).limit(1)

        sv_result = query.execute()
        if not sv_result.data:
            # Try without data_key filter
            sv_result = (
                sb.table("seed_versions")
                .select("schema_s3_url,data_key,version")
                .eq("env_key", repo_name)
                .not_.is_("schema_s3_url", "null")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )

        if not sv_result.data or not sv_result.data[0].get("schema_s3_url"):
            print(f"  {env_key}: no schema found (repo={repo_name})")
            continue

        schema_url = sv_result.data[0]["schema_s3_url"]
        sv_data_key = sv_result.data[0]["data_key"]
        sv_version = sv_result.data[0]["version"]

        try:
            resp = requests.get(schema_url, timeout=30)
            resp.raise_for_status()
            raw_sql = resp.text
            tables = parse_sql_schema(raw_sql)
            compact = format_compact_schema(tables)
            result[env_key] = compact
            print(f"  {env_key}: {len(tables)} tables (data={sv_data_key}:{sv_version})")
        except Exception as e:
            print(f"  {env_key}: schema download failed: {e}")

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved schema cache to {cache_path}")

    return result


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_task_gen_dataset_sft(
    tasks_json: str,
    output_dir: str,
    env_context_dir: Optional[str] = None,
    eval_ratio: float = 0.15,
    min_verifier_len: int = 50,
    max_examples_per_env: int = 3,
    max_tasks: Optional[int] = None,
):
    """Build SFT dataset from existing Fleet tasks.

    Groups tasks by environment, creates (env_context, task_spec) pairs.
    Holds out some tasks per env as few-shot examples in the prompt.

    Args:
        tasks_json: Path to Fleet tasks JSON
        output_dir: Output directory for parquet files
        env_context_dir: Directory with per-env context configs
        eval_ratio: Fraction for evaluation split
        min_verifier_len: Minimum verifier code length to include
        max_examples_per_env: Number of example tasks to include in prompt
        max_tasks: Maximum total tasks to include (for testing)
    """
    print(f"Loading tasks from {tasks_json}...")
    tasks = load_tasks(tasks_json)
    print(f"Loaded {len(tasks)} tasks")

    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
        print(f"Truncated to {max_tasks} tasks")

    # Filter: must have verifier code
    tasks_with_verifier = []
    for t in tasks:
        verifier = t.get("verifier_func") or t.get("verifier_code", "")
        if verifier and len(verifier) >= min_verifier_len:
            tasks_with_verifier.append(t)
    print(f"Tasks with verifier (>= {min_verifier_len} chars): {len(tasks_with_verifier)}")

    # Group by environment
    tasks_by_env: Dict[str, List[Dict]] = defaultdict(list)
    for t in tasks_with_verifier:
        env_key = t.get("env_key") or t.get("env_id") or "unknown"
        tasks_by_env[env_key].append(t)

    print(f"\nEnvironments: {len(tasks_by_env)}")
    for env_key, env_tasks in sorted(tasks_by_env.items()):
        print(f"  {env_key}: {len(env_tasks)} tasks")

    # Build dataset records
    all_records = []

    for env_key, env_tasks in tasks_by_env.items():
        # Load env context if available
        env_ctx = None
        if env_context_dir:
            env_ctx = load_env_context(env_context_dir, env_key)

        # Extract tool names from env context or from task verifiers
        tools = []
        if env_ctx:
            tools = env_ctx.get("tools", [])

        # Use first N tasks as few-shot examples, rest as training targets
        example_tasks = env_tasks[:max_examples_per_env]
        target_tasks = env_tasks[max_examples_per_env:]

        if not target_tasks:
            # If too few tasks, use all as both examples and targets
            target_tasks = env_tasks

        # Build the system prompt for this environment
        example_dicts = [{"prompt": t.get("prompt", "")} for t in example_tasks]

        system_prompt = format_env_context_prompt(
            env_key=env_key,
            tools=tools,
            schema=env_ctx.get("schema", "") if env_ctx else "",
            example_tasks=example_dicts,
        )

        for task in target_tasks:
            task_spec = format_task_spec(task)
            task_key = task.get("key") or task.get("task_key", "unknown")

            record = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Generate a task for the {env_key} environment.",
                    },
                ],
                "response": task_spec,
                "env_class": "task_gen",
                "data_source": env_key,
                "task_key": task_key,
                # Store env context as extras for the TaskGenEnv
                "env_key": env_key,
                "env_version": task.get("version") or task.get("env_version", ""),
                "env_tools": json.dumps(tools),
            }
            all_records.append(record)

    print(f"\nTotal records: {len(all_records)}")

    # Split into train/eval
    train_records = []
    eval_records = []

    for record in all_records:
        h = hashlib.md5(record["task_key"].encode()).hexdigest()
        if int(h[:8], 16) / (16**8) < eval_ratio:
            eval_records.append(record)
        else:
            train_records.append(record)

    print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    if train_records:
        train_ds = Dataset.from_list(train_records)
        train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
        print(f"Saved train to {output_dir}/train.parquet")

    if eval_records:
        eval_ds = Dataset.from_list(eval_records)
        eval_ds.to_parquet(os.path.join(output_dir, "validation.parquet"))
        print(f"Saved validation to {output_dir}/validation.parquet")

    # Print per-env breakdown
    env_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "eval": 0})
    for r in train_records:
        env_counts[r["data_source"]]["train"] += 1
    for r in eval_records:
        env_counts[r["data_source"]]["eval"] += 1

    print(f"\n{'Environment':<20} {'Train':>8} {'Eval':>8}")
    print("-" * 40)
    for env_key in sorted(env_counts.keys()):
        c = env_counts[env_key]
        print(f"{env_key:<20} {c['train']:>8} {c['eval']:>8}")


def build_task_gen_dataset_grpo(
    tasks_json: str,
    output_dir: str,
    eval_ratio: float = 0.15,
    min_verifier_len: int = 50,
    max_tasks: Optional[int] = None,
    discover_tools: bool = True,
    discover_schemas: bool = True,
    tools_cache: Optional[str] = None,
    schema_cache: Optional[str] = None,
    api_key: Optional[str] = None,
    env_keys_filter: Optional[List[str]] = None,
):
    """Build GRPO dataset from existing Fleet tasks.

    Creates prompt-only records: each record has the env context as the prompt
    (system + user messages) but no response. The reward comes from inner-loop
    rollouts during GRPO training.

    When discover_tools=True, provisions each Fleet environment to discover
    real tool schemas via MCP. These are stored as env_tools_schema (full
    OpenAI-format JSON) and env_tools (tool name list) in each record.

    When discover_schemas=True, queries Supabase seed_versions for DB schemas,
    downloads from S3, and stores compact table→column mappings.

    Args:
        tasks_json: Path to Fleet tasks JSON
        output_dir: Output directory for parquet files
        eval_ratio: Fraction for evaluation split
        min_verifier_len: Minimum verifier code length to include
        max_tasks: Maximum total tasks to include (for testing)
        discover_tools: If True, provision Fleet envs to discover tools
        discover_schemas: If True, fetch DB schemas from Supabase/S3
        tools_cache: Path to JSON cache file for discovered tools
        schema_cache: Path to JSON cache file for discovered schemas
        api_key: Fleet API key (required if discover_tools=True)
        env_keys_filter: If set, only include these environment keys
    """
    print(f"Loading tasks from {tasks_json}...")
    tasks = load_tasks(tasks_json)
    print(f"Loaded {len(tasks)} tasks")

    # Filter: must have verifier code (so we know the env produces real tasks)
    tasks_with_verifier = []
    for t in tasks:
        verifier = t.get("verifier_func") or t.get("verifier_code", "")
        if verifier and len(verifier) >= min_verifier_len:
            tasks_with_verifier.append(t)
    print(f"Tasks with verifier (>= {min_verifier_len} chars): {len(tasks_with_verifier)}")

    # Exclude environments known to overflow context (too many tools)
    _EXCLUDED_ENVS = {"github"}
    before = len(tasks_with_verifier)
    tasks_with_verifier = [
        t for t in tasks_with_verifier if (t.get("env_key") or t.get("env_id") or "unknown") not in _EXCLUDED_ENVS
    ]
    if before != len(tasks_with_verifier):
        print(f"Excluded {_EXCLUDED_ENVS}: {before} -> {len(tasks_with_verifier)} tasks")

    # Filter by env_keys if specified (before max_tasks truncation)
    if env_keys_filter:
        allowed = set(env_keys_filter)
        before = len(tasks_with_verifier)
        tasks_with_verifier = [
            t for t in tasks_with_verifier if (t.get("env_key") or t.get("env_id") or "unknown") in allowed
        ]
        print(f"Filtered to env_keys={env_keys_filter}: {before} -> {len(tasks_with_verifier)} tasks")

    # Truncate after filtering
    if max_tasks and len(tasks_with_verifier) > max_tasks:
        tasks_with_verifier = tasks_with_verifier[:max_tasks]
        print(f"Truncated to {max_tasks} tasks")

    # Group by environment
    tasks_by_env: Dict[str, List[Dict]] = defaultdict(list)
    for t in tasks_with_verifier:
        env_key = t.get("env_key") or t.get("env_id") or "unknown"
        tasks_by_env[env_key].append(t)

    print(f"\nEnvironments: {len(tasks_by_env)}")
    for env_key, env_tasks in sorted(tasks_by_env.items()):
        print(f"  {env_key}: {len(env_tasks)} tasks")

    # Collect per-env metadata (data_key, data_version, env_variable_keys)
    env_metadata = _collect_env_metadata(tasks_by_env)

    print("\nEnvironment metadata:")
    for env_key in sorted(env_metadata.keys()):
        meta = env_metadata[env_key]
        print(f"  {env_key}: data={meta['data_key']}:{meta['data_version']}, " f"env_vars={meta['env_variable_keys']}")

    # Discover tools from Fleet environments
    env_tools_map: Dict[str, List[Dict[str, Any]]] = {}
    if discover_tools:
        if not api_key:
            api_key = os.environ.get("FLEET_API_KEY")
        if not api_key:
            print("WARNING: No FLEET_API_KEY set, skipping tool discovery")
        else:
            env_tools_map = discover_all_env_tools(
                env_keys=list(tasks_by_env.keys()),
                api_key=api_key,
                env_metadata=env_metadata,
                cache_path=tools_cache,
            )

    # Discover DB schemas from Supabase/S3
    env_schemas_map: Dict[str, str] = {}
    if discover_schemas:
        env_schemas_map = discover_env_schemas(
            env_metadata=env_metadata,
            cache_path=schema_cache,
        )

    # Build GRPO records: one prompt per task (prompt-only, no response)
    all_records = []

    for env_key, env_tasks in tasks_by_env.items():
        tool_schemas = env_tools_map.get(env_key, [])
        tool_names = [t["function"]["name"] for t in tool_schemas if "function" in t]
        meta = env_metadata.get(env_key, {})
        env_var_keys = meta.get("env_variable_keys", [])
        env_schema = env_schemas_map.get(env_key, "")

        for task in env_tasks:
            task_key = task.get("key") or task.get("task_key", "unknown")
            record = {
                "prompt": [
                    {"role": "system", "content": ""},  # built at runtime by TaskGenEnv
                    {
                        "role": "user",
                        "content": f"Generate a task for the {env_key} environment.",
                    },
                ],
                "env_class": "task_gen",
                "data_source": env_key,
                "task_key": task_key,
                "env_key": env_key,
                "env_version": task.get("version") or task.get("env_version", ""),
                "data_key": task.get("data_key") or "",
                "data_version": task.get("data_version") or "",
                "env_tools": json.dumps(tool_names),
                "env_tools_schema": json.dumps(tool_schemas),
                "env_variable_keys": json.dumps(env_var_keys),
                "env_variables": json.dumps(meta.get("env_variables", {})),
                "env_schema": env_schema,
            }
            all_records.append(record)

    print(f"\nTotal GRPO records: {len(all_records)}")

    # Print tool discovery summary
    if env_tools_map:
        print("\nTool discovery summary:")
        for env_key in sorted(env_tools_map.keys()):
            schemas = env_tools_map[env_key]
            names = [t["function"]["name"] for t in schemas if "function" in t]
            print(f"  {env_key}: {len(names)} tools")

    # Split into train/eval
    train_records = []
    eval_records = []

    for record in all_records:
        h = hashlib.md5(record["task_key"].encode()).hexdigest()
        if int(h[:8], 16) / (16**8) < eval_ratio:
            eval_records.append(record)
        else:
            train_records.append(record)

    print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    if train_records:
        train_ds = Dataset.from_list(train_records)
        train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
        print(f"Saved train to {output_dir}/train.parquet")

    if eval_records:
        eval_ds = Dataset.from_list(eval_records)
        eval_ds.to_parquet(os.path.join(output_dir, "validation.parquet"))
        print(f"Saved validation to {output_dir}/validation.parquet")

    # Print per-env breakdown
    env_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "eval": 0})
    for r in train_records:
        env_counts[r["data_source"]]["train"] += 1
    for r in eval_records:
        env_counts[r["data_source"]]["eval"] += 1

    print(f"\n{'Environment':<20} {'Train':>8} {'Eval':>8}")
    print("-" * 40)
    for env_key in sorted(env_counts.keys()):
        c = env_counts[env_key]
        print(f"{env_key:<20} {c['train']:>8} {c['eval']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for task generation")
    parser.add_argument(
        "--tasks-json",
        type=str,
        required=True,
        help="Path to Fleet tasks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/task_gen",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="grpo",
        choices=["grpo", "sft"],
        help="Dataset mode: 'grpo' (prompt-only) or 'sft' (prompt+response)",
    )
    parser.add_argument(
        "--env-context-dir",
        type=str,
        default=None,
        help="Directory with per-env context configs (YAML/JSON)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for evaluation",
    )
    parser.add_argument(
        "--min-verifier-len",
        type=int,
        default=50,
        help="Minimum verifier code length to include",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to include (for testing)",
    )
    parser.add_argument(
        "--env-keys",
        type=str,
        default=None,
        help="Comma-separated environment filter (e.g., ticketmaster,booking)",
    )
    parser.add_argument(
        "--no-discover-tools",
        action="store_true",
        help="Skip Fleet provisioning for tool discovery (local testing)",
    )
    parser.add_argument(
        "--tools-cache",
        type=str,
        default=None,
        help="JSON cache file for discovered tools (skip re-provisioning on re-runs)",
    )
    parser.add_argument(
        "--no-discover-schemas",
        action="store_true",
        help="Skip Supabase/S3 schema discovery (local testing without credentials)",
    )
    parser.add_argument(
        "--schema-cache",
        type=str,
        default=None,
        help="JSON cache file for discovered DB schemas",
    )

    args = parser.parse_args()

    env_keys_filter = [k.strip() for k in args.env_keys.split(",")] if args.env_keys else None

    if args.mode == "grpo":
        build_task_gen_dataset_grpo(
            tasks_json=args.tasks_json,
            output_dir=args.output_dir,
            eval_ratio=args.eval_ratio,
            min_verifier_len=args.min_verifier_len,
            max_tasks=args.max_tasks,
            discover_tools=not args.no_discover_tools,
            discover_schemas=not args.no_discover_schemas,
            tools_cache=args.tools_cache,
            schema_cache=args.schema_cache,
            env_keys_filter=env_keys_filter,
        )
    else:
        build_task_gen_dataset_sft(
            tasks_json=args.tasks_json,
            output_dir=args.output_dir,
            env_context_dir=args.env_context_dir,
            eval_ratio=args.eval_ratio,
            min_verifier_len=args.min_verifier_len,
            max_tasks=args.max_tasks,
        )


if __name__ == "__main__":
    main()
