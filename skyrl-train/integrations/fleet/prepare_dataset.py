"""
Prepare Fleet tasks for SkyRL training.

Converts Fleet task JSON files to SkyRL parquet dataset format.

Usage:
    python -m integrations.fleet.prepare_dataset \
        --tasks-json /path/to/all_tool_use.json \
        --output-dir ./data/fleet \
        --modality tool_use

Split Strategy:
    - Stratified by environment (each env maintains train/eval ratio)
    - Hash-based deterministic assignment (same task always goes to same split)
    - 20% eval ratio, capped at 20 samples per env (MAX_EVAL_SAMPLES)
    - Minimum 5 eval samples per env (otherwise all go to train)
    - Held-out eval envs: instacart (computer_use only)

v0.3.2 Changes:
    - Increased eval_ratio from 10% to 20% to include carlisle/outlook in eval
    - Result: 11 envs in eval (was 9), ~183 eval samples (was ~146)

v0.3.1 Changes:
    - Added MAX_ENV_TRAIN_RATIO=0.20 to prevent any single env from dominating
    - Hash-based deterministic sampling for reproducibility

v0.3.0 Changes:
    - Increased eval_ratio from 2% to 10%
    - Added MAX_EVAL_SAMPLES=30 cap per environment
    - MIN_EVAL_SAMPLES stays at 5
    - Result: ticketmaster now gets ~22 eval samples for trace analysis
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from datasets import Dataset

# Held-out environments for eval only (not used in train)
HELD_OUT_ENVS = {
    "tool_use": [],  # v0.3: all envs split normally (outlook now included in train)
    "computer_use": [],
}

# Excluded environments (removed from both train and eval)
# v0.3.6: google-maps excluded due to broken MCP server (502 errors, "database is locked")
# v0.4.0: dropbox excluded due to broken env (instance creation timeouts)
EXCLUDED_ENVS = {
    "tool_use": ["dropbox"],
    "computer_use": ["dropbox"],
}

# Tasks excluded due to missing CURRENT_DATE in env_variables (v0.4.0)
# These tasks have partial dates (e.g., "January 30th" without year) but their
# tool calls require mm/dd/yy format. Without CURRENT_DATE, the model cannot
# compute the correct year, causing date validation failures.
# See: https://github.com/fleet-ai/SkyRL/pull/246
TASKS_MISSING_CURRENT_DATE = {
    "task_a44hx6crecg4_1769052238469_i7dxxtjvq",  # zillow - February 1st
    "task_a7rlslof7gdy_1768337837679_8be6pguu3",  # zillow - March 11th
    "task_axtmgwocana_1768544478249_k2ozcylyf",  # zillow - January 21st
    "task_b1fxgn0k3yms_1768542773490_ddbhj5bai",  # zillow - January 30th
    "task_b4v77hb3owof_1768546181946_efsedxv9g",  # zillow - February 14th
    "task_b5zt6ipf0nbl_1768346335430_i23gknp4t",  # zillow - January 15th
    "task_bafrpi5qgyzh_1768546181946_2cebmq91r",  # zillow - February 14th
    "task_bdmnfipwxlqv_1769052238469_4nglwjqfm",  # zillow - February 1st
    "task_bxqzfjc2dbte_1768337837679_2qvnm9rq7",  # zillow - March 11th
    "task_c3jwlxmfvbop_1768544478249_efo6hxylr",  # zillow - January 21st
    "task_c7o0c7ehhv9t_1768542773490_2t9w2l1z5",  # zillow - January 30th
    "task_ceqj4h9t0ygi_1768346335430_8j1w8w5xp",  # zillow - January 15th
    "task_cgpxfxp78bvp_1768346335430_6v4n8wlt8",  # zillow - January 15th
    "task_cgsz56tqjlv6_1768346335430_hqgsjy4wt",  # zillow - January 15th
    "task_dpv4bpdpz6db_1768542773490_f3g6w8e8g",  # zillow - January 30th
    "task_f7lgb6fxfwln_1768337837679_d1dxk6ahv",  # zillow - March 11th
    "task_fl1rq3d2wbj9_1768337837679_d2x4k8p93",  # zillow - March 11th
    "task_fn1k5mvjx6r1_1768544478249_1nfmnp6r2",  # zillow - January 21st
    "task_fnh5f0x7hv6w_1768544478249_8wptm6zqp",  # zillow - January 21st
    "task_g2dwb1rfx69c_1769052238469_bc1y9h9d7",  # zillow - February 1st
    "task_g3wpj1mcl0lf_1768546181946_59vtqn9fw",  # zillow - February 14th
}

# Minimum number of samples required to create an eval split for an env
MIN_EVAL_SAMPLES = 5

# Maximum number of eval samples per environment (v0.3.1: reduced from 30 to 20)
# Ensures small envs get eval traces without blowing up eval set size
MAX_EVAL_SAMPLES = 20

# Maximum fraction of training data any single environment can have (v0.3.1)
# Prevents dominant environments from skewing training
MAX_ENV_TRAIN_RATIO = 0.20

# Maximum total eval prompts across all environments (v0.3.2)
# With eval_n_samples_per_prompt=3 and 30s per trajectory:
# 60 prompts × 3 samples = 180 trajectories ≈ 90 minutes per eval
MAX_EVAL_PROMPTS = 60


def load_tasks_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load tasks from JSON file (Fleet export format)."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both formats: array or {"tasks": [...]}
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    else:
        raise ValueError("Invalid JSON format: expected array or object with 'tasks' key")


def hash_to_split(task_key: str, eval_ratio: float = 0.10) -> str:
    """Deterministically assign task to train or eval based on hash.

    Uses MD5 hash of task_key to get a deterministic float in [0, 1).
    This ensures the same task always goes to the same split.
    """
    hash_bytes = hashlib.md5(task_key.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    hash_float = hash_int / (2**64)
    return "eval" if hash_float < eval_ratio else "train"


def hash_to_float(task_key: str) -> float:
    """Convert task_key to deterministic float in [0, 1) for sampling."""
    hash_bytes = hashlib.md5(task_key.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    return hash_int / (2**64)


def cap_training_distribution(
    train_records: List[Dict[str, Any]],
    max_env_ratio: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Cap each environment's contribution to training data.

    Uses hash-based deterministic sampling so the same tasks are always selected.

    Args:
        train_records: List of training records with 'data_source' (env_key) and 'task_key'
        max_env_ratio: Maximum fraction any single env can contribute (e.g., 0.20 = 20%)

    Returns:
        Tuple of (capped_records, cap_stats) where cap_stats shows per-env before/after counts
    """
    if max_env_ratio >= 1.0:
        return train_records, {}

    total_train = len(train_records)
    max_per_env = int(total_train * max_env_ratio)

    # Group by environment
    records_by_env: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        env_key = record.get("data_source", "unknown")
        records_by_env[env_key].append(record)

    # Cap each environment
    capped_records = []
    cap_stats: Dict[str, Dict[str, int]] = {}

    for env_key, records in records_by_env.items():
        before_count = len(records)

        if before_count <= max_per_env:
            # No capping needed
            capped_records.extend(records)
            cap_stats[env_key] = {"before": before_count, "after": before_count, "capped": False}
        else:
            # Sort by hash for deterministic selection
            records_sorted = sorted(records, key=lambda r: hash_to_float(r.get("task_key", "")))
            selected = records_sorted[:max_per_env]
            capped_records.extend(selected)
            cap_stats[env_key] = {"before": before_count, "after": max_per_env, "capped": True}

    return capped_records, cap_stats


def prepare_fleet_dataset(
    tasks_json: str,
    output_dir: str,
    modality: Optional[str] = "tool_use",
    eval_ratio: float = 0.20,  # v0.3.2: increased to 20% to include carlisle/outlook in eval
    env_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None,  # v0.4.0: filter by difficulty (1=easy, 2=medium, 3=hard)
    max_tasks: Optional[int] = None,
    max_env_ratio: float = MAX_ENV_TRAIN_RATIO,  # v0.3.1: cap dominant environments
    max_eval_prompts: Optional[int] = MAX_EVAL_PROMPTS,  # v0.3.2: cap total eval prompts
):
    """
    Convert Fleet tasks JSON to SkyRL parquet dataset.

    Args:
        tasks_json: Path to Fleet tasks JSON file
        output_dir: Output directory for parquet files
        modality: Task modality filter ("tool_use" or "computer_use"), None for all
        eval_ratio: Fraction of data for evaluation (default: 0.02)
        env_filter: Optional env_key filter (e.g., "github", "booking")
        max_tasks: Optional maximum number of tasks to include
        max_env_ratio: Maximum fraction any single env can contribute to training (default: 0.20)
    """
    # Log applied filters at the start
    print("\n=== Dataset Filters ===")
    print(f"  Source: {tasks_json}")
    print(f"  Modality: {modality or 'all'}")
    print(f"  Env filter: {env_filter or 'none'}")
    print(f"  Difficulty filter: {difficulty_filter or 'all (1,2,3)'}")
    print(f"  Max tasks: {max_tasks or 'unlimited'}")
    print(f"  Max env ratio: {max_env_ratio:.0%}")
    print(f"  Max eval prompts: {max_eval_prompts or 'unlimited'}")
    print()

    print(f"Loading tasks from {tasks_json}...")
    tasks = load_tasks_from_json(tasks_json)
    print(f"Loaded {len(tasks)} tasks")

    # Filter by modality if specified
    if modality:
        tasks = [t for t in tasks if t.get("task_modality") == modality]
        print(f"After modality filter ({modality}): {len(tasks)} tasks")

    # Filter by env_key(s) if specified - supports comma-separated list
    if env_filter:
        env_list = [e.strip() for e in env_filter.split(",") if e.strip()]
        tasks = [t for t in tasks if t.get("env_key") in env_list or t.get("env_id") in env_list]
        print(f"After env filter ({env_list}): {len(tasks)} tasks")

    # Filter by difficulty if specified - supports comma-separated list (e.g., "1,2" for easy+medium)
    if difficulty_filter:
        diff_list = [int(d.strip()) for d in difficulty_filter.split(",") if d.strip()]
        tasks = [t for t in tasks if t.get("difficulty") in diff_list]
        print(f"After difficulty filter ({diff_list}): {len(tasks)} tasks")

    # Limit tasks if specified
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
        print(f"Limited to {max_tasks} tasks")

    if not tasks:
        print("No tasks remaining after filtering. Exiting.")
        return

    # Deduplicate by task_key (keep first occurrence)
    seen_task_keys: set = set()
    unique_tasks = []
    duplicate_count = 0
    env_duplicate_counts: Dict[str, int] = defaultdict(int)

    for task in tasks:
        task_key = task.get("key") or task.get("task_key")
        if not task_key:
            continue
        if task_key in seen_task_keys:
            duplicate_count += 1
            env_key = task.get("env_key") or task.get("env_id") or "unknown"
            env_duplicate_counts[env_key] += 1
        else:
            seen_task_keys.add(task_key)
            unique_tasks.append(task)

    if duplicate_count > 0:
        print(f"\n⚠️  WARNING: Removed {duplicate_count} duplicate task_keys")
        print("  By environment:")
        for env, count in sorted(env_duplicate_counts.items(), key=lambda x: -x[1]):
            print(f"    {env}: {count} duplicates removed")
        print()

    tasks = unique_tasks
    print(f"After deduplication: {len(tasks)} unique tasks")

    # Get excluded envs for this modality (removed entirely)
    excluded_envs = set(EXCLUDED_ENVS.get(modality, []))
    if excluded_envs:
        before_count = len(tasks)
        tasks = [t for t in tasks if t.get("env_key") not in excluded_envs]
        print(f"Excluded environments: {excluded_envs}")
        print(f"After excluding envs: {len(tasks)} tasks (removed {before_count - len(tasks)})")

    # Exclude specific tasks missing CURRENT_DATE
    if TASKS_MISSING_CURRENT_DATE:
        before_count = len(tasks)
        tasks = [t for t in tasks if (t.get("key") or t.get("task_key")) not in TASKS_MISSING_CURRENT_DATE]
        removed = before_count - len(tasks)
        if removed > 0:
            print(f"Excluded tasks missing CURRENT_DATE: {removed} tasks")

    # Get held-out envs for this modality
    held_out_envs = set(HELD_OUT_ENVS.get(modality, []))
    if held_out_envs:
        print(f"Held-out test environments: {held_out_envs}")

    # Group tasks by environment
    tasks_by_env: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        env_key = task.get("env_key") or task.get("env_id") or "unknown"
        tasks_by_env[env_key].append(task)

    # Prepare records with stratified split
    train_records = []
    eval_records = []

    # Track per-env counts for summary table
    env_split_counts: Dict[str, Dict[str, int]] = {}

    print("\n=== Per-Environment Split ===")
    for env_key in sorted(tasks_by_env.keys()):
        env_tasks = tasks_by_env[env_key]

        # Check if this env is held out for eval only
        if env_key in held_out_envs:
            env_eval_count = 0
            for task in env_tasks:
                record = _task_to_record(task, env_key)
                if record:
                    eval_records.append(record)
                    env_eval_count += 1
            env_split_counts[env_key] = {"train": 0, "eval": env_eval_count}
            print(f"  {env_key}: {len(env_tasks)} -> EVAL only (held-out)")
            continue

        # Calculate target eval size: use ratio but cap at MAX_EVAL_SAMPLES
        target_eval_size = min(int(len(env_tasks) * eval_ratio), MAX_EVAL_SAMPLES)

        # If not enough samples for eval, put all in train
        if target_eval_size < MIN_EVAL_SAMPLES:
            env_train_count = 0
            for task in env_tasks:
                record = _task_to_record(task, env_key)
                if record:
                    train_records.append(record)
                    env_train_count += 1
            env_split_counts[env_key] = {"train": env_train_count, "eval": 0}
            print(f"  {env_key}: {len(env_tasks)} -> all TRAIN (< {MIN_EVAL_SAMPLES} eval samples)")
            continue

        # Compute effective eval ratio to achieve target_eval_size (capped at MAX_EVAL_SAMPLES)
        effective_eval_ratio = target_eval_size / len(env_tasks)

        # Stratified split using hash with effective ratio
        env_train = 0
        env_eval = 0
        for task in env_tasks:
            task_key = task.get("key") or task.get("task_key")
            record = _task_to_record(task, env_key)
            if not record:
                continue

            split = hash_to_split(task_key, effective_eval_ratio)
            if split == "eval":
                eval_records.append(record)
                env_eval += 1
            else:
                train_records.append(record)
                env_train += 1

        env_split_counts[env_key] = {"train": env_train, "eval": env_eval}
        print(f"  {env_key}: {len(env_tasks)} -> {env_train} train, {env_eval} eval")

    print(f"\nTotal: {len(train_records)} train, {len(eval_records)} eval")

    # Apply total eval cap (v0.3.2) - stratified sampling across environments
    if max_eval_prompts and len(eval_records) > max_eval_prompts:
        print(f"\n=== Capping Eval Prompts ({max_eval_prompts} max total) ===")

        # Group by environment
        eval_by_env: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in eval_records:
            eval_by_env[record.get("data_source", "unknown")].append(record)

        # Take min(4, available) from each env, then distribute remaining quota proportionally
        min_per_env = 4
        capped_eval_records = []

        for env_key, records in eval_by_env.items():
            # Sort by hash for deterministic selection
            records.sort(key=lambda r: hash_to_float(r.get("task_key", "")))
            # Take at least min_per_env (or all if fewer available)
            take = min(min_per_env, len(records))
            capped_eval_records.extend(records[:take])

        # If we have budget remaining, distribute round-robin across envs
        remaining_budget = max_eval_prompts - len(capped_eval_records)
        if remaining_budget > 0:
            # Records not yet selected (sorted by hash for determinism)
            remaining_by_env = {
                env: records[min_per_env:] for env, records in eval_by_env.items() if len(records) > min_per_env
            }

            # Round-robin until budget exhausted
            env_keys = sorted(remaining_by_env.keys())
            idx = 0
            while remaining_budget > 0 and any(remaining_by_env.values()):
                env = env_keys[idx % len(env_keys)]
                if remaining_by_env[env]:
                    capped_eval_records.append(remaining_by_env[env].pop(0))
                    remaining_budget -= 1
                idx += 1

        # Update env_split_counts
        for env_key in eval_by_env:
            count = sum(1 for r in capped_eval_records if r.get("data_source") == env_key)
            if env_key in env_split_counts:
                env_split_counts[env_key]["eval"] = count
            print(f"  {env_key}: {len(eval_by_env[env_key])} -> {count}")

        eval_records = capped_eval_records
        print(f"\nAfter capping: {len(eval_records)} eval prompts")

    # Apply per-environment cap to training data (v0.3.1)
    if max_env_ratio < 1.0 and train_records:
        train_records, cap_stats = cap_training_distribution(train_records, max_env_ratio)

        # Print capping summary
        capped_envs = [env for env, stats in cap_stats.items() if stats["capped"]]
        if capped_envs:
            print(f"\n=== Training Distribution Cap ({max_env_ratio:.0%} max per env) ===")
            for env in sorted(capped_envs):
                stats = cap_stats[env]
                print(f"  {env}: {stats['before']} -> {stats['after']} ({stats['before'] - stats['after']} removed)")
            print(f"\nAfter capping: {len(train_records)} train")

        # Update env_split_counts with capped values
        for env, stats in cap_stats.items():
            if env in env_split_counts:
                env_split_counts[env]["train"] = stats["after"]

    # Create datasets
    train_dataset = Dataset.from_list(train_records) if train_records else None
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)

    if train_dataset:
        train_path = os.path.join(output_dir, "train.parquet")
        train_dataset.to_parquet(train_path)
        print(f"Saved train dataset to {train_path}")

    if eval_dataset:
        eval_path = os.path.join(output_dir, "validation.parquet")
        eval_dataset.to_parquet(eval_path)
        print(f"Saved validation dataset to {eval_path}")

    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Train: {len(train_records)}")
    print(f"Eval:  {len(eval_records)} (includes held-out: {held_out_envs or 'none'})")

    # Print per-environment breakdown table
    print("\n=== Per-Environment Breakdown ===")
    print(f"{'Environment':<20} {'Train':>8} {'Eval':>8} {'Total':>8}")
    print("-" * 48)
    for env_key in sorted(env_split_counts.keys()):
        counts = env_split_counts[env_key]
        total = counts["train"] + counts["eval"]
        print(f"{env_key:<20} {counts['train']:>8} {counts['eval']:>8} {total:>8}")
    print("-" * 48)
    print(
        f"{'TOTAL':<20} {len(train_records):>8} {len(eval_records):>8} " f"{len(train_records) + len(eval_records):>8}"
    )


def _task_to_record(task: Dict[str, Any], env_key: str) -> Optional[Dict[str, Any]]:
    """Convert a task dict to a dataset record."""
    task_key = task.get("key") or task.get("task_key")
    prompt = task.get("prompt", "")

    if not task_key or not prompt:
        return None

    return {
        # Required fields for SkyRL
        "prompt": [{"role": "user", "content": prompt}],
        "env_class": "fleet_task",  # This tells SkyRL to use FleetTaskEnv
        # Task identification (passed as env_extras)
        "task_key": task_key,
        # Data source for per-environment metrics in WandB
        "data_source": env_key,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Fleet tasks for SkyRL training")
    parser.add_argument(
        "--tasks-json",
        type=str,
        required=True,
        help="Path to Fleet tasks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/fleet",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="tool_use",
        choices=["tool_use", "computer_use", "all"],
        help="Task modality filter ('all' for no filter)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.20,
        help="Fraction of data for evaluation (default: 0.20)",
    )
    parser.add_argument(
        "--env-filter",
        type=str,
        default=None,
        help="Optional env_key filter (e.g., 'github', 'booking')",
    )
    parser.add_argument(
        "--difficulty-filter",
        type=str,
        default=None,
        help="Optional difficulty filter: 1=easy, 2=medium, 3=hard (e.g., '1,2' for easy+medium)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to include",
    )
    parser.add_argument(
        "--max-env-ratio",
        type=float,
        default=MAX_ENV_TRAIN_RATIO,
        help=f"Maximum fraction of training data per environment (default: {MAX_ENV_TRAIN_RATIO})",
    )
    parser.add_argument(
        "--max-eval-prompts",
        type=int,
        default=MAX_EVAL_PROMPTS,
        help=f"Maximum total eval prompts across all environments (default: {MAX_EVAL_PROMPTS})",
    )

    args = parser.parse_args()

    # Handle 'all' modality
    modality = None if args.modality == "all" else args.modality

    prepare_fleet_dataset(
        tasks_json=args.tasks_json,
        output_dir=args.output_dir,
        modality=modality,
        eval_ratio=args.eval_ratio,
        env_filter=args.env_filter,
        difficulty_filter=args.difficulty_filter,
        max_tasks=args.max_tasks,
        max_env_ratio=args.max_env_ratio,
        max_eval_prompts=args.max_eval_prompts,
    )


if __name__ == "__main__":
    main()
