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
    - Minimum 5 eval samples per env (otherwise all go to train)
    - Held-out eval envs: instacart (computer_use), outlook (tool_use)
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
    "tool_use": ["outlook"],
    "computer_use": ["instacart"],
}

# Minimum number of samples required to create an eval split for an env
MIN_EVAL_SAMPLES = 5


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


def hash_to_split(task_key: str, eval_ratio: float = 0.02) -> str:
    """Deterministically assign task to train or eval based on hash.

    Uses MD5 hash of task_key to get a deterministic float in [0, 1).
    This ensures the same task always goes to the same split.
    """
    hash_bytes = hashlib.md5(task_key.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder="big")
    hash_float = hash_int / (2**64)
    return "eval" if hash_float < eval_ratio else "train"


def prepare_fleet_dataset(
    tasks_json: str,
    output_dir: str,
    modality: Optional[str] = "tool_use",
    eval_ratio: float = 0.02,
    env_filter: Optional[str] = None,
    max_tasks: Optional[int] = None,
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
    """
    print(f"Loading tasks from {tasks_json}...")
    tasks = load_tasks_from_json(tasks_json)
    print(f"Loaded {len(tasks)} tasks")

    # Filter by modality if specified
    if modality:
        tasks = [t for t in tasks if t.get("task_modality") == modality]
        print(f"After modality filter ({modality}): {len(tasks)} tasks")

    # Filter by env_key if specified
    if env_filter:
        tasks = [t for t in tasks if t.get("env_key") == env_filter or t.get("env_id") == env_filter]
        print(f"After env filter ({env_filter}): {len(tasks)} tasks")

    # Limit tasks if specified
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
        print(f"Limited to {max_tasks} tasks")

    if not tasks:
        print("No tasks remaining after filtering. Exiting.")
        return

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

        # Calculate expected eval size
        expected_eval_size = int(len(env_tasks) * eval_ratio)

        # If not enough samples for eval, put all in train
        if expected_eval_size < MIN_EVAL_SAMPLES:
            env_train_count = 0
            for task in env_tasks:
                record = _task_to_record(task, env_key)
                if record:
                    train_records.append(record)
                    env_train_count += 1
            env_split_counts[env_key] = {"train": env_train_count, "eval": 0}
            print(f"  {env_key}: {len(env_tasks)} -> all TRAIN (< {MIN_EVAL_SAMPLES} eval samples)")
            continue

        # Stratified split using hash
        env_train = 0
        env_eval = 0
        for task in env_tasks:
            task_key = task.get("key") or task.get("task_key")
            record = _task_to_record(task, env_key)
            if not record:
                continue

            split = hash_to_split(task_key, eval_ratio)
            if split == "eval":
                eval_records.append(record)
                env_eval += 1
            else:
                train_records.append(record)
                env_train += 1

        env_split_counts[env_key] = {"train": env_train, "eval": env_eval}
        print(f"  {env_key}: {len(env_tasks)} -> {env_train} train, {env_eval} eval")

    print(f"\nTotal: {len(train_records)} train, {len(eval_records)} eval")

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
        default=0.02,
        help="Fraction of data for evaluation (default: 0.02)",
    )
    parser.add_argument(
        "--env-filter",
        type=str,
        default=None,
        help="Optional env_key filter (e.g., 'github', 'booking')",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to include",
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
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
