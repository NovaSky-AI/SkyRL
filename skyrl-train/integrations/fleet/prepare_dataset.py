"""
Prepare Fleet tasks for SkyRL training.

Converts Fleet task JSON files to SkyRL parquet dataset format.

Usage:
    python -m integrations.fleet.prepare_dataset \
        --tasks-json /path/to/all_tool_use.json \
        --output-dir ./data/fleet \
        --modality tool_use
"""

import argparse
import json
import os
from typing import List, Dict, Any, Optional

from datasets import Dataset


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


def prepare_fleet_dataset(
    tasks_json: str,
    output_dir: str,
    modality: Optional[str] = "tool_use",
    train_split: float = 0.9,
    env_filter: Optional[str] = None,
    max_tasks: Optional[int] = None,
):
    """
    Convert Fleet tasks JSON to SkyRL parquet dataset.

    Args:
        tasks_json: Path to Fleet tasks JSON file
        output_dir: Output directory for parquet files
        modality: Task modality filter ("tool_use" or "computer_use"), None for all
        train_split: Fraction of data for training (rest is validation)
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

    # Convert to SkyRL dataset format
    records = []
    for task in tasks:
        task_key = task.get("key") or task.get("task_key")
        prompt = task.get("prompt", "")

        if not task_key or not prompt:
            print(f"Skipping task with missing key or prompt: {task.get('key', 'unknown')}")
            continue

        records.append(
            {
                # Required fields for SkyRL
                "prompt": [{"role": "user", "content": prompt}],
                "env_class": "fleet_task",  # This tells SkyRL to use FleetTaskEnv
                # Task identification (passed as env_extras)
                "task_key": task_key,
            }
        )

    print(f"Prepared {len(records)} records for dataset")

    # Shuffle records for better train/val split
    import random

    random.seed(42)
    random.shuffle(records)

    # Split into train/validation
    split_idx = int(len(records) * train_split)
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    print(f"Train: {len(train_records)}, Validation: {len(val_records)}")

    # Create datasets
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)

    print(f"Saved train dataset to {train_path}")
    print(f"Saved validation dataset to {val_path}")

    # Print summary statistics
    print("\n=== Dataset Summary ===")

    # Count tasks by env
    env_counts = {}
    for task in tasks:
        env_key = task.get("env_key") or task.get("env_id") or "unknown"
        env_counts[env_key] = env_counts.get(env_key, 0) + 1

    print("Tasks by environment:")
    for env_key, count in sorted(env_counts.items(), key=lambda x: -x[1]):
        print(f"  {env_key}: {count}")


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
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (default: 0.9)",
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
        train_split=args.train_split,
        env_filter=args.env_filter,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
