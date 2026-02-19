"""
Prepare SkyRL-compatible dataset from swe-task-generator task instances.

Reads task.json files from a tasks/ directory and outputs train.parquet + validation.parquet
in the format expected by SkyRL's PromptDataset and our MiniSweAgentGenerator.

Dataset columns (matching mini_swe_agent format):
  - data_source: str
  - prompt: List[Dict[str, str]]  (chat messages)
  - env_class: str  ("null" — custom Docker env)
  - instance: Dict  (full task metadata: instance_id, eval_script, image_name, gold_patch, etc.)

Usage:
  uv run --isolated integrations/swe_tasks/prepare_dataset.py \\
    --tasks-dir /path/to/swe-task-generator/tasks \\
    --output-dir ~/data/swe_tasks
"""

import argparse
import json
import os
import glob

import datasets


def load_tasks(tasks_dir: str) -> list:
    """Load all task.json files from the tasks directory."""
    tasks = []
    task_dirs = sorted(glob.glob(os.path.join(tasks_dir, "task_*")))
    for task_dir in task_dirs:
        task_json = os.path.join(task_dir, "task.json")
        if os.path.exists(task_json):
            with open(task_json) as f:
                task = json.load(f)
            tasks.append(task)
            print(f"  Loaded: {task['instance_id']}")
    return tasks


def make_dataset(tasks: list, data_source: str) -> datasets.Dataset:
    """Convert task list to a HuggingFace Dataset in SkyRL format."""
    records = []
    for task in tasks:
        records.append(
            {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": task["problem_statement"]}],
                "env_class": "null",
                # Store full instance dict — pyarrow will serialize nested dicts
                "instance_id": task["instance_id"],
                "problem_statement": task["problem_statement"],
                "eval_script": task["eval_script"],
                "image_name": task["image_name"],
                "gold_patch": task.get("gold_patch", ""),
                "test_patch": task.get("test_patch", ""),
                "fix_patch": task.get("fix_patch", ""),
                "repo": task.get("repo", ""),
                "base_commit": task.get("base_commit", ""),
            }
        )

    return datasets.Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare SkyRL dataset from swe-task-generator tasks")
    parser.add_argument("--tasks-dir", required=True, help="Path to tasks/ directory")
    parser.add_argument("--output-dir", default="~/data/swe_tasks", help="Output directory for parquet files")
    parser.add_argument("--data-source", default="swe-task-generator", help="Dataset identifier")
    args = parser.parse_args()

    tasks_dir = os.path.abspath(args.tasks_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tasks from {tasks_dir}...")
    tasks = load_tasks(tasks_dir)

    if not tasks:
        print("ERROR: No task instances found!")
        return

    print(f"\nFound {len(tasks)} task instances")

    ds = make_dataset(tasks, args.data_source)

    # For our small dataset, use same set for train and validation
    # NOTE: In production with more tasks, split into train/val
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")

    ds.to_parquet(train_path)
    print(f"  Wrote train: {train_path} ({len(ds)} instances)")

    ds.to_parquet(val_path)
    print(f"  Wrote val:   {val_path} ({len(ds)} instances)")

    print(f"\nDataset ready at {output_dir}/")


if __name__ == "__main__":
    main()
