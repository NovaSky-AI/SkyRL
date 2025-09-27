import argparse
import os
from typing import Any, Dict, List

from datasets import Dataset


def build_row(task_path: str, data_source: str) -> Dict[str, Any]:
    task_path_abs = os.path.abspath(task_path)
    return {
        "data_source": data_source,
        "prompt": "dummy",
        "terminal_bench": {
            "task_path": task_path_abs,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet dataset from a terminal_bench directory.")
    parser.add_argument("--task_dir", required=True, help="Directory containing task subdirectories.")
    parser.add_argument("--output_dir", default="~/data/terminal_bench", help="Output directory for Parquet file.")
    parser.add_argument("--output_name", default="train", help="Output name for Parquet file.")

    args = parser.parse_args()

    task_dir = os.path.abspath(os.path.expanduser(args.task_dir))
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Enumerate task subdirectories
    task_dirs: List[str] = []
    try:
        for entry in sorted(os.listdir(task_dir)):
            if entry.startswith("."):
                continue
            full_path = os.path.join(task_dir, entry)
            if os.path.isdir(full_path):
                task_dirs.append(full_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    data_source = "terminal_bench"
    rows = [build_row(task_path=task_dir, data_source=data_source) for task_dir in task_dirs]

    dataset = Dataset.from_list(rows)

    # Save to Parquet
    out_path = os.path.join(output_dir, f"{args.output_name}.parquet")
    dataset.to_parquet(out_path)
