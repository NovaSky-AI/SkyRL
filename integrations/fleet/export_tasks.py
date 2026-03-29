#!/usr/bin/env python3
"""Export tasks from Fleet API to JSON file.

Usage:
    python -m integrations.fleet.export_tasks --output ~/data/fleet/tasks.json
    python -m integrations.fleet.export_tasks --output ~/data/fleet/tasks.json --env-key github
"""

import argparse
import json
import os
import sys


def export_tasks(output_file: str, env_key: str | None = None, modality: str = "tool_use"):
    """Export tasks from Fleet API to JSON file."""
    try:
        from fleet import Fleet
    except ImportError:
        print("Fleet SDK not available. Install with: pip install fleet-python")
        sys.exit(1)

    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        print("ERROR: FLEET_API_KEY environment variable not set")
        sys.exit(1)

    fleet = Fleet(api_key=api_key)

    print(f"Loading tasks from Fleet API (env_key={env_key})...")
    tasks = fleet.load_tasks(env_key=env_key)
    print(f"Loaded {len(tasks)} tasks")

    # Convert to JSON format
    task_dicts = []
    for task in tasks:
        task_dicts.append(
            {
                "key": task.key,
                "prompt": task.prompt,
                "env_id": task.env_id,
                "version": task.version,
                "data_id": task.data_id,
                "data_version": task.data_version,
                "verifier_func": task.verifier_func,
                "task_modality": modality,
            }
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.expanduser(output_file)), exist_ok=True)

    output_path = os.path.expanduser(output_file)
    with open(output_path, "w") as f:
        json.dump(task_dicts, f, indent=2)

    print(f"Exported {len(task_dicts)} tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Fleet tasks to JSON")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--env-key", default=None, help="Filter by environment key")
    parser.add_argument("--modality", default="tool_use", help="Task modality")
    args = parser.parse_args()

    export_tasks(args.output, args.env_key, args.modality)


if __name__ == "__main__":
    main()
