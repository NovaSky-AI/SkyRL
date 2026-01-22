#!/usr/bin/env python3
"""Prepare Fleet task dataset for SkyRL training."""

import json
import os
import sys
from pathlib import Path

import pandas as pd


def main():
    tasks_file = os.environ.get("TASKS_FILE", "/workspace/skyrl-agent/data/fleet_booking_sample.json")
    output_dir = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/data/fleet/tool_use")))

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(tasks_file) as f:
        tasks = json.load(f)

    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    # Convert to dataframe format expected by skyrl-agent
    records = []
    for task in tasks:
        records.append(
            {
                "instance": task,
                "data_source": "fleet",
            }
        )

    df = pd.DataFrame(records)

    # Split 90/10
    train_size = int(len(df) * 0.9)
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_df.to_parquet(output_dir / "train.parquet")
    val_df.to_parquet(output_dir / "validation.parquet")

    print(f"Created {len(train_df)} train and {len(val_df)} validation samples")


if __name__ == "__main__":
    main()
