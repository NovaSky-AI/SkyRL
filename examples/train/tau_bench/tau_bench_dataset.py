"""Build parquet datasets for the tau-bench retail environment.

Each row corresponds to one retail task. The (large, static) retail policy/wiki and
tool schemas are NOT stored per row — ``TauBenchEnv.init`` builds the system prompt
from the vendored domain data and obtains the opening user message from the user
simulator. So the dataset only needs to identify the task and (for record-keeping)
the gold trajectory.

Usage:
    uv run --isolated python examples/train/tau_bench/tau_bench_dataset.py \
        --output_dir /mnt/cluster_storage/data/tau_bench
"""

import argparse
import os

import pandas as pd

from skyrl_gym.envs.tau_bench.tau_core.retail.tasks_test import TASKS_TEST
from skyrl_gym.envs.tau_bench.tau_core.retail.tasks_train import TASKS_TRAIN


def build_rows(tasks, split: str):
    rows = []
    for task_index, task in enumerate(tasks):
        reward_spec = {
            "method": "rule",
            "ground_truth": {
                "actions": [{"name": a.name, "kwargs": a.kwargs} for a in task.actions],
                "outputs": list(task.outputs),
            },
        }
        rows.append(
            {
                "data_source": "tau_bench_retail",
                # Placeholder only — the real prompt (system wiki + dynamic opening user
                # message) is constructed in TauBenchEnv.init, which overrides this. It must
                # NOT contain the hidden user instruction. A user turn is required so the
                # chat template applies cleanly at dataset load (Qwen errors on system-only).
                "prompt": [
                    {"role": "system", "content": "You are a retail customer service agent."},
                    {"role": "user", "content": "Hi."},
                ],
                "env_class": "tau_bench",
                "reward_spec": reward_spec,
                "task_index": task_index,
                "task_split": split,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.path.expanduser("~/data/tau_bench"))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    val_rows = build_rows(TASKS_TEST, "test")
    train_rows = build_rows(TASKS_TRAIN, "train")

    val_path = os.path.join(args.output_dir, "retail_test.parquet")
    train_path = os.path.join(args.output_dir, "retail_train.parquet")
    pd.DataFrame(val_rows).to_parquet(val_path)
    pd.DataFrame(train_rows).to_parquet(train_path)

    print(f"Wrote {len(val_rows)} eval tasks  -> {val_path}")
    print(f"Wrote {len(train_rows)} train tasks -> {train_path}")


if __name__ == "__main__":
    main()
