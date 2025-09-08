import argparse
import os
from functools import partial
from typing import Any, Dict

from verifiers import load_environment


def extract_env_name(env_id: str) -> str:
    """Return only the environment name from strings like 'org/name@version' or 'name@version'."""
    base = env_id.split("/")[-1]
    return base.split("@")[0]


def build_row(sample: Dict[str, Any], data_source: str, env_name: str) -> Dict[str, Any]:
    if "prompt" not in sample:
        raise ValueError("Example must contain a 'prompt' field")
    prompt = sample["prompt"]  # Already formatted by the environment as chat messages

    answer = sample.get("answer", "")
    info = sample.get("info", None)
    task = sample.get("task", "default")

    full_sample = {
        "data_source": data_source,
        "prompt": prompt,
        "verifiers": {
            "answer": answer,
            "task": task,
            "environment": env_name,
        },
    }

    if info:
        full_sample["verifiers"]["info"] = info

    return full_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet dataset from a verifiers environment.")
    parser.add_argument("--env_id", default="wordle", help="Environment identifier to load (e.g., 'wordle').")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory for Parquet files. Defaults to ~/data/{env_id} in-code."
    )
    parser.add_argument(
        "--num_train", type=int, default=-1, help="Number of training examples to generate. -1 for no limit."
    )
    parser.add_argument(
        "--num_eval", type=int, default=-1, help="Number of evaluation examples to generate. -1 for no limit."
    )

    args = parser.parse_args()

    # Resolve output directory
    output_dir_name = args.output_dir if args.output_dir else f"~/data/{args.env_id}"
    output_dir = os.path.expanduser(output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load verifiers environment
    env_name = extract_env_name(args.env_id)
    vf_env = load_environment(env_id=env_name)
    data_source = f"verifiers/{env_name}"
    map_fn = partial(build_row, data_source=data_source, env_name=env_name)

    # Load train dataset
    try:
        train_ds = vf_env.get_dataset(args.num_train)
    except ValueError:
        train_ds = None
        print(f"WARNING: Environment {args.env_id} does not have a training dataset. Loading the eval dataset only.")
    if train_ds:
        train_ds = train_ds.map(map_fn, num_proc=16)
        # Drop top-level 'info' column, which often defaults to empty dict and cannot be serialized to parquet.
        if "info" in train_ds.column_names:
            train_ds = train_ds.remove_columns("info")
        train_path = os.path.join(output_dir, "train.parquet")
        train_ds.to_parquet(train_path)

    # Load eval dataset
    eval_ds = vf_env.get_eval_dataset(args.num_eval)
    eval_ds = eval_ds.map(map_fn, num_proc=16)
    # Drop top-level 'info' column, which often defaults to empty dict and cannot be serialized to parquet.
    if "info" in eval_ds.column_names:
        eval_ds = eval_ds.remove_columns("info")
    val_path = os.path.join(output_dir, "validation.parquet")
    eval_ds.to_parquet(val_path)
