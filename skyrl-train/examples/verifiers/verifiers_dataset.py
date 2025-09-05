import argparse
import os
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

    if info not in [None, {}]:
        full_sample["verifiers"]["info"] = info

    return full_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Parquet dataset from a verifiers environment.")
    parser.add_argument("--env_id", default="wordle", help="Environment identifier to load (e.g., 'wordle').")
    parser.add_argument("--output_dir", default="~/data/verifiers/wordle", help="Output directory for Parquet files.")
    parser.add_argument("--num_train", type=int, default=-1, help="Number of training examples to generate. -1 for no limit.")
    parser.add_argument("--num_eval", type=int, default=-1, help="Number of evaluation examples to generate. -1 for no limit.")

    args = parser.parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load verifiers environment
    env_name = extract_env_name(args.env_id)
    vf_env = load_environment(env_id=args.env_id)
    
    # Get the datasets
    train_ds = vf_env.get_dataset(args.num_train)
    eval_ds = vf_env.get_eval_dataset(args.num_eval)
    data_source = f"verifiers/{env_name}"

    # Convert to SkyRL format
    train_ds = train_ds.map(
        lambda sample, idx: build_row(sample, data_source=data_source, env_name=env_name),
        with_indices=True,
    )
    eval_ds = eval_ds.map(
        lambda sample, idx: build_row(sample, data_source=data_source, env_name=env_name),
        with_indices=True,
    )

    # TODO(tgriggs): Reconsider this. Can we not use parquet?
    # Drop top-level 'info' column, which often defaults to empty dict.
    train_ds = train_ds.remove_columns([c for c in ["info"] if c in train_ds.column_names])
    eval_ds = eval_ds.remove_columns([c for c in ["info"] if c in eval_ds.column_names])

    # Save to Parquet
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "validation.parquet")
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(val_path)