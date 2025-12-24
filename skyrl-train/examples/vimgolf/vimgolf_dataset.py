"""
Preprocess the dataset for the 'VimGolf' environment in parquet format.
"""

import argparse
import json
import os
import random

from datasets import Dataset

_DATASET_LOADPATH = os.path.dirname(os.path.abspath(__file__)) + "/vimgolf_public_challenges.jsonl"


def retrieve_vimgolf_public_challenges(train_ratio: float):
    """
    Retrieve all 612 VimGolf public challenges.
    Split into train and test sets based on the given ratio.
    """
    with open(_DATASET_LOADPATH, "r") as f:
        all_tasks = f.readlines()
        all_tasks = [json.loads(line) for line in all_tasks if line.strip()]
        random.shuffle(all_tasks)

    train_count = int(len(all_tasks) * train_ratio)
    train_tasks = all_tasks[:train_count]
    test_tasks = all_tasks[train_count:]
    ret = dict(train=train_tasks, test=test_tasks)
    return ret


def create_dataset(data: list[dict], split_name: str):
    """Create a dataset of vimgolf problems."""
    examples = []

    for it in data:
        input_text = it["input"]
        output_text = it["output"]
        challenge_id = it["id"]
        detail = it["metadata"]["detail"]
        example_data = {
            "data_source": "vimgolf_public_challenges",
            "prompt": [],
            "env_class": "vimgolf-single-turn",
            "reward_spec": {
                "method": "vimgolf_evaluator",
                "input_text": input_text,
                "output_text": output_text,
                "detail": detail,
            },
            "extra_info": {
                "challenge_id": challenge_id,
                "split": split_name,
            },
        }
        examples.append(example_data)

    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/VimGolf")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training examples")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Seed for reproducibility
    random.seed(args.random_seed)

    # Generate datasets
    datasets = retrieve_vimgolf_public_challenges(args.train_ratio)
    train_dataset = create_dataset(data=datasets["train"], split_name="train")
    val_dataset = create_dataset(data=datasets["test"], split_name="val")

    # Save datasets
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    train_size = len(datasets["train"])
    test_size = len(datasets["test"])

    print(f"Generated {train_size} training examples and {test_size} test examples")
    print(f"Using {args.random_seed} as random seed")
    print(f"Saved to {output_dir}")
