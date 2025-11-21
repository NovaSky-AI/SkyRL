# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the OpenEnv dataset to parquet format
"""

import argparse
import os
from typing import List
from datasets import Dataset
import math


def example_question(env_name: str) -> List[str]:
    """
    Return simple example prompts for each environment.
    Each prompt instructs the LLM to generate an <action>...</action> response.
    """

    # Shared instruction appended to each example
    instruction = "Wrap the action between <action> and </action> tags.\n" "For example: <action>ACTION_HERE</action>."

    if env_name == "echo_env":
        # Echo environment simply echoes the text back.
        examples = [
            "Send a short test message to the echo environment.",
        ]
    elif env_name == "coding_env":
        # Coding environment executes code and returns stdout.
        examples = [
            "Print 'I love RL!' using Python.",
            "Provide a function fib(n) that can compute the n-th Fibonacci number",
            "Using a for loop, print numbers 1 to 5 and their squares.",
        ]
        instruction = (
            "Write the python code inside <action>...</action> tags. Example: <action>print('Hello, World!')</action>"
        )
    else:
        # NOTE: we only provide dummy train set examples for echo_env and coding_env
        # For other environments, you need to provide your own train / validation set examples.
        raise ValueError(f"Unknown environment name: {env_name}")

    return [f"{ex}\n\n{instruction}" for ex in examples]


def build_dataset(env_name: str, output_dir: str, data_source="openenv", target_size=32):
    """Generate and save train/validation datasets for one environment."""
    base_questions = example_question(env_name)
    repeat_factor = math.ceil(target_size / len(base_questions))
    questions = (base_questions * repeat_factor)[:target_size]

    # Split the questions into train and validation sets
    split_index = target_size // 2
    train_questions = questions[:split_index]
    val_questions = questions[split_index:]

    def make_map_fn(split):
        def process_fn(example, idx):
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": example["question"]}],
                "env_class": "openenv",
                "env_name": env_name,
            }

        return process_fn

    train_dataset = Dataset.from_dict({"question": train_questions})
    val_dataset = Dataset.from_dict({"question": val_questions})

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(make_map_fn("validation"), with_indices=True)

    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(f"Saved {len(train_dataset)} train and {len(val_dataset)} validation examples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/openenv")
    parser.add_argument("--env_name", default=None)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    all_envs = ["echo_env", "coding_env"]

    if args.env_name is None:
        print("No --env_name specified. Preparing datasets for all available environments:")
        for env in all_envs:
            print(f" - {env}")
        print()
        for env in all_envs:
            build_dataset(env, os.path.join(args.output_dir, env))
    else:
        if args.env_name not in all_envs:
            print(f"Error: Unknown environment '{args.env_name}'. Available environments are: {', '.join(all_envs)}")
            exit(1)
        build_dataset(args.env_name, os.path.join(args.output_dir, args.env_name))
