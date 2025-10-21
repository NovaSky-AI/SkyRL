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
    instruction = (
        "Wrap the action between <action> and </action> tags.\n"
        "For example: <action>ACTION_HERE</action>."
    )

    if env_name == "echo_env":
        # Echo environment simply echoes the text back.
        examples = [
            "Say hello to the environment.",
            "Send a short test message.",
            "Write one more message to echo back.",
        ]
        return [f"{ex}\n\n{instruction}" for ex in examples]

    elif env_name == "coding_env":
        # Coding environment executes code and returns stdout.
        examples = [
            "Print 'What is that?' using Python.",
            "Define a variable x = 5 + 3 and print its value.",
            "Use a loop to print numbers 1 to 3 and their squares.",
        ]
        instruction.append("Write the python code inside <action>...</action> tags. Example: <action>print('Hello, World!')</action>")
        return [f"{ex}\n\n{instruction}" for ex in examples]

    elif env_name == "openspiel-env":
        # OpenSpiel environment uses action IDs (0, 1, 2...).
        examples = [
            "Choose the first legal action (action id 0).",
        ]
        return [f"{ex}\n\n{instruction}" for ex in examples]

    elif env_name == "atari-env":
        # Atari environment: expect numeric action IDs as above
        examples = [
            "Perform an action with id 0 to start the game.",
        ]
        return [f"{ex}\n\n{instruction}" for ex in examples]

    else:
        raise ValueError(f"Unknown environment name: {env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/openenv")
    parser.add_argument("--env_name", default="echo_env")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(os.path.join(args.output_dir, args.env_name))
    os.makedirs(args.output_dir, exist_ok=True)

    base_questions = example_question(args.env_name)

    # Repeat until 32 examples total
    target_size = 32
    repeat_factor = math.ceil(target_size / len(base_questions))
    questions = (base_questions * repeat_factor)[:target_size]

    # Split evenly into train/validation
    split_index = target_size // 2
    train_questions = questions[:split_index]
    val_questions = questions[split_index:]

    data_source = "openenv"

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["question"]
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": question},
                ],
                "env_class": "openenv",
                "env_name": args.env_name,
            }

        return process_fn

    train_dataset = Dataset.from_dict({"question": train_questions})
    val_dataset = Dataset.from_dict({"question": val_questions})

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(make_map_fn("validation"), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(args.output_dir, "validation.parquet"))

    print(f"Saved {len(train_dataset)} train and {len(val_dataset)} validation examples to {args.output_dir}")