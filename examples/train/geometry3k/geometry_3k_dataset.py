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
Preprocess the Geometry-3K dataset to parquet format for multi-modal RL training.

Dataset source: hiyouga/geometry3k
Fields: 'images' (list of PIL images), 'problem' (text with <image> placeholder), 'answer' (ground truth)
"""

import argparse
import base64
import io
import os

import datasets
from PIL import Image


QUESTION_TEMPLATE = (
    "You are a math/geometry expert. Solve the user's question carefully and verify your work. "
    "Reason step by step as an internal monologue wrapped inside <think>...</think> tags.\n\n"
    "You have access to a tool to check your answer:\n"
    '  <tool_call>{{"name": "calc_score", "arguments": {{"answer": "<your_answer>"}}}}</tool_call>\n\n'
    "Use this tool to verify your solution. If your answer is wrong, you can try again with a different approach.\n"
    r"When you are confident in your final answer, present it as: Answer: \boxed{{$Answer}}"
    "\n\n{Question}"
)


def _pil_to_data_uri(img: Image.Image) -> str:
    """Convert a PIL Image to a base64 data URI string."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _sanitize_text(text: str) -> str:
    """Replace invalid/overlong UTF-8 sequences that crash ujson."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def make_map_fn(split):
    def process_fn(example, idx):
        answer = _sanitize_text(example["answer"].strip())

        problem_text = _sanitize_text(example["problem"])

        content = []

        images = example["images"]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _pil_to_data_uri(img)}})

        # Add the problem text with our question template
        content.append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=problem_text)})

        data = {
            "prompt": [
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "env_class": "geometry3k",
            "reward_spec": {
                "method": "rule",
                "ground_truth": answer,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "problem": problem_text,
                "answer": answer,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/geometry_3k")
    parser.add_argument("--dev_size", type=int, default=500, help="Size of the dev subset")

    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)

    data_source = "hiyouga/geometry3k"

    print(f"Loading dataset from {data_source}...")
    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    print(f"Loaded {len(train_dataset)} training examples")

    # Process the dataset
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=os.cpu_count(),
        remove_columns=train_dataset.column_names,
        desc="Processing dataset",
    )

    # Save the full dataset
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_parquet_path = os.path.join(output_dir, "train.parquet")
    train_dataset.to_parquet(train_parquet_path)
    print(f"Saved full training set ({len(train_dataset)} examples) to {train_parquet_path}")

    # Create a smaller dev subset for testing
    dev_size = min(args.dev_size, len(train_dataset))
    dev_dataset = train_dataset.select(range(dev_size))
    dev_parquet_path = os.path.join(output_dir, "train-dev.parquet")
    dev_dataset.to_parquet(dev_parquet_path)
    print(f"Saved dev subset ({dev_size} examples) to {dev_parquet_path}")

    # If there's a test split, process it too
    if "test" in dataset:
        test_dataset = dataset["test"]
        test_dataset = test_dataset.map(
            function=make_map_fn("test"),
            with_indices=True,
            num_proc=os.cpu_count(),
            remove_columns=test_dataset.column_names,
            desc="Processing test dataset",
        )
        test_parquet_path = os.path.join(output_dir, "test.parquet")
        test_dataset.to_parquet(test_parquet_path)
        print(f"Saved test set ({len(test_dataset)} examples) to {test_parquet_path}")

    print(f"\nDataset preparation complete! Output directory: {output_dir}")

    # Print a sample from the training set
    print("\n" + "=" * 60)
    print("SAMPLE ENTRY (index 0)")
    print("=" * 60)
    sample = train_dataset[0]
    print(f"\n--- Prompt ---")
    for msg in sample["prompt"]:
        print(f"[{msg['role']}]")
        for part in msg["content"]:
            if part["type"] == "text":
                print(part["text"])
            elif part["type"] == "image":
                uri = part["image"]
                print(f"  <image: data URI, {len(uri)} chars>")
    print(f"\n--- Reward Spec ---")
    print(f"  method: {sample['reward_spec']['method']}")
    print(f"  ground_truth: {sample['reward_spec']['ground_truth']}")
    print(f"\n--- Extra Info ---")
    for k, v in sample["extra_info"].items():
        print(f"  {k}: {v}")
