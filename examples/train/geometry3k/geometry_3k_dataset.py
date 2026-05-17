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
Adapted from VeRL.
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

    # Process and save the val split
    if "validation" in dataset:
        val_dataset = dataset["validation"]
        print(f"Loaded {len(val_dataset)} validation examples")
        val_dataset = val_dataset.map(
            function=make_map_fn("validation"),
            with_indices=True,
            num_proc=os.cpu_count(),
            remove_columns=val_dataset.column_names,
            desc="Processing val dataset",
        )
        val_parquet_path = os.path.join(output_dir, "val.parquet")
        val_dataset.to_parquet(val_parquet_path)
        print(f"Saved val set ({len(val_dataset)} examples) to {val_parquet_path}")

    # Process and save the test split
    if "test" in dataset:
        test_dataset = dataset["test"]
        print(f"Loaded {len(test_dataset)} test examples")
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
