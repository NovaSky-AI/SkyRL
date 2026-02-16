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
Preprocess a summarization dataset to parquet format for LLM-as-a-judge training.

Expected input format (JSON/JSONL):
{
    "prompt": "Your task is to summarize...",
    "original_document": "The full document text...",
    "user_intent": {
        "purpose": "...",
        "audience": "...",
        "tone": "...",
        "target_words": 300,
        "focus_areas": "..."
    },
    "sample_id": "unique-id",
    "document_id": "doc-id"
}
"""

import argparse
import json
import os
from typing import List, Dict, Any

import datasets


def load_local_data(input_path: str) -> List[Dict[str, Any]]:
    """Load data from a local JSON or JSONL file."""
    data = []
    
    if input_path.endswith('.jsonl'):
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                data = loaded
            else:
                data = [loaded]
    else:
        raise ValueError(f"Unsupported file format: {input_path}. Use .json or .jsonl")
    
    return data


def make_map_fn(split: str):
    """Create a mapping function to convert raw data to training format."""
    
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # Extract the prompt - this is what the model will see
        prompt_text = example.get("prompt", "")
        
        # If prompt is not provided, construct it from user_intent and original_document
        if not prompt_text:
            user_intent = example.get("user_intent", {})
            original_document = example.get("original_document", "")
            
            prompt_text = f"""Your task is to summarize the user provided document, based on the user intent.

## User Intent
- Purpose: {user_intent.get('purpose', 'N/A')}
- Audience: {user_intent.get('audience', 'N/A')}
- Tone: {user_intent.get('tone', 'neutral')}
- Target Words: {user_intent.get('target_words', 300)}
- Focus Areas: {user_intent.get('focus_areas', 'N/A')}

## Document
{original_document}

Please provide a summary that meets the above requirements."""

        # Ground truth contains the data needed by the grader
        ground_truth = {
            "user_intent": example.get("user_intent", {}),
            "original_document": example.get("original_document", ""),
        }
        
        # Build the training data format
        data = {
            "data_source": "summarization",
            "prompt": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "env_class": "summarization_judge",
            "reward_spec": {
                "method": "llm_judge",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "sample_id": example.get("sample_id", str(idx)),
                "document_id": example.get("document_id", ""),
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert summarization data to parquet format for LLM-as-a-judge training"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON or JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/summarization_judge",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (default: all)",
    )

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    
    # Load the data
    print(f"Loading data from {args.input_file}...")
    raw_data = load_local_data(args.input_file)
    print(f"Loaded {len(raw_data)} samples")
    
    # Optionally limit samples
    if args.max_samples is not None:
        raw_data = raw_data[:args.max_samples]
        print(f"Limited to {len(raw_data)} samples")
    
    # Convert to HuggingFace dataset
    dataset = datasets.Dataset.from_list(raw_data)
    
    # Split into train/val
    split_idx = int(len(dataset) * args.train_split)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(dataset)))
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Apply mapping function
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    # Save to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")
    
    train_dataset.to_parquet(train_path)
    val_dataset.to_parquet(val_path)
    
    print(f"Saved train dataset to {train_path}")
    print(f"Saved validation dataset to {val_path}")
