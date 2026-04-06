#!/usr/bin/env python3
"""
Merge a LoRA adapter into its base model and save as a standalone HF model.

Usage:
    python scripts/merge_lora_adapter.py \
        --base-model /path/to/base/model \
        --export-path /path/to/export \
        [--global-step 100]  # optional: defaults to latest

The script looks for the adapter at:
    {export_path}/global_step_{N}/policy/
and saves the merged model to:
    {export_path}/
"""

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="Path or HF name of base model")
    parser.add_argument("--export-path", required=True, help="Directory containing global_step_* subdirs")
    parser.add_argument("--global-step", type=int, default=None, help="Specific step to merge (default: latest)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.export_path):
        raise ValueError(f"export-path {args.export_path} does not exist.")

    if args.global_step is not None:
        global_step = args.global_step
    else:
        steps = [
            int(x.split("_")[-1])
            for x in os.listdir(args.export_path)
            if x.startswith("global_step_")
        ]
        if not steps:
            raise ValueError(f"No global_step_* folders found in {args.export_path}")
        global_step = max(steps)

    adapter_path = os.path.join(args.export_path, f"global_step_{global_step}", "policy")
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path {adapter_path} does not exist.")

    print(f"Merging adapter from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(args.export_path)
    tokenizer.save_pretrained(args.export_path)
    print(f"Merged model saved to: {args.export_path}")


if __name__ == "__main__":
    main()
