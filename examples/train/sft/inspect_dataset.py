"""
Quick inspection of the alphaXiv/page-labels dataset.

Usage:
    uv run python examples/train/sft/inspect_dataset.py
"""

from datasets import load_dataset
from pprint import pprint

ds = load_dataset("alphaXiv/page-labels")
print("=== Dataset structure ===")
print(ds)
print()

for split_name in ds:
    split = ds[split_name]
    print(f"=== Split: {split_name} ===")
    print(f"  Num rows: {len(split)}")
    print(f"  Features: {split.features}")
    print(f"  Column names: {split.column_names}")
    print()

    print(f"  --- First 3 samples from '{split_name}' ---")
    for i in range(min(3, len(split))):
        print(f"\n  Sample {i}:")
        sample = split[i]
        for key, value in sample.items():
            val_repr = repr(value)
            if len(val_repr) > 300:
                val_repr = val_repr[:300] + "..."
            print(f"    {key}: {val_repr}")
    print()
