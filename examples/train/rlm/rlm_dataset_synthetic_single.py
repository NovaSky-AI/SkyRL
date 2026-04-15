"""
Create RLM dataset parquets from the alphaXiv/single-paper-synthetic HuggingFace dataset.

The HF dataset only has a train split, so this script shuffles with a fixed seed and
splits it into train/validation/test itself.

Produces three files:
  train.parquet      - majority of rows (after removing val + test)
  validation.parquet - first --n_val rows of the shuffled dataset (default: 10)
  test.parquet       - next --n_test rows after the val slice (default: 128)

Each example:
- prompt: "Find snippets of text that can be used to answer the query: <question>"
- context_text (in extra_info): full paper text
- reward_spec.evidence: list of ground-truth text spans used to compute F1 reward

Run:
    uv run -- python examples/train/rlm/rlm_dataset_synthetic_single.py --output_dir ~/data/rlm-synthetic
    uv run -- python examples/train/rlm/rlm_dataset_synthetic_single.py --output_dir ~/data/rlm-synthetic --n_val 200 --no_test
"""

import argparse
import json
import os
import random

import datasets


_HF_DATASET = "alphaXiv/single-paper-synthetic"


def convert(row: dict, max_turns: int) -> dict:
    evidence_raw = json.loads(row["evidence"])
    evidence = [item["text"] for item in evidence_raw]
    return {
        "prompt": [{"role": "user", "content": f"Find snippets of text that can be used to answer the query: {row['question']}"}],
        "env_class": "rlm",
        "reward_spec": {
            "ground_truth": None,
            "evidence": evidence,
        },
        "max_turns": max_turns,
        "extra_info": {
            "context_text": row["paper_text"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/rlm-synthetic")
    parser.add_argument("--hf_dataset", default=_HF_DATASET)
    parser.add_argument("--n_val",   type=int, default=10,  help="Validation set size (default: 10)")
    parser.add_argument("--n_test",  type=int, default=128, help="Test set size (default: 128)")
    parser.add_argument("--no_test", action="store_true", help="Skip allocating examples for the test split")
    parser.add_argument("--n_train", type=int, default=None, help="Cap training examples (default: all remaining)")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_ctx_chars", type=int, default=0, help="Skip examples with context shorter than this")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    print(f"Loading {args.hf_dataset} ...")
    raw_ds = datasets.load_dataset(args.hf_dataset, split="train")
    rows = list(raw_ds)

    if args.min_ctx_chars > 0:
        before = len(rows)
        rows = [r for r in rows if len(r["paper_text"]) >= args.min_ctx_chars]
        print(f"Filtered: {before} -> {len(rows)} rows (min_ctx_chars={args.min_ctx_chars})")

    before = len(rows)
    rows = [r for r in rows if json.loads(r["evidence"])]
    print(f"Filtered: {before} -> {len(rows)} rows (removed empty evidence)")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n_val = args.n_val
    n_test = 0 if args.no_test else args.n_test
    val_raw   = rows[:n_val]
    test_raw  = rows[n_val:n_val + n_test]
    train_raw = rows[n_val + n_test:]

    if args.n_train is not None:
        train_raw = train_raw[:args.n_train]

    print(f"Split: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    splits = {
        "train":      datasets.Dataset.from_list([convert(r, args.max_turns) for r in train_raw]),
        "validation": datasets.Dataset.from_list([convert(r, args.max_turns) for r in val_raw]),
    }
    if not args.no_test:
        splits["test"] = datasets.Dataset.from_list([convert(r, args.max_turns) for r in test_raw])

    n_show = 3
    for split_name, ds in splits.items():
        print(f"\nFirst {n_show} {split_name} examples ({len(ds)} total):")
        for i in range(min(n_show, len(ds))):
            ex = ds[i]
            ctx = ex["extra_info"]["context_text"]
            ev = ex["reward_spec"]["evidence"]
            print(f"  [{i}] {ex['prompt'][0]['content'][:100]}")
            print(f"       evidence: {len(ev)} spans, first: {str(ev[0])[:80] if ev else 'none'}...")
            print(f"       context: {len(ctx):,} chars")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, ds in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds.to_parquet(path)

    total = sum(len(ds) for ds in splits.values())
    print(f"\nWrote {len(splits)} splits ({total} total rows) to {args.output_dir}")


if __name__ == "__main__":
    main()
