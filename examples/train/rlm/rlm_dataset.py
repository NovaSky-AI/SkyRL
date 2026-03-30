"""
Create RLM dataset parquets from the three QASPER splits.

Produces three files:
  train.parquet      - full qasper-train-cleaned.json  (cap with --n_train)
  validation.parquet - first --n_val rows of qasper-validation-cleaned.json
                       (default 1; single example used to track rollout latency during training)
  test.parquet       - full qasper-test-cleaned.json   (cap with --n_test)

Each example:
- prompt: "Find snippets of text that can be used to answer the query: <question>"
- context_text (in extra_info): full paper text (paragraphs joined with double newline)
- reward_spec.evidence: list of ground-truth text spans used to compute F1 reward
  (reward_fn and search/extract_section tools are built at runtime by the generator)

Run:
    uv run -- python examples/train/rlm/rlm_dataset.py --output_dir ~/data/rlm
    uv run -- python examples/train/rlm/rlm_dataset.py --output_dir ~/data/rlm --n_val 200
"""

import argparse
import json
import os
import sys

import datasets

# Resolve data paths relative to this file so it works from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DATA_DIR = os.path.join(_REPO_ROOT, "data")


def load_json(path: str) -> list:
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def convert(row: dict, max_turns: int) -> dict:
    ctx = "\n\n".join(row["paragraphs"])
    return {
        "prompt": [{"role": "user", "content": f"Find snippets of text that can be used to answer the query: {row['question']}"}],
        "env_class": "rlm",
        "reward_spec": {
            "ground_truth": None,
            "evidence": row["evidence"],
        },
        "max_turns": max_turns,
        "extra_info": {
            "context_text": ctx,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/rlm")
    parser.add_argument("--train_data_path", default=os.path.join(_DATA_DIR, "qasper-train-cleaned.json"))
    parser.add_argument("--val_data_path",   default=os.path.join(_DATA_DIR, "qasper-validation-cleaned.json"))
    parser.add_argument("--test_data_path",  default=os.path.join(_DATA_DIR, "qasper-test-cleaned.json"))
    parser.add_argument("--n_train", type=int, default=None, help="Cap training examples (default: all)")
    parser.add_argument("--n_val",   type=int, default=10,   help="Cap validation examples (default: 10)")
    parser.add_argument("--n_test",  type=int, default=None, help="Cap test examples (default: all)")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--min_ctx_chars", type=int, default=0, help="Skip examples with context shorter than this")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    train_raw = load_json(args.train_data_path)
    val_raw   = load_json(args.val_data_path)
    test_raw  = load_json(args.test_data_path)

    if args.min_ctx_chars > 0:
        for name, raw in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
            before = len(raw)
            raw[:] = [r for r in raw if len("\n\n".join(r["paragraphs"])) >= args.min_ctx_chars]
            print(f"Filtered {name}: {before} -> {len(raw)} rows (min_ctx_chars={args.min_ctx_chars})")

    if args.n_train is not None:
        train_raw = train_raw[:args.n_train]
    if args.n_val is not None:
        val_raw = val_raw[:args.n_val]
    if args.n_test is not None:
        test_raw = test_raw[:args.n_test]

    splits = {
        "train":      datasets.Dataset.from_list([convert(r, args.max_turns) for r in train_raw]),
        "validation": datasets.Dataset.from_list([convert(r, args.max_turns) for r in val_raw]),
        "test":       datasets.Dataset.from_list([convert(r, args.max_turns) for r in test_raw]),
    }

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
