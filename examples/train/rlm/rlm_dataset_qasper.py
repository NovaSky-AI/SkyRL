"""
Create RLM dataset parquets from the alphaXiv/qasper-transformed HuggingFace dataset.

Loads three splits (train, eval, test) and writes:
  train_qasper.parquet
  eval_qasper.parquet
  test_qasper.parquet

Each example:
- prompt: "Find snippets of text that can be used to answer the query: <question>"
- context_text (in extra_info): full paper text (paragraphs joined with double newline)
- reward_spec.evidence: list of ground-truth text spans used to compute F1 reward
  (reward_fn and search/extract_section tools are built at runtime by the generator)

Run:
    uv run -- python examples/train/rlm/rlm_dataset_qasper.py --output_dir ~/data/rlm
    uv run -- python examples/train/rlm/rlm_dataset_qasper.py --output_dir ~/data/rlm --n_eval 200
"""

import argparse
import os

import datasets


HF_DATASET = "alphaXiv/qasper-transformed"


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
    parser.add_argument("--dataset", default=HF_DATASET, help="HuggingFace dataset name")
    parser.add_argument("--n_train", type=int, default=None, help="Cap training examples (default: all)")
    parser.add_argument("--n_eval",  type=int, default=200,   help="Cap eval examples (default: 200)")
    parser.add_argument("--n_test",  type=int, default=None, help="Cap test examples (default: all)")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--min_ctx_chars", type=int, default=0, help="Skip examples with context shorter than this")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    ds_dict = datasets.load_dataset(args.dataset)

    split_map = {
        "train_qasper": ("train", args.n_train),
        "eval_qasper":  ("eval",  args.n_eval),
        "test_qasper":  ("test",  args.n_test),
    }

    output_splits = {}
    for out_name, (hf_split, cap) in split_map.items():
        raw = list(ds_dict[hf_split])

        if args.min_ctx_chars > 0:
            before = len(raw)
            raw = [r for r in raw if len("\n\n".join(r["paragraphs"])) >= args.min_ctx_chars]
            print(f"Filtered {out_name}: {before} -> {len(raw)} rows (min_ctx_chars={args.min_ctx_chars})")

        if cap is not None:
            raw = raw[:cap]

        output_splits[out_name] = datasets.Dataset.from_list([convert(r, args.max_turns) for r in raw])

    n_show = 3
    for split_name, ds in output_splits.items():
        print(f"\nFirst {n_show} {split_name} examples ({len(ds)} total):")
        for i in range(min(n_show, len(ds))):
            ex = ds[i]
            ctx = ex["extra_info"]["context_text"]
            ev = ex["reward_spec"]["evidence"]
            print(f"  [{i}] {ex['prompt'][0]['content'][:100]}")
            print(f"       evidence: {len(ev)} spans, first: {str(ev[0])[:80] if ev else 'none'}...")
            print(f"       context: {len(ctx):,} chars")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, ds in output_splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds.to_parquet(path)

    total = sum(len(ds) for ds in output_splits.values())
    print(f"\nWrote {len(output_splits)} splits ({total} total rows) to {args.output_dir}")


if __name__ == "__main__":
    main()
