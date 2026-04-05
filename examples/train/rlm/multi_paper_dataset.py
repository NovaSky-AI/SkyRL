"""
Create multi-paper RLM dataset parquets from alphaXiv/multi-paper-v1.

Produces three files:
  train.parquet      - training split (cap with --n_train)
  validation.parquet - validation split (cap with --n_val, default 10)
  test.parquet       - test split (cap with --n_test)

Each HF row has one source paper + N questions; we flatten to one row per question.

Each example:
- prompt: "Extract verbatim text passages from the context that serve as evidence for the query: <question>"
- extra_info.context_text: dict {paperId -> "### PAPER: title\\n<abstract>\\ntext"}
- reward_spec.evidence: list of {paperId, selections: [{text}]} (ground-truth spans)
- env_class: "rlm"

Run:
    uv run -- python examples/train/rlm/multi_paper_dataset.py --output_dir ~/data/multi-paper
    uv run -- python examples/train/rlm/multi_paper_dataset.py --output_dir ~/data/multi-paper --n_val 200
"""

import argparse
import json
import os

import datasets


def build_context(papers: list) -> dict:
    ctx = {}
    for paper in papers:
        paper_id = paper["paperId"]
        title = paper["title"]
        abstract = paper.get("abstract", "")
        text = paper["text"]
        abstract_block = f"<abstract>\n{abstract}\n</abstract>\n" if abstract else ""
        ctx[paper_id] = f"### PAPER: {title}\n{abstract_block}{text}"
    return ctx


def flatten_datapoint(datapoint: dict, max_turns: int) -> list:
    ctx = build_context(datapoint["papers"])
    rows = []
    for question in datapoint["questions"]:
        q_text = question["question"]
        rows.append({
            "prompt": [{"role": "user", "content": (
                f"Extract verbatim text passages from the context that serve as evidence for the query: {q_text}\n"
                f"Return a Python list of exact substrings copied from the context. No paraphrasing, no commentary."
            )}],
            "env_class": "rlm",
            "reward_spec": {
                "ground_truth": None,
                "evidence": question["evidence"],
            },
            "max_turns": max_turns,
            "extra_info": {
                "context_text": json.dumps(ctx),
            },
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/multi-paper")
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val",   type=int, default=10)
    parser.add_argument("--n_test",  type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--load_only_train", action="store_true",
                        help="Load only the train split and derive val/test from it")
    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)

    hf_ds = datasets.load_dataset("alphaXiv/multi-paper-v1")

    if args.load_only_train:
        split_map = {
            "train": ("train", args.n_train),
            "validation": ("train", args.n_val),
            "test": ("train", args.n_test),
        }
    else:
        split_map = {
            "train": ("train", args.n_train),
            "validation": ("validation" if "validation" in hf_ds else "train", args.n_val),
            "test": ("test" if "test" in hf_ds else "train", args.n_test),
        }

    splits = {}
    for out_name, (hf_split, cap) in split_map.items():
        raw = list(hf_ds[hf_split])
        rows = []
        for dp in raw:
            rows.extend(flatten_datapoint(dp, args.max_turns))
            if cap is not None and len(rows) >= cap:
                rows = rows[:cap]
                break
        splits[out_name] = datasets.Dataset.from_list(rows)

    n_show = 3
    for split_name, ds in splits.items():
        print(f"\nFirst {n_show} {split_name} examples ({len(ds)} total):")
        for i in range(min(n_show, len(ds))):
            ex = ds[i]
            ctx = json.loads(ex["extra_info"]["context_text"])
            ev = ex["reward_spec"]["evidence"]
            n_papers = len(ctx)
            print(f"  [{i}] {ex['prompt'][0]['content'][:100]}")
            print(f"       evidence: {len(ev)} spans")
            print(f"       papers: {n_papers}, ids: {list(ctx.keys())[:3]}")

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, ds in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.parquet")
        ds.to_parquet(path)

    total = sum(len(ds) for ds in splits.values())
    print(f"\nWrote {len(splits)} splits ({total} total rows) to {args.output_dir}")


if __name__ == "__main__":
    main()
