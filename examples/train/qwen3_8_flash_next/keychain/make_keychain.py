"""KeyChain-style long-context multi-hop QA (after LoongRL, arXiv 2510.19363) for SkyRL's single-turn
``search`` env.

Seed questions come from HotpotQA (distractor), MuSiQue (answerable only) and 2WikiMultihopQA;
yes/no answers are dropped because they can be guessed. Each prompt holds:

- the seed's own paragraphs (supporting and distractor) plus padding paragraphs from other seeds,
  shuffled, up to ``--tokens`` tokens;
- a true chain of 3-5 keys whose last key holds the seed's question, and 4 decoy chains of the same
  length that end in other seeds' questions, with the links scattered among the documents;
- an instruction to start from the true chain's first key, follow it, and answer the question it
  leads to inside <answer></answer>.

Rows set ``env_class=search`` and ``max_turns=1``: the search env scores the answer by exact match
(SQuAD-normalized) against the answer and its aliases, and makes no retrieval calls.

    uv run --isolated --extra megatron python examples/train/qwen3_8_flash_next/keychain/make_keychain.py \\
        --model-path /path/to/Qwen3.8-Flash-Next --out-dir $HOME/data/keychain

writes ``pool_train.parquet`` (seeds from the train splits) and ``pool_val.parquet`` (validation
splits). The defaults (62,000-token documents, ~65.5K-token prompts) are what the 262K CP8 recipes
were tested with. ``calibrate_keychain.py`` and ``filter_keychain.py`` then keep the prompts the
base model sometimes, but not always, answers correctly.
"""

import argparse
import os
import random

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "hotpot": ("hotpotqa/hotpot_qa", "distractor"),
    "musique": ("dgslibisey/MuSiQue", None),
    "2wiki": ("framolfese/2WikiMultihopQA", None),
}

INSTRUCTION = (
    "\n\nThe documents above contain several chains of keys. Each key entry has the form "
    '"[Key X] The next key is Y.", and the last key of each chain holds a question, in the form '
    '"[Key X] Question: ...".\n'
    "Start from key {k0}. Follow its chain to the end to find YOUR question; questions at the end of "
    "other chains are not yours. Then answer your question using the documents. Think it through, "
    "then give only the final answer, as a short phrase, inside <answer> and </answer>."
)


def load_seeds(src: str, split: str, n: int) -> list:
    """Up to ``n`` seeds: dict(question, answers, paragraphs=[(title, text)], src, id)."""
    name, config = SOURCES[src]
    ds = load_dataset(name, config, split=split, streaming=True)
    seeds = []
    for ex in ds:
        if src == "musique":
            if not ex["answerable"]:
                continue
            paragraphs = [(p["title"], p["paragraph_text"]) for p in ex["paragraphs"]]
            answers = [ex["answer"]] + list(ex["answer_aliases"] or [])
        else:
            c = ex["context"]
            paragraphs = [(t, "".join(s)) for t, s in zip(c["title"], c["sentences"])]
            answers = [ex["answer"]]
        if not answers[0].strip() or answers[0].strip().lower() in ("yes", "no"):
            continue
        seeds.append(
            {"question": ex["question"], "answers": answers, "paragraphs": paragraphs, "src": src, "id": ex["id"]}
        )
        if len(seeds) >= n:
            break
    return seeds


def build(seeds, padding, rng, tok, tokens: int, split: str) -> list:
    def new_key():
        return "%032X" % rng.getrandbits(128)

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = []
    for i, seed in enumerate(seeds):
        others = [o for o in seeds if o is not seed]
        chain_len = rng.randint(3, 5)
        chains = [(seed["question"], True)] + [(o["question"], False) for o in rng.sample(others, 4)]
        links, start = [], None
        for question, true in chains:
            keys = [new_key() for _ in range(chain_len)]
            if true:
                start = keys[0]
            links += [f"[Key {a}] The next key is {b}." for a, b in zip(keys[:-1], keys[1:])]
            links.append(f"[Key {keys[-1]}] Question: {question}")

        own_titles = {t for t, _ in seed["paragraphs"]}
        docs = [f"Title: {t}\n{x}" for t, x in seed["paragraphs"]]
        used = sum(count(d) for d in docs) + 60 * len(links) + 200
        order = list(range(len(padding)))
        rng.shuffle(order)
        for j in order:
            title, text, n = padding[j]
            if title in own_titles:
                continue
            if used + n + 8 > tokens - 300:
                if used > tokens - 2000:
                    break
                continue
            docs.append(f"Title: {title}\n{text}")
            used += n + 8
        rng.shuffle(docs)
        for link in links:
            docs.insert(rng.randint(0, len(docs)), link)

        body = "\n\n".join(f"Document {k + 1}:\n{d}" for k, d in enumerate(docs))
        rows.append(
            {
                "data_source": f"keychain_{seed['src']}",
                "prompt": [{"role": "user", "content": body + INSTRUCTION.format(k0=start)}],
                "env_class": "search",
                "reward_spec": {"method": "rule", "ground_truth": {"target": seed["answers"]}},
                "max_turns": 1,
                "extra_info": {
                    "index": i,
                    "src": seed["src"],
                    "seed_id": seed["id"],
                    "chain_len": chain_len,
                    "split": split,
                },
            }
        )
        if i < 2 or i % 200 == 0:
            text = tok.apply_chat_template(rows[-1]["prompt"], add_generation_prompt=True, tokenize=False)
            print(f"[{split}] prompt {i}: {count(text)} tokens, {seed['src']}, chain {chain_len}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="tokenizer used to size the prompts")
    parser.add_argument("--out-dir", default=os.path.expanduser("~/data/keychain"))
    parser.add_argument("--tokens", type=int, default=62000, help="document tokens per prompt")
    parser.add_argument("--n-train-per-source", type=int, default=420)
    parser.add_argument("--n-val-per-source", type=int, default=70)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    rng = random.Random(args.seed)
    for split, hf_split, n in (
        ("train", "train", args.n_train_per_source),
        ("val", "validation", args.n_val_per_source),
    ):
        seeds = []
        for src in SOURCES:
            got = load_seeds(src, hf_split, n)
            print(f"[{split}] loaded {len(got)} {src} seeds ({hf_split})", flush=True)
            seeds += got
        rng.shuffle(seeds)
        # Padding: the paragraphs of every seed of this split (a prompt skips its own titles).
        seen, pool = set(), []
        for seed in seeds:
            for title, text in seed["paragraphs"]:
                if title not in seen:
                    seen.add(title)
                    pool.append((title, text))
        counts = [len(ids) for ids in tok([x for _, x in pool], add_special_tokens=False)["input_ids"]]
        padding = [(t, x, c) for (t, x), c in zip(pool, counts)]
        rows = build(seeds, padding, rng, tok, args.tokens, split)
        path = os.path.join(args.out_dir, f"pool_{split}.parquet")
        pd.DataFrame(rows).to_parquet(path)
        print(f"[{split}] wrote {len(rows)} prompts to {path}", flush=True)


if __name__ == "__main__":
    main()
