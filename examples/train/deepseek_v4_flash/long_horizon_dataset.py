"""GSM8k as long-horizon multi-turn agent rollouts (env ``gsm8k_long_horizon``), for throughput runs shaped like a
long tool-use agent: each rollout is ``--max-turns`` turns of model output followed by an ``--obs-tokens``-token
synthetic tool observation, then a scored final answer.

    python examples/train/deepseek_v4_flash/long_horizon_dataset.py --output_dir ~/data/gsm8k_long_horizon \
        --max-turns 16 --obs-tokens 1870
"""

import argparse
import os
import re

import datasets


def extract_solution(solution_str: str) -> str:
    m = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert m is not None
    return m.group(0).split("#### ")[1].replace(",", "")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="~/data/gsm8k_long_horizon")
    p.add_argument("--max-turns", type=int, default=16)
    p.add_argument("--obs-tokens", type=int, default=1870)
    args = p.parse_args()
    out = os.path.expanduser(args.output_dir)

    ds = datasets.load_dataset("openai/gsm8k", "main")
    instruction = 'Let\'s think step by step and output a tentative numeric answer after "####".'

    def make_map_fn(split):
        def fn(example, idx):
            return {
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": example["question"] + " " + instruction}],
                "env_class": "gsm8k_long_horizon",
                "reward_spec": {"method": "rule", "ground_truth": extract_solution(example["answer"])},
                "extra_info": {
                    "max_turns": args.max_turns,
                    "obs_tokens": args.obs_tokens,
                    "seed": idx,
                    "split": split,
                    "index": idx,
                },
            }

        return fn

    for split, name in (("train", "train"), ("test", "validation")):
        d = ds[split].map(make_map_fn(name), with_indices=True, remove_columns=ds[split].column_names)
        d.to_parquet(os.path.join(out, f"{name}.parquet"))
    print(f"wrote {out} (max_turns={args.max_turns}, obs_tokens={args.obs_tokens})")
