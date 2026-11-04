"""Score a KeyChain pool with the base model: per prompt, how many of ``--n`` samples are correct.

Samples like the recipes' training (temperature 1.0, top_p 1.0, 12,288-token responses) on one
8-GPU node (vLLM TP8), and scores with the search env's exact match. ``--shard`` / ``--num-shards``
split the pool across nodes (one process per node, then pass every shard's file to
``filter_keychain.py``). A 1,260-prompt train pool plus the 210-prompt val pool take ~20 minutes
per node on 4 nodes of 8xB200.

    uv run --isolated --extra megatron python examples/train/qwen3_8_flash_next/keychain/calibrate_keychain.py \\
        --model-path /path/to/Qwen3.8-Flash-Next --pool $HOME/data/keychain/pool_train.parquet \\
        --out $HOME/data/keychain/calib_train_s0.jsonl
"""

import argparse
import json
import time

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from skyrl_gym.envs.search.utils import compute_score


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pool", required=True, help="pool_<split>.parquet from make_keychain.py")
    parser.add_argument("--out", required=True, help="output .jsonl")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n", type=int, default=4, help="samples per prompt")
    parser.add_argument("--max-tokens", type=int, default=12288, help="the recipes' MAX_RESPONSE_LENGTH")
    parser.add_argument("--max-model-len", type=int, default=81920)
    args = parser.parse_args()

    df = pd.read_parquet(args.pool)
    idx = [i for i in range(len(df)) if i % args.num_shards == args.shard]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=8,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
        language_model_only=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
        max_num_seqs=64,
        generation_config="vllm",
    )
    sampling = SamplingParams(n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=args.shard)
    prompts = [
        tok.apply_chat_template(list(df.iloc[i]["prompt"]), add_generation_prompt=True, tokenize=False) for i in idx
    ]
    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"generated {len(idx)} prompts x {args.n} in {time.time() - t0:.0f} s", flush=True)
    with open(args.out, "w") as f:
        for i, out in zip(idx, outputs):
            row = df.iloc[i]
            target = row["reward_spec"]["ground_truth"]
            completions = out.outputs
            record = {
                "row": i,
                "n": args.n,
                "correct": sum(compute_score(c.text, target) > 0 for c in completions),
                "trunc": sum(c.finish_reason == "length" for c in completions),
                "resp_tokens": [len(c.token_ids) for c in completions],
                "prompt_tokens": len(out.prompt_token_ids),
                "src": row["extra_info"]["src"],
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
