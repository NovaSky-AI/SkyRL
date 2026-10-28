"""Sample DAPO prompts from a DeepSeek-V4.1 checkpoint with vLLM and report sampled tokens whose logprob is
non-finite or at vLLM's -9999 floor (what SkyRL's rollout-vs-train logprob metric spikes on).

Usage (one 8-GPU node, no weight sync involved):
  python probe_vllm_logprobs_v41.py <model_dir> <dapo.parquet> <out.jsonl> [--prompts 128] [--n 12] [--max-tokens 2048]

--poison-kv fills every KV cache tensor with 0xFF bytes (NaN in fp8 e4m3 and bf16) before sampling: if the rate
jumps, attention reads slots that were never written (after a sleep/wake the cache holds whatever training left).
"""

import argparse
import json
import math

import pandas as pd
from vllm import LLM, SamplingParams


def _poison_kv_cache(worker) -> int:
    """Fill this worker's KV cache tensors with 0xFF bytes; returns how many tensors were filled."""
    import torch

    runner = getattr(worker, "model_runner", worker)
    caches = getattr(runner, "kv_caches", None)
    seen, count = set(), 0

    def fill(obj):
        nonlocal count
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.data_ptr() not in seen:
                seen.add(obj.data_ptr())
                obj.view(torch.uint8).fill_(255)
                count += 1
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                fill(item)
        elif isinstance(obj, dict):
            for item in obj.values():
                fill(item)

    fill(caches)
    torch.cuda.synchronize()
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--prompts", type=int, default=128)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--poison-kv", action="store_true")
    args = p.parse_args()

    df = pd.read_parquet(args.data).head(args.prompts)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=2048 + args.max_tokens + 512,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend="FLASHMLA_SPARSE_DSV41",
        logprobs_mode="processed_logprobs",
        enable_prefix_caching=False,
        seed=args.seed,
    )
    if args.poison_kv:
        print("PROBE poisoned KV tensors per rank:", llm.collective_rpc(_poison_kv_cache), flush=True)
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
        for _, row in df.iterrows()
    ]
    params = SamplingParams(
        n=args.n, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=1, seed=args.seed
    )
    outputs = llm.generate(prompts, params)

    n_tokens = n_bad = 0
    with open(args.out, "w") as f:
        for pi, out in enumerate(outputs):
            for si, comp in enumerate(out.outputs):
                for pos, (tid, lp_dict) in enumerate(zip(comp.token_ids, comp.logprobs)):
                    n_tokens += 1
                    entry = lp_dict.get(tid) if lp_dict else None
                    lp = entry.logprob if entry is not None else None
                    if lp is None or not math.isfinite(lp) or lp <= -9999.0:
                        n_bad += 1
                        top = {int(k): (v.logprob, v.decoded_token) for k, v in (lp_dict or {}).items()}
                        f.write(
                            json.dumps(
                                {
                                    "prompt": pi,
                                    "sample": si,
                                    "pos": pos,
                                    "prompt_len": len(out.prompt_token_ids),
                                    "token": tid,
                                    "text": tok.decode([tid]),
                                    "logprob": None if lp is None else repr(lp),
                                    "top": {k: (repr(v[0]), v[1]) for k, v in top.items()},
                                    "len": len(comp.token_ids),
                                    "finish": comp.finish_reason,
                                }
                            )
                            + "\n"
                        )
    print(f"PROBE tokens={n_tokens} bad={n_bad} rate={n_bad / max(1, n_tokens):.2e}", flush=True)


if __name__ == "__main__":
    main()
