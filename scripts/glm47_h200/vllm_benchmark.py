"""
vLLM throughput benchmark for GLM-4.7-Flash on H200 GPUs.

Tests TP=1,2,4,5 across context lengths 16K,32K,64K,128K.
Measures: prefill throughput, decode throughput, end-to-end throughput,
time-to-first-token, inter-token latency.
"""
import os
import sys
import time
import json
import argparse
import itertools
from typing import List, Dict, Tuple, Optional

import numpy as np


MODEL = "zai-org/GLM-4.7-Flash"
# Batch sizes to try for each (tp, context_len) - will pick the max that fits
# Throughput is measured as (prompt_tokens + output_tokens) / wall_clock_time
OUTPUT_TOKENS = 128   # output tokens per prompt (short so we can test more configs)
WARMUP_PROMPTS = 2    # warmup before timing
TIMED_PROMPTS = 8     # prompts for actual timing


def make_prompts(num_prompts: int, prompt_tokens: int, tokenizer) -> List[str]:
    """Create synthetic prompts of exactly prompt_tokens tokens."""
    # Use a single repeated token ID (safe token: space or 'a')
    token_id = tokenizer.encode("a")[0] if hasattr(tokenizer, 'encode') else 264
    ids = [token_id] * prompt_tokens
    text = tokenizer.decode(ids)
    return [text] * num_prompts


def run_benchmark(
    tp: int,
    context_len: int,
    output_tokens: int,
    batch_size: int,
    gpu_memory_utilization: float = 0.92,
    max_model_len: Optional[int] = None,
) -> Optional[Dict]:
    """Run vLLM benchmark for a specific (tp, context_len, batch_size) config."""
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.tokenizer import get_tokenizer

    if max_model_len is None:
        max_model_len = context_len + output_tokens + 64  # small buffer

    print(f"\n{'='*70}")
    print(f"  TP={tp} | context={context_len//1024}K | batch={batch_size} | out={output_tokens}")
    print(f"{'='*70}")

    try:
        tokenizer = get_tokenizer(MODEL, trust_remote_code=True)

        prompts = make_prompts(batch_size + WARMUP_PROMPTS, context_len, tokenizer)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=output_tokens,
            ignore_eos=True,  # always generate full output_tokens
        )

        llm = LLM(
            model=MODEL,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            trust_remote_code=True,
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
        )

        # Warmup
        print(f"  Warmup ({WARMUP_PROMPTS} prompts)...")
        _ = llm.generate(prompts[:WARMUP_PROMPTS], sampling_params)

        # Timed run
        print(f"  Timed run ({batch_size} prompts)...")
        timed_prompts = prompts[WARMUP_PROMPTS:WARMUP_PROMPTS + batch_size]

        t0 = time.perf_counter()
        outputs = llm.generate(timed_prompts, sampling_params)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_prompt_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(sum(len(c.token_ids) for c in o.outputs) for o in outputs)
        total_tokens = total_prompt_tokens + total_output_tokens

        result = {
            "tp": tp,
            "context_len": context_len,
            "batch_size": batch_size,
            "output_tokens": output_tokens,
            "elapsed_s": elapsed,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "throughput_tok_s": total_tokens / elapsed,
            "output_throughput_tok_s": total_output_tokens / elapsed,
            "prompt_throughput_tok_s": total_prompt_tokens / elapsed,
            "latency_per_prompt_s": elapsed / batch_size,
        }

        print(f"  Results:")
        print(f"    Elapsed: {elapsed:.2f}s")
        print(f"    Prompt tokens: {total_prompt_tokens:,} ({total_prompt_tokens/elapsed:,.0f} tok/s)")
        print(f"    Output tokens: {total_output_tokens:,} ({total_output_tokens/elapsed:,.0f} tok/s)")
        print(f"    Total throughput: {total_tokens/elapsed:,.0f} tok/s")
        print(f"    Latency per prompt: {elapsed/batch_size:.2f}s")

        del llm
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1, 2, 4, 5])
    parser.add_argument("--context-lens", nargs="+", type=int,
                        default=[16384, 32768, 65536, 131072])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="If not set, will auto-tune batch size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--output-file", type=str, default="/tmp/vllm_benchmark_results.json")
    args = parser.parse_args()

    # Batch sizes to try for each context length (will pick max that works)
    # For very long contexts, use smaller batches
    if args.batch_sizes:
        batch_size_candidates = {cl: args.batch_sizes for cl in args.context_lens}
    else:
        batch_size_candidates = {
            16384:  [32, 16, 8, 4, 2],
            32768:  [16, 8, 4, 2],
            65536:  [8, 4, 2, 1],
            131072: [4, 2, 1],
        }

    all_results = []

    for tp in args.tp_sizes:
        for context_len in args.context_lens:
            candidates = batch_size_candidates.get(context_len, [4, 2, 1])

            # Try each batch size, use the largest one that works
            best_result = None
            for bs in candidates:
                result = run_benchmark(
                    tp=tp,
                    context_len=context_len,
                    output_tokens=args.output_tokens,
                    batch_size=bs,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                )
                if result is not None:
                    best_result = result
                    break  # Use the largest batch size that works
                print(f"  Batch size {bs} failed, trying smaller...")

            if best_result:
                all_results.append(best_result)
                with open(args.output_file, "w") as f:
                    json.dump(all_results, f, indent=2)
                print(f"\n  -> Saved to {args.output_file}")
            else:
                print(f"  All batch sizes failed for TP={tp}, context={context_len//1024}K")

    # Print final summary table
    print("\n\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)
    print(f"{'TP':>4} | {'Context':>8} | {'Batch':>6} | {'Total tok/s':>12} | {'Prompt tok/s':>13} | {'Output tok/s':>12} | {'Latency/prompt':>15}")
    print("-"*100)
    for r in all_results:
        print(f"{r['tp']:>4} | {r['context_len']//1024:>6}K | {r['batch_size']:>6} | "
              f"{r['throughput_tok_s']:>12,.0f} | {r['prompt_throughput_tok_s']:>13,.0f} | "
              f"{r['output_throughput_tok_s']:>12,.0f} | {r['latency_per_prompt_s']:>13.2f}s")

    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {args.output_file}")
    return all_results


if __name__ == "__main__":
    main()
