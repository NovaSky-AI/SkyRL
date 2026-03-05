"""
Optimization benchmark for GLM-4.7-Flash on H200.
Tests various vLLM knobs vs baseline TP=4.

Optimizations tested:
1. FP8 KV cache (kv_cache_dtype=fp8_e5m2) - 2x KV memory reduction
2. Prefill context parallelism (prefill_context_parallel_size=2) - splits seq across 8 GPUs
3. FP8 + context parallelism combined
4. Block size 32 (vs default 16)
5. FlashInfer attention backend

Focus on TP=4, context lengths 32K, 64K, 128K.
"""
import os
import sys
import json
import time
import textwrap
import tempfile
import subprocess

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

MODEL = "zai-org/GLM-4.7-Flash"
SKYRL_VENV = "/home/ray/SkyRL/.venv/bin/python"
RESULTS_FILE = "/tmp/vllm_opt_benchmark_results.json"
GPU_MEM_UTIL = 0.92
OUTPUT_TOKENS = 256


BENCH_SCRIPT = textwrap.dedent(r"""
import os, sys, time, json, gc
import torch

MODEL = "{model}"
TP = {tp}
CONTEXT_LEN = {context_len}
OUTPUT_TOKENS = {output_tokens}
GPU_MEM_UTIL = {gpu_mem_util}
N_PROMPTS = {n_prompts}
WARMUP = 2

KV_CACHE_DTYPE = "{kv_cache_dtype}"
ENABLE_CHUNKED_PREFILL = {enable_chunked_prefill}
PREFILL_CP = {prefill_cp}
BLOCK_SIZE = {block_size}
ATTN_BACKEND = "{attn_backend}"

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sample = "The quick brown fox jumps over the lazy dog. "
    encoded = tok.encode(sample)
    n_repeat = (CONTEXT_LEN // len(encoded)) + 1
    token_ids = (encoded * n_repeat)[:CONTEXT_LEN]
    prompt_text = tok.decode(token_ids, skip_special_tokens=True)
    print(f"Prompt: {{len(tok.encode(prompt_text))}} tokens", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )

    max_model_len = CONTEXT_LEN + OUTPUT_TOKENS + 128

    kwargs = dict(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp" if (TP * PREFILL_CP) > 1 else "uni",
        kv_cache_dtype=KV_CACHE_DTYPE,
    )

    if ENABLE_CHUNKED_PREFILL is not None:
        kwargs["enable_chunked_prefill"] = ENABLE_CHUNKED_PREFILL
    if PREFILL_CP > 1:
        kwargs["prefill_context_parallel_size"] = PREFILL_CP
    if BLOCK_SIZE:
        kwargs["block_size"] = BLOCK_SIZE
    if ATTN_BACKEND and ATTN_BACKEND != "default":
        os.environ["VLLM_ATTENTION_BACKEND"] = ATTN_BACKEND

    print(f"LLM kwargs: TP={{TP}} cp={{PREFILL_CP}} kv={{KV_CACHE_DTYPE}} blk={{BLOCK_SIZE}} attn={{ATTN_BACKEND}}", flush=True)

    llm = LLM(**kwargs)

    # Warmup
    print(f"Warmup {{WARMUP}} prompts...", flush=True)
    _ = llm.generate([prompt_text] * WARMUP, sampling_params)

    # Timed run
    print(f"Timing {{N_PROMPTS}} prompts...", flush=True)
    t0 = time.perf_counter()
    outputs = llm.generate([prompt_text] * N_PROMPTS, sampling_params)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    total_input = sum(len(o.prompt_token_ids) for o in outputs)
    total_output = sum(sum(len(c.token_ids) for c in o.outputs) for o in outputs)
    total = total_input + total_output

    result = {{
        "tp": TP,
        "prefill_cp": PREFILL_CP,
        "context_len": CONTEXT_LEN,
        "n_prompts": N_PROMPTS,
        "output_tokens": OUTPUT_TOKENS,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "enable_chunked_prefill": ENABLE_CHUNKED_PREFILL,
        "block_size": BLOCK_SIZE,
        "attn_backend": ATTN_BACKEND,
        "elapsed_s": round(elapsed, 3),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
        "input_throughput_tok_s": round(total_input / elapsed, 1),
        "output_throughput_tok_s": round(total_output / elapsed, 1),
        "total_throughput_tok_s": round(total / elapsed, 1),
        "avg_latency_per_prompt_s": round(elapsed / N_PROMPTS, 3),
    }}
    print("RESULT:" + json.dumps(result))
    return result

if __name__ == "__main__":
    main()
""")


@ray.remote(num_cpus=1)
def run_opt_benchmark(node_id, tp, context_len, n_prompts,
                      kv_cache_dtype="auto",
                      enable_chunked_prefill=None,
                      prefill_cp=1,
                      block_size=None,
                      attn_backend="default"):
    """Run a single optimized benchmark configuration."""
    import subprocess, os, json

    total_gpus = tp * prefill_cp
    cuda_devices = ",".join(str(i) for i in range(total_gpus))

    script = BENCH_SCRIPT.format(
        model=MODEL,
        tp=tp,
        context_len=context_len,
        output_tokens=OUTPUT_TOKENS,
        gpu_mem_util=GPU_MEM_UTIL,
        n_prompts=n_prompts,
        kv_cache_dtype=kv_cache_dtype,
        enable_chunked_prefill=str(enable_chunked_prefill) if enable_chunked_prefill is not None else "None",
        prefill_cp=prefill_cp,
        block_size=block_size if block_size else 0,
        attn_backend=attn_backend,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    env.pop("RAY_ADDRESS", None)

    label = f"TP={tp} CP={prefill_cp} ctx={context_len//1024}K kv={kv_cache_dtype} blk={block_size} attn={attn_backend}"
    print(f"[{os.uname().nodename}] {label} CUDA={cuda_devices}", flush=True)

    try:
        proc = subprocess.run(
            [SKYRL_VENV, script_path],
            capture_output=True, text=True, env=env,
            timeout=3600,
        )
        os.unlink(script_path)
        output = proc.stdout + proc.stderr
        print(output[-4000:] if len(output) > 4000 else output, flush=True)

        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])

        print(f"ERROR: No RESULT. returncode={proc.returncode}")
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {label}")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None


def compute_n_prompts(tp, context_len, prefill_cp=1, kv_fp8=False):
    """Estimate number of prompts that fit in GPU memory."""
    total_gpu_mem_gb = tp * prefill_cp * 141 * GPU_MEM_UTIL
    model_mem_gb = 44  # total model size, split across TP
    kv_per_token_mb = (2 * 20 * 128 * (1 if kv_fp8 else 2) * 47) / (1024 * 1024)
    kv_cache_available_gb = total_gpu_mem_gb - model_mem_gb
    max_kv_tokens = kv_cache_available_gb * 1024 / kv_per_token_mb
    tokens_per_prompt = context_len + OUTPUT_TOKENS
    n = max(1, int(max_kv_tokens * 0.8 / tokens_per_prompt))

    # Cap per context length for timing
    caps = {128*1024: 16, 64*1024: 32, 32*1024: 64, 16*1024: 128}
    for ctx_thresh, cap in caps.items():
        if context_len >= ctx_thresh:
            return min(n, cap)
    return min(n, 256)


def run_config(sched, tp, context_len, label, **opt_kwargs):
    """Run a benchmark configuration and return result."""
    kv_fp8 = opt_kwargs.get("kv_cache_dtype", "auto") != "auto"
    prefill_cp = opt_kwargs.get("prefill_cp", 1)
    n = compute_n_prompts(tp, context_len, prefill_cp=prefill_cp, kv_fp8=kv_fp8)

    print(f"\n{'='*70}")
    print(f"  {label} | ctx={context_len//1024}K | n={n}")
    print(f"{'='*70}")

    ref = run_opt_benchmark.options(scheduling_strategy=sched).remote(
        None, tp, context_len, n, **opt_kwargs
    )
    result = ray.get(ref, timeout=3600)
    if result:
        result["label"] = label
        print(f"  -> {result['total_throughput_tok_s']:.0f} tok/s total "
              f"| {result['input_throughput_tok_s']:.0f} input tok/s "
              f"| lat={result['avg_latency_per_prompt_s']:.2f}s/prompt")
    else:
        print(f"  -> FAILED")
    return result


def main():
    ray.init(address="auto")

    gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) >= 5]
    if not gpu_nodes:
        gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) > 0]

    bench_node = gpu_nodes[0]
    node_id = bench_node['NodeID']
    node_ip = bench_node['NodeManagerAddress']
    n_gpus = int(bench_node['Resources'].get('GPU', 0))

    print(f"Optimization benchmark on: {node_ip} ({n_gpus} GPUs)")
    sched = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    # Context lengths to test for optimization (focus on long context)
    CONTEXT_LENS = [32 * 1024, 64 * 1024, 128 * 1024]

    all_results = []

    for context_len in CONTEXT_LENS:
        ctx_label = f"{context_len//1024}K"
        print(f"\n\n{'#'*70}")
        print(f"# CONTEXT: {ctx_label}")
        print(f"{'#'*70}")

        configs = [
            # (label, kwargs)
            (f"A_baseline_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="auto", prefill_cp=1)),
            (f"B_fp8kv_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="fp8_e5m2", prefill_cp=1)),
            (f"C_cp2_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="auto", prefill_cp=2)),
            (f"D_fp8kv_cp2_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="fp8_e5m2", prefill_cp=2)),
            (f"E_chunked_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="auto", prefill_cp=1,
                enable_chunked_prefill=True)),
            (f"F_blk32_tp4_{ctx_label}", dict(
                tp=4, kv_cache_dtype="auto", prefill_cp=1,
                block_size=32)),
        ]

        for label, kwargs in configs:
            tp = kwargs.pop("tp")
            total_gpus = tp * kwargs.get("prefill_cp", 1)
            if total_gpus > n_gpus:
                print(f"  Skipping {label}: needs {total_gpus} GPUs, only {n_gpus} available")
                continue

            result = run_config(sched, tp, context_len, label, **kwargs)
            if result:
                all_results.append(result)
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Print final summary
    print("\n\n" + "="*120)
    print("OPTIMIZATION BENCHMARK RESULTS - GLM-4.7-Flash on H200 (TP=4)")
    print("="*120)
    header = f"{'Config':<35} | {'Context':>8} | {'Prompts':>7} | {'Total tok/s':>12} | {'Input tok/s':>12} | {'Out tok/s':>10} | {'Lat/prompt':>12} | {'Speedup':>8}"
    print(header)
    print("-"*120)

    # Group by context length for baseline comparison
    from collections import defaultdict
    by_ctx = defaultdict(list)
    for r in all_results:
        by_ctx[r['context_len']].append(r)

    for ctx_len in sorted(by_ctx.keys()):
        results = by_ctx[ctx_len]
        baseline = next((r for r in results if 'baseline' in r.get('label', '')), None)
        baseline_tps = baseline['total_throughput_tok_s'] if baseline else None

        for r in results:
            lbl = r.get('label', 'unknown')[:35]
            speedup = (r['total_throughput_tok_s'] / baseline_tps) if baseline_tps else 1.0
            print(f"{lbl:<35} | {r['context_len']//1024:>6}K | {r['n_prompts']:>7} | "
                  f"{r['total_throughput_tok_s']:>12,.0f} | {r['input_throughput_tok_s']:>12,.0f} | "
                  f"{r['output_throughput_tok_s']:>10,.0f} | {r['avg_latency_per_prompt_s']:>10.2f}s | "
                  f"{speedup:>7.2f}x")
        print()

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
