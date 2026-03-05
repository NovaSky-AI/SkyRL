"""
Ray-based benchmark runner for GLM-4.7-Flash vLLM throughput.

Runs each (TP, context_len) config sequentially on a single GPU node,
using subprocess to avoid Ray GPU allocation conflicts between vLLM runs.
"""
import os
import sys
import json
import time
import subprocess
import tempfile
import textwrap
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

MODEL = "zai-org/GLM-4.7-Flash"
SKYRL_VENV = "/home/ray/SkyRL/.venv/bin/python"
RESULTS_FILE = "/tmp/vllm_benchmark_results.json"

# TP sizes to test (must divide GLM's 20 attention heads evenly)
TP_SIZES = [1, 2, 4, 5]

# Context lengths to test (GLM supports up to 202K natively)
CONTEXT_LENS = [16 * 1024, 32 * 1024, 64 * 1024, 128 * 1024]

# Output tokens per prompt (fixed; enough to measure decode throughput)
OUTPUT_TOKENS = 256

# GPU memory utilization for vLLM
GPU_MEM_UTIL = 0.92


BENCH_SCRIPT = textwrap.dedent(r"""
import os, sys, time, json, gc
import torch

MODEL = "{model}"
TP = {tp}
CONTEXT_LEN = {context_len}
OUTPUT_TOKENS = {output_tokens}
GPU_MEM_UTIL = {gpu_mem_util}
WARMUP = 2
N_PROMPTS = {n_prompts}

def main():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # Build a prompt of exactly CONTEXT_LEN tokens
    # Use repetition of a common token
    sample = "The quick brown fox jumps over the lazy dog. "
    encoded = tok.encode(sample)
    n_repeat = (CONTEXT_LEN // len(encoded)) + 1
    token_ids = (encoded * n_repeat)[:CONTEXT_LEN]
    prompt_text = tok.decode(token_ids, skip_special_tokens=True)

    print(f"Prompt length: {{len(tok.encode(prompt_text))}} tokens", flush=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )

    max_model_len = CONTEXT_LEN + OUTPUT_TOKENS + 128

    print(f"Loading vLLM: TP={{TP}}, max_model_len={{max_model_len}}...", flush=True)
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=TP,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
        enable_prefix_caching=False,
        disable_custom_all_reduce=True,
        distributed_executor_backend="mp" if TP > 1 else "uni",
    )

    # Warmup
    print(f"Warmup ({{WARMUP}} prompts)...", flush=True)
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

    # Collect GPU memory stats
    gpu_mem = {{}}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        used = torch.cuda.memory_allocated(i) / 1024**3
        total_mem = props.total_memory / 1024**3
        gpu_mem[str(i)] = {{"used_gb": round(used, 2), "total_gb": round(total_mem, 1)}}

    result = {{
        "tp": TP,
        "context_len": CONTEXT_LEN,
        "n_prompts": N_PROMPTS,
        "output_tokens": OUTPUT_TOKENS,
        "elapsed_s": round(elapsed, 3),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
        "input_throughput_tok_s": round(total_input / elapsed, 1),
        "output_throughput_tok_s": round(total_output / elapsed, 1),
        "total_throughput_tok_s": round(total / elapsed, 1),
        "avg_latency_per_prompt_s": round(elapsed / N_PROMPTS, 3),
        "gpu_mem": gpu_mem,
    }}
    print("RESULT:" + json.dumps(result))
    return result

if __name__ == "__main__":
    main()
""")


@ray.remote(num_cpus=1)
def run_single_benchmark(node_id: str, tp: int, context_len: int,
                          n_prompts: int, output_tokens: int):
    """Run a single vLLM benchmark configuration as a subprocess."""
    import subprocess, os, json

    # Write benchmark script to temp file
    script = BENCH_SCRIPT.format(
        model=MODEL,
        tp=tp,
        context_len=context_len,
        output_tokens=output_tokens,
        gpu_mem_util=GPU_MEM_UTIL,
        n_prompts=n_prompts,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    # GPU IDs: 0..tp-1
    cuda_devices = ",".join(str(i) for i in range(tp))

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_devices,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",  # use mp subprocess for EngineCore
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    # Remove Ray address so vLLM doesn't try to use the Ray cluster for TP workers
    env.pop("RAY_ADDRESS", None)
    env.pop("VLLM_CONFIGURE_LOGGING", None)

    print(f"[node={os.uname().nodename}] TP={tp} ctx={context_len//1024}K "
          f"n={n_prompts} CUDA={cuda_devices}", flush=True)

    try:
        proc = subprocess.run(
            [SKYRL_VENV, script_path],
            capture_output=True, text=True, env=env,
            timeout=3600,  # 1 hour max
        )
        os.unlink(script_path)

        output = proc.stdout + proc.stderr
        print(output[-3000:] if len(output) > 3000 else output, flush=True)

        # Extract RESULT: line from output
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])

        print(f"ERROR: No RESULT line found. Return code: {proc.returncode}")
        print("STDOUT:", proc.stdout[-2000:])
        print("STDERR:", proc.stderr[-2000:])
        return None
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: TP={tp} ctx={context_len//1024}K")
        return None
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return None


def get_n_prompts(tp: int, context_len: int) -> int:
    """
    Determine how many prompts to use based on GPU count and context length.
    Each H200 has 141GB. With tp GPUs, total = tp * 141GB.
    Model ~44GB total (bp16), KV cache varies.
    head_dim for GLM = hidden_size / num_heads? Actually 2048/20 = 102.4...
    Let's use 128 as conservative estimate.
    KV cache per token = 2 * 20 * 128 * 2 * 47 = 483,328 bytes ≈ 0.46 MB
    """
    total_gpu_mem_gb = tp * 141 * GPU_MEM_UTIL
    model_mem_gb = 44  # total model in bf16 (fixed, split across TP)
    kv_per_token_mb = (2 * 20 * 128 * 2 * 47) / (1024 * 1024)  # ~0.46 MB
    kv_cache_available_gb = total_gpu_mem_gb - model_mem_gb
    max_kv_tokens = kv_cache_available_gb * 1024 / kv_per_token_mb

    # Each prompt uses context_len + output_tokens tokens for KV
    tokens_per_prompt = context_len + OUTPUT_TOKENS
    max_prompts = max(1, int(max_kv_tokens * 0.8 / tokens_per_prompt))

    # Cap for practical timing
    if context_len >= 128 * 1024:
        max_prompts = min(max_prompts, 8)
    elif context_len >= 64 * 1024:
        max_prompts = min(max_prompts, 16)
    elif context_len >= 32 * 1024:
        max_prompts = min(max_prompts, 32)
    else:
        max_prompts = min(max_prompts, 64)

    return max(1, max_prompts)


def main():
    ray.init(address="auto")

    # Pick a single GPU node to run all benchmarks
    gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) >= 5]
    if not gpu_nodes:
        gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) > 0]

    if not gpu_nodes:
        print("ERROR: No GPU nodes found!")
        return

    # Use the first available node
    bench_node = gpu_nodes[0]
    node_id = bench_node['NodeID']
    node_ip = bench_node['NodeManagerAddress']
    n_gpus = int(bench_node['Resources'].get('GPU', 0))

    print(f"Benchmarking on node: {node_ip} ({n_gpus} GPUs)")
    print(f"TP sizes: {TP_SIZES}")
    print(f"Context lengths: {[f'{c//1024}K' for c in CONTEXT_LENS]}")

    all_results = []

    for tp in TP_SIZES:
        if tp > n_gpus:
            print(f"Skipping TP={tp}: node only has {n_gpus} GPUs")
            continue

        for context_len in CONTEXT_LENS:
            n_prompts = get_n_prompts(tp, context_len)
            print(f"\n{'='*60}")
            print(f"Running: TP={tp}, ctx={context_len//1024}K, n_prompts={n_prompts}")
            print(f"{'='*60}")

            # Run benchmark with NodeAffinity to pin to bench_node
            sched = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ref = run_single_benchmark.options(
                scheduling_strategy=sched
            ).remote(node_id, tp, context_len, n_prompts, OUTPUT_TOKENS)

            result = ray.get(ref, timeout=3600)

            if result:
                all_results.append(result)
                print(f"SUCCESS: {result['total_throughput_tok_s']:.0f} total tok/s")
            else:
                print(f"FAILED: TP={tp} ctx={context_len//1024}K")
                # Try with fewer prompts if failed
                if n_prompts > 1:
                    print(f"Retrying with 1 prompt...")
                    ref = run_single_benchmark.options(
                        scheduling_strategy=sched
                    ).remote(node_id, tp, context_len, 1, OUTPUT_TOKENS)
                    result = ray.get(ref, timeout=3600)
                    if result:
                        all_results.append(result)
                        print(f"RETRY SUCCESS: {result['total_throughput_tok_s']:.0f} total tok/s")

            # Save intermediate results
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)

    # Print summary
    print("\n\n" + "="*110)
    print("FINAL BENCHMARK RESULTS - GLM-4.7-Flash on H200")
    print("="*110)
    header = f"{'TP':>4} | {'Context':>8} | {'Prompts':>7} | {'Total tok/s':>12} | {'Input tok/s':>12} | {'Out tok/s':>10} | {'Lat/prompt':>11}"
    print(header)
    print("-"*110)
    for r in all_results:
        print(f"{r['tp']:>4} | {r['context_len']//1024:>6}K | {r['n_prompts']:>7} | "
              f"{r['total_throughput_tok_s']:>12,.0f} | {r['input_throughput_tok_s']:>12,.0f} | "
              f"{r['output_throughput_tok_s']:>10,.0f} | {r['avg_latency_per_prompt_s']:>10.2f}s")

    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
