"""
Test Flash Attention 3 (via TransformerEngine fused attention) for Megatron training.

Phase 6 uses NVTE_FUSED_ATTN=0, forcing Megatron to use flash-attn v2 (FA2).
This test checks if NVTE_FUSED_ATTN=1 (FA3 via cuDNN on Hopper) works and
how much faster it is.

Tests:
1. Single-GPU forward+backward with FA2 (NVTE_FUSED_ATTN=0)
2. Single-GPU forward+backward with FA3 (NVTE_FUSED_ATTN=1)
3. Compare speed and numerical differences

Usage:
    RAY_ADDRESS=auto python scripts/glm47_h200/test_fa3_training.py
"""
import os
import sys
import json
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

MODEL = "zai-org/GLM-4.7-Flash"
SKYRL_VENV = "/home/ray/SkyRL/.venv/bin/python"


TEST_SCRIPT = r"""
import os
import sys
import time
import json
import gc

# Set NVTE_FUSED_ATTN BEFORE any imports
NVTE_FUSED_ATTN = "{nvte_fused_attn}"
os.environ["NVTE_FUSED_ATTN"] = NVTE_FUSED_ATTN

import torch
import torch.nn.functional as F

def run_test():
    results = {{}}
    results["nvte_fused_attn"] = NVTE_FUSED_ATTN

    # Check what TE reports
    try:
        import transformer_engine as te
        results["te_version"] = te.__version__
    except ImportError:
        results["te_error"] = "not installed"
        return results

    # Check flash-attn version
    try:
        import flash_attn
        results["flash_attn_version"] = flash_attn.__version__
    except ImportError:
        results["flash_attn_version"] = "not installed"

    device = torch.device("cuda:0")

    # Create synthetic data matching GLM-4.7-Flash dimensions
    # GLM: hidden_size=2048, num_attention_heads=20, head_dim=102.4 (actually uses MLA)
    # For this test, use standard attention dimensions
    batch_size = {batch_size}
    seq_len = {seq_len}
    num_heads = 20
    head_dim = 128  # Standard head dim for flash attention
    hidden_size = num_heads * head_dim  # 2560

    print(f"Testing NVTE_FUSED_ATTN={{NVTE_FUSED_ATTN}} batch={{batch_size}} seq={{seq_len}}", flush=True)

    # Create Q, K, V tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Test flash attention directly
    from flash_attn import flash_attn_func

    # Warmup
    for _ in range(3):
        out = flash_attn_func(q, k, v, causal=True)
        loss = out.sum()
        loss.backward()
        q.grad = None; k.grad = None; v.grad = None

    torch.cuda.synchronize()

    # Timed forward + backward with flash-attn (FA2)
    fa2_times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, causal=True)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        fa2_times.append(t1 - t0)
        q.grad = None; k.grad = None; v.grad = None

    results["flash_attn_v2_fwd_bwd_ms"] = round(sum(fa2_times) / len(fa2_times) * 1000, 2)
    results["flash_attn_v2_output_sample"] = out[0, 0, 0, :4].tolist()

    # Test TE DotProductAttention if NVTE_FUSED_ATTN=1
    try:
        from transformer_engine.pytorch.attention import DotProductAttention as TEDotProductAttention
        te_dpa = TEDotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            attention_dropout=0.0,
        ).to(device).to(torch.bfloat16)

        # TE expects (seq_len, batch, num_heads, head_dim)
        q_te = q.transpose(0, 1).contiguous()
        k_te = k.transpose(0, 1).contiguous()
        v_te = v.transpose(0, 1).contiguous()

        # Warmup
        for _ in range(3):
            out_te = te_dpa(q_te, k_te, v_te)
            loss_te = out_te.sum()
            loss_te.backward()
            q_te.grad = None; k_te.grad = None; v_te.grad = None

        torch.cuda.synchronize()

        # Timed
        te_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_te = te_dpa(q_te, k_te, v_te)
            loss_te = out_te.sum()
            loss_te.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            te_times.append(t1 - t0)
            q_te.grad = None; k_te.grad = None; v_te.grad = None

        results["te_dpa_fwd_bwd_ms"] = round(sum(te_times) / len(te_times) * 1000, 2)
        results["te_dpa_output_sample"] = out_te[0, 0, 0, :4].tolist()
        results["te_dpa_speedup_vs_fa2"] = round(results["flash_attn_v2_fwd_bwd_ms"] / results["te_dpa_fwd_bwd_ms"], 2)
    except Exception as e:
        results["te_dpa_error"] = str(e)

    # GPU memory
    results["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
    results["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)

    print("RESULT:" + json.dumps(results))
    return results

if __name__ == "__main__":
    run_test()
"""


@ray.remote(num_cpus=1)
def run_fa_test(nvte_fused_attn, batch_size, seq_len):
    import subprocess, os, json, tempfile

    script = TEST_SCRIPT.format(
        nvte_fused_attn=nvte_fused_attn,
        batch_size=batch_size,
        seq_len=seq_len,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        script_path = f.name

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "NVTE_FUSED_ATTN": nvte_fused_attn,
    }
    env.pop("RAY_ADDRESS", None)

    try:
        proc = subprocess.run(
            [SKYRL_VENV, script_path],
            capture_output=True, text=True, env=env,
            timeout=300,
        )
        os.unlink(script_path)

        output = proc.stdout + proc.stderr
        print(output[-3000:] if len(output) > 3000 else output, flush=True)

        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return json.loads(line[7:])

        print(f"ERROR: No RESULT. returncode={proc.returncode}")
        return {"error": "no result", "stderr": proc.stderr[-1000:]}
    except Exception as e:
        return {"error": str(e)}


def main():
    ray.init(address="auto")

    gpu_nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) > 0]
    bench_node = gpu_nodes[0]
    sched = NodeAffinitySchedulingStrategy(node_id=bench_node['NodeID'], soft=False)

    print(f"Testing FA2 vs FA3 on {bench_node['NodeManagerAddress']}")

    test_configs = [
        # (batch, seq_len) - representative of training workloads
        (1, 8192),    # micro_batch=1, 8K context (Phase 6 config)
        (2, 8192),    # micro_batch=2, 8K context
        (4, 4096),    # micro_batch=4, 4K context
        (1, 16384),   # micro_batch=1, 16K context
    ]

    all_results = []

    for batch_size, seq_len in test_configs:
        print(f"\n{'='*60}")
        print(f"  batch={batch_size}, seq_len={seq_len}")
        print(f"{'='*60}")

        for nvte in ["0", "1"]:
            label = "FA2 (NVTE_FUSED_ATTN=0)" if nvte == "0" else "FA3 (NVTE_FUSED_ATTN=1)"
            print(f"\n  {label}:")
            ref = run_fa_test.options(scheduling_strategy=sched).remote(nvte, batch_size, seq_len)
            result = ray.get(ref, timeout=300)
            if result:
                result["batch_size"] = batch_size
                result["seq_len"] = seq_len
                all_results.append(result)
                if "flash_attn_v2_fwd_bwd_ms" in result:
                    print(f"    FA2 fwd+bwd: {result['flash_attn_v2_fwd_bwd_ms']} ms")
                if "te_dpa_fwd_bwd_ms" in result:
                    print(f"    TE DPA fwd+bwd: {result['te_dpa_fwd_bwd_ms']} ms")
                    print(f"    Speedup: {result.get('te_dpa_speedup_vs_fa2', 'N/A')}x")
                if "te_dpa_error" in result:
                    print(f"    TE DPA error: {result['te_dpa_error']}")

    # Summary
    print(f"\n\n{'='*70}")
    print("FA2 vs FA3 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        nvte = r.get("nvte_fused_attn", "?")
        label = "FA2" if nvte == "0" else "FA3"
        bs = r.get("batch_size", "?")
        sl = r.get("seq_len", "?")
        fa2 = r.get("flash_attn_v2_fwd_bwd_ms", "N/A")
        te = r.get("te_dpa_fwd_bwd_ms", "N/A")
        speedup = r.get("te_dpa_speedup_vs_fa2", "N/A")
        print(f"  NVTE={nvte} batch={bs} seq={sl}: FA2={fa2}ms TE_DPA={te}ms speedup={speedup}x")

    results_path = "/tmp/fa3_benchmark.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
