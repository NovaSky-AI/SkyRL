"""Tune vLLM's dense LoRA shrink / expand Triton configs for DeepSeek-V4.1-Flash's LoRA modules (TP8, rank 32).

Writes {GPU}_SHRINK.json, {GPU}_EXPAND_TRUE.json, {GPU}_EXPAND_FALSE.json into OUT_DIR in vLLM's
VLLM_TUNED_CONFIG_FOLDER layout: config[max_loras][num_slices][m][k][n] (shrink k=input dim, n=rank;
expand k=rank, n=output dim). Each config is the fastest of a sweep, timed with CUDA events on the real ops.
"""
import itertools, json, os, sys, time
import torch

out_dir = sys.argv[1]
R, MAX_LORAS = 32, 1
MS = [128, 640, 1024, 2048, 4096, 8192]
SHRINK = [(5120, 2), (1280, 1), (1024, 1), (288, 1)]          # (K = input dim, num_slices)
EXPAND = [(1280, 2), (4096, 1), (5120, 1), (288, 2)]          # (N = output dim per slice, num_slices)

import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_mod
import vllm.lora.ops.triton_ops.lora_expand_op as expand_mod
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

dev = torch.device("cuda")
meta = LoRAKernelMeta.make(max_loras=MAX_LORAS, max_num_tokens=max(MS), device=dev)
current = {}
def fake_cfg(*a, **k):
    return current["cfg"]
shrink_mod.get_lora_op_configs = fake_cfg
expand_mod.get_lora_op_configs = fake_cfg

def bench(fn, reps=20):
    """GPU time per call: capture REPS calls in one CUDA graph and time its replay (the engine replays these
    kernels inside CUDA graphs, so host-side launch overhead must not count)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(3):
        g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (3 * reps)

def best_of(cands, fn):
    best = None
    for c in cands:
        current["cfg"] = c
        try:
            t = bench(fn)
        except Exception:
            continue
        if best is None or t < best[0]:
            best = (t, c)
    return best

shrink_cands = [dict(block_m=bm, block_n=32, block_k=bk, split_k=sk, num_warps=nw, num_ctas=1, group_size_m=8,
                     num_stages=ns, max_nreg=None)
                for bm, bk, sk, nw, ns in itertools.product([16, 32, 64], [64, 128, 256], [1, 2, 4, 8, 16], [4, 8], [2, 3, 4])]
expand_cands = [dict(block_m=bm, block_n=bn, block_k=bk, num_warps=nw, num_ctas=1, num_stages=ns, max_nreg=None)
                for bm, bn, bk, nw, ns in itertools.product([16, 32, 64, 128], [64, 128, 256], [16, 32], [4, 8], [2, 3])]
default_shrink = lambda m: dict(block_m=32, block_n=16, block_k=256 if m < 128 else 32, split_k=64 if m < 128 else 8,
                                num_warps=4, num_ctas=1, group_size_m=8, num_stages=2, max_nreg=None)

shrink_tab, expand_tab = {}, {}
for m in MS:
    meta.prepare_tensors(torch.zeros(m, dtype=torch.int32, device=dev))
    margs = meta.meta_args(m, specialize_active_lora=False)
    for K, S in SHRINK:
        x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)
        A = [torch.randn(MAX_LORAS, R, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(S)]
        y = torch.zeros(S, m, R, dtype=torch.float32, device=dev)
        fn = lambda: shrink_mod._lora_shrink(x, A, y, *margs, 1.0)
        current["cfg"] = default_shrink(m); t0 = bench(fn)
        t, c = best_of(shrink_cands, fn)
        shrink_tab.setdefault(str(S), {}).setdefault(str(m), {}).setdefault(str(K), {})[str(R)] = c
        print(f"shrink m={m} K={K} S={S}: default {t0*1e3:.1f}us best {t*1e3:.1f}us {c}", flush=True)
    for N, S in EXPAND:
        xin = torch.randn(S, m, R, dtype=torch.bfloat16, device=dev)
        B = [torch.randn(MAX_LORAS, N, R, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(S)]
        y = torch.zeros(m, N * S, dtype=torch.bfloat16, device=dev)
        fn = lambda: expand_mod._lora_expand(xin, B, y, *margs, 0, True)
        current["cfg"] = dict(block_m=64, block_n=64 if S > 1 else 128, block_k=32, num_warps=4, num_ctas=1,
                              num_stages=2, max_nreg=None)
        t0 = bench(fn)
        t, c = best_of(expand_cands, fn)
        expand_tab.setdefault(str(S), {}).setdefault(str(m), {}).setdefault(str(R), {})[str(N)] = c
        print(f"expand m={m} N={N} S={S}: default {t0*1e3:.1f}us best {t*1e3:.1f}us {c}", flush=True)

gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
os.makedirs(out_dir, exist_ok=True)
json.dump({str(MAX_LORAS): shrink_tab}, open(os.path.join(out_dir, f"{gpu}_SHRINK.json"), "w"), indent=1)
for flag in ("TRUE", "FALSE"):
    json.dump({str(MAX_LORAS): expand_tab}, open(os.path.join(out_dir, f"{gpu}_EXPAND_{flag}.json"), "w"), indent=1)
print("WROTE", gpu, flush=True)
