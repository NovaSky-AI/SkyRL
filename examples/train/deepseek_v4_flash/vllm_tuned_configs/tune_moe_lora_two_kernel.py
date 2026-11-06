"""Tune vLLM's two-kernel fused MoE-LoRA path (shrink + expand) for DeepSeek-V4.1-Flash and write the four
VLLM_TUNED_CONFIG_FOLDER files ({GPU}_FUSED_MOE_LORA_{W13,W2}_{SHRINK,EXPAND}.json). block_m per token count is the
padding-minimizing choice already used (make_moe_lora_configs.py); n/k/warps/stages/split_k are swept. GPU time is
measured inside CUDA graphs."""
import itertools, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_lora as b  # reuse setup / call / bench helpers (its __main__ loop is guarded below)

out_dir, gpu = sys.argv[1], torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
b.fm._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = 0
MS = [128, 256, 640, 1024, 2048, 4096, 8192]
tables = {k: {} for k in ("W13_SHRINK", "W13_EXPAND", "W2_SHRINK", "W2_EXPAND")}
for name, K, N, slices, mul in [("W13", b.H, b.I, 2, False), ("W2", b.I, b.H, 1, True)]:
    for m in MS:
        bm = b.block_m_for(m)
        d = b.setup(m, K, N, slices, bm)
        best_s = min(b.shrink_cands, key=lambda c: b.safe_bench(lambda: b.call(d, bm, c, b.default_e, mul)))
        best_e = min(b.expand_cands, key=lambda c: b.safe_bench(lambda: b.call(d, bm, best_s, c, mul)))
        t = b.safe_bench(lambda: b.call(d, bm, best_s, best_e, mul), reps=10)
        print(f"{name} m={m} bm={bm}: tuned {t:.1f}us shrink={best_s} expand={best_e}", flush=True)
        common = dict(block_m=bm, group_size_m=8)
        s_cfg = dict(common, block_n=best_s["bn"], block_k=best_s["bk"], num_warps=best_s["nw"],
                     num_stages=best_s["ns"], split_k=best_s["sk"])
        e_cfg = dict(common, block_n=best_e["bn"], block_k=best_e["bk"], num_warps=best_e["nw"],
                     num_stages=best_e["ns"], split_k=1)
        for slc in ("1", "2"):
            for key, cfg in ((f"{name}_SHRINK", s_cfg), (f"{name}_EXPAND", e_cfg)):
                # nearest-key lookup: one k / n / intermediate entry per m is enough
                tables[key].setdefault(slc, {})[str(m)] = {"5120": {"32": {"384": cfg}}}
for key, tab in tables.items():
    with open(os.path.join(out_dir, f"{gpu}_FUSED_MOE_LORA_{key}.json"), "w") as f:
        json.dump({"1": tab}, f, indent=1)
print("WROTE", gpu, flush=True)
