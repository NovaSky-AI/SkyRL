"""Fused MoE-LoRA: vLLM's one-shot kernel vs its two-kernel (shrink + expand) path at DeepSeek-V4.1-Flash shapes.

384 experts, top-6, rank 32, one active adapter, TP8 (padded intermediate 384). GPU time per call inside CUDA graphs.
"""
import itertools, sys
import torch
import vllm._custom_ops as ops
import vllm.lora.ops.triton_ops.fused_moe_lora_op as fm

E, TOPK, R, H, I = 384, 6, 32, 5120, 384
dev = torch.device("cuda")
torch.manual_seed(0)


def block_m_for(m):
    avg = m * TOPK / E
    for b in (16, 32, 64, 128):
        if avg <= 0.9 * b:
            return b
    return 128


def bench(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(3):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (3 * reps) * 1e3  # us


def safe_bench(fn, reps=5):
    try:
        return bench(fn, reps)
    except Exception:
        torch.cuda.synchronize()
        return float("inf")


def setup(m, K, N, slices, bm):
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(m)]).to(torch.int32)
    topk_w = torch.rand(m, TOPK, device=dev, dtype=torch.float32)
    token_lora = torch.zeros(m, dtype=torch.int32, device=dev)
    adapter_enabled = torch.ones(2, dtype=torch.int32, device=dev)
    lora_ids = torch.tensor([0, -1], dtype=torch.int32, device=dev)
    max_pad = topk_ids.numel() + E * (bm - 1)
    max_blocks = (max_pad + bm - 1) // bm
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=dev)
    ntpp = torch.empty(1, dtype=torch.int32, device=dev)
    ops.moe_lora_align_block_size(topk_ids, token_lora, E, bm, 1, max_pad, max_blocks, sorted_ids, expert_ids,
                                  ntpp, adapter_enabled, lora_ids, None)
    x = torch.randn(m, K, dtype=torch.bfloat16, device=dev)
    A = [torch.randn(1, E, R, K, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(slices)]
    B = [torch.randn(1, E, N, R, dtype=torch.bfloat16, device=dev) * 0.02 for _ in range(slices)]
    out = torch.zeros(m, TOPK, N * slices, dtype=torch.bfloat16, device=dev)
    return dict(x=x, A=A, B=B, out=out, topk_w=topk_w, sorted_ids=sorted_ids.view(1, -1),
                expert_ids=expert_ids.view(1, -1), ntpp=ntpp, token_lora=token_lora, lora_ids=lora_ids,
                num_active=torch.tensor([1], dtype=torch.int32), adapter_enabled=adapter_enabled)


def call(d, bm, cfg_s, cfg_e, mul_routed):
    fm._fused_moe_lora(
        d["out"], d["x"], d["A"], d["B"], d["topk_w"], d["sorted_ids"], d["expert_ids"], d["ntpp"],
        d["token_lora"], R, TOPK, d["lora_ids"], d["num_active"], d["adapter_enabled"],
        bm, cfg_s["bn"], cfg_s["bk"], 8, cfg_s["nw"], cfg_s["ns"], cfg_s["sk"],
        bm, cfg_e["bn"], cfg_e["bk"], 8, cfg_e["nw"], cfg_e["ns"], 1,
        mul_routed_weight=mul_routed, add_inputs=False,
    )


shrink_cands = [dict(bn=32, bk=bk, nw=nw, ns=ns, sk=sk) for bk, nw, ns, sk in
                itertools.product([64, 128, 256], [4, 8], [2, 3, 4], [1, 2, 4, 8])]
expand_cands = [dict(bn=bn, bk=32, nw=nw, ns=ns) for bn, nw, ns in itertools.product([64, 128, 256], [4, 8], [2, 3, 4])]
default_s, default_e = dict(bn=32, bk=32, nw=4, ns=3, sk=1), dict(bn=64, bk=32, nw=4, ns=3)
one_shot_max = fm._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK

if __name__ != "__main__":
    pass
else:
  for name, K, N, slices, mul in [("w13", H, I, 2, False), ("w2", I, H, 1, True)]:
      for m in (640, 2048, 8192):
          bm = block_m_for(m)
          d = setup(m, K, N, slices, bm)
          fm._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = one_shot_max
          t_one = bench(lambda: call(d, bm, default_s, default_e, mul))
          fm._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = 0  # force the two-kernel path
          t_two_def = bench(lambda: call(d, bm, default_s, default_e, mul))
          # tune shrink (with default expand), then expand (with best shrink)
          best_s = min(shrink_cands, key=lambda c: safe_bench(lambda: call(d, bm, c, default_e, mul)))
          best_e = min(expand_cands, key=lambda c: safe_bench(lambda: call(d, bm, best_s, c, mul)))
          t_two_best = bench(lambda: call(d, bm, best_s, best_e, mul))
          fm._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = one_shot_max
          print(f"{name} m={m} bm={bm}: one-shot {t_one:.1f}us | two-kernel default {t_two_def:.1f}us "
                f"tuned {t_two_best:.1f}us shrink={best_s} expand={best_e}", flush=True)
  print("DONE", flush=True)
