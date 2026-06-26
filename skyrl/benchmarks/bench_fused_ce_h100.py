"""
Benchmark: LM-head training signal — baseline vs NVIDIA fused CE vs OUR fused linear CE.

Mirrors the table in https://github.com/novasky-ai/skyrl/pull/1841 but exercises
*our* fork's verl/bytedance-derived fused kernel (PR #1765) rather than #1841's
liger port:

  baseline     : logits = hidden @ weightᵀ, then megatron-core's eager
                 ``vocab_parallel_cross_entropy`` (materializes [B,S,vocab//TP]
                 logits + an fp32 grad of the same shape).
  nvidia       : same logits, then ``fused_vocab_parallel_cross_entropy``
                 (@jit_fuser fused CE — fuses the softmax/CE stages, still
                 materializes the logits).
  fused-torch  : FusedLinearLogprob (pure-torch chunked fused linear logprob —
                 folds the projection into the chunked TP log-prob; logits never
                 materialized).
  fused-triton : FusedLinearLogprobTriton (vendored verl/bytedance flash kernel).

All run forward+backward so grads flow to hidden *and* weight.

Usage (single GPU; --vocab is the FULL vocab, per-rank shard = vocab // world):
    uv run --isolated --extra megatron torchrun --nproc_per_node=1 \\
        skyrl/benchmarks/bench_fused_ce_h100.py
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

# Qwen3.6-35B-A3B: hidden=2048, vocab=248320. Defaults show the per-rank shard
# for TP=4 (62080) and the full vocab (248320).
HIDDEN = 2048
VOCAB_SHARDS = [62080, 248320]
SEQ_LENS = [8192, 16384, 32768, 65536, 131072, 262144]
CHUNK_SIZE = 1024
WARMUP_REPS = 1
BENCH_REPS = 3
MODES = ["baseline", "nvidia", "fused-torch", "fused-triton"]


def _loss(mode, hidden, weight, target, vocab_local, chunk_size, tp_group):
    """Per-token CE summed to a scalar (so all modes produce identical grads)."""
    from megatron.core.fusions.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        FusedLinearLogprob,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
        FusedLinearLogprobTriton,
    )

    if mode == "fused-torch":
        lp = FusedLinearLogprob.apply(hidden, weight, target, 0, vocab_local, chunk_size, tp_group, False)
        return (-lp).sum()
    if mode == "fused-triton":
        lp = FusedLinearLogprobTriton.apply(hidden, weight, target, 0, vocab_local, chunk_size, tp_group, False)
        return (-lp).sum()
    logits = torch.matmul(hidden, weight.t())  # [B, S, vocab//TP]
    if mode == "baseline":
        ce = vocab_parallel_cross_entropy(logits, target, 0.0, tp_group)
    elif mode == "nvidia":
        ce = fused_vocab_parallel_cross_entropy(logits, target, tp_group)
    else:
        raise ValueError(mode)
    return ce.sum()


def _measure(mode, seq_len, vocab_local, chunk_size, tp_group, device, reps):
    """forward+backward; return (mean_ms, mean_peak_bytes) or (None, None) on OOM."""
    times, peaks = [], []
    for _ in range(reps):
        hidden = weight = target = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            hidden = torch.randn(1, seq_len, HIDDEN, dtype=torch.bfloat16, device=device, requires_grad=True)
            weight = (
                torch.randn(vocab_local, HIDDEN, dtype=torch.bfloat16, device=device) * (HIDDEN**-0.5)
            ).requires_grad_(True)
            target = torch.randint(0, vocab_local, (1, seq_len), device=device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = _loss(mode, hidden, weight, target, vocab_local, chunk_size, tp_group)
            loss.backward()
            torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated(device))
        except torch.OutOfMemoryError:
            return None, None
        finally:
            del hidden, weight, target, loss
            torch.cuda.empty_cache()
    return sum(times) / len(times), sum(peaks) / len(peaks)


def _correctness(vocab_local, tp_group, device):
    """All modes must produce the same loss + grad_hidden (fair comparison)."""
    torch.manual_seed(0)
    S = 64
    h0 = torch.randn(1, S, HIDDEN, dtype=torch.bfloat16, device=device)
    w0 = torch.randn(vocab_local, HIDDEN, dtype=torch.bfloat16, device=device) * (HIDDEN**-0.5)
    tgt = torch.randint(0, vocab_local, (1, S), device=device)
    ref = {}
    for mode in MODES:
        h = h0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        _loss(mode, h, w, tgt, vocab_local, CHUNK_SIZE, tp_group).backward()
        ref[mode] = h.grad.float().clone()
    base = ref["baseline"]
    return [f"{mode}: max|dgrad_hidden vs baseline|={(ref[mode] - base).abs().max().item():.2e}" for mode in MODES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=None, help="full vocab; per-rank shard = vocab // world_size")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    import megatron.core.parallel_state as mpu

    world = dist.get_world_size()
    mpu.initialize_model_parallel(tensor_model_parallel_size=world)
    tp_group = mpu.get_tensor_model_parallel_group()
    device = torch.device("cuda", local_rank)
    rank0 = dist.get_rank() == 0

    vocab_shards = [args.vocab // world] if args.vocab else VOCAB_SHARDS
    if rank0:
        print(f"Device {torch.cuda.get_device_name(device)} | TP(world)={world} | hidden={HIDDEN} | chunk={args.chunk_size}")
        print(
            "baseline = vocab_parallel_cross_entropy (eager) | nvidia = fused_vocab_parallel_cross_entropy | "
            "fused-torch = FusedLinearLogprob | fused-triton = FusedLinearLogprobTriton"
        )
        print("all: hidden+weight -> per-token CE -> sum -> backward\n")
        print("correctness (TP=%d, vocab=%d):" % (world, vocab_shards[0]))
        for line in _correctness(vocab_shards[0], tp_group, device):
            print("  " + line)
        print()

    cw = 14
    for vlocal in vocab_shards:
        if rank0:
            print(f"=== per-rank vocab shard = {vlocal:,} ===")
            hdr = (
                f"{'seq_len':>9} |"
                + "".join(f" {m + ' MB':>{cw}} |" for m in MODES)
                + "".join(f" {m + ' ms':>{cw}} |" for m in MODES)
            )
            print(hdr)
            print("-" * len(hdr))
        for s in SEQ_LENS:
            res = {}
            for mode in MODES:
                for _ in range(WARMUP_REPS):
                    _measure(mode, s, vlocal, args.chunk_size, tp_group, device, 1)
                res[mode] = _measure(mode, s, vlocal, args.chunk_size, tp_group, device, BENCH_REPS)
            if rank0:

                def mb(m):
                    p = res[m][1]
                    return "OOM" if p is None else f"{p / 1024**2:.0f}"

                def ms(m):
                    t = res[m][0]
                    return "OOM" if t is None else f"{t:.0f}"

                row = (
                    f"{s:>9} |"
                    + "".join(f" {mb(m):>{cw}} |" for m in MODES)
                    + "".join(f" {ms(m):>{cw}} |" for m in MODES)
                )
                print(row)
        if rank0:
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
