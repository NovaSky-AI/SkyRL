"""
Benchmark: chunked vs non-chunked logprob computation.

Tests log_softmax + gather over large vocab × large sequence,
which is the bottleneck in from_parallel_logits_to_logprobs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python examples/benchmarks/bench_chunked_logprobs.py
"""

import time
import torch
import torch.nn.functional as F

VOCAB_SIZE = 32000
SEQ_LENS = [32768, 65536, 131072]
CHUNK_SIZES = [None, 1024, 4096, 8192]
WARMUP_REPS = 2
BENCH_REPS = 5


def logprobs_chunked(logits: torch.Tensor, labels: torch.Tensor, chunk_size=None) -> torch.Tensor:
    """
    Compute log-probs matching the SkyRL chunked pattern.

    logits : [T, V]  — requires_grad must be True for gradient path
    labels : [T]     — token indices in [0, V)
    Returns: [T]     — per-token log-probs
    """
    if chunk_size is None:
        # Non-chunked: materialise full float32 logits at once
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    results = []
    for i in range(0, logits.shape[0], chunk_size):
        chunk = logits[i : i + chunk_size].float()
        lp = F.log_softmax(chunk, dim=-1)
        results.append(lp.gather(-1, labels[i : i + chunk_size].unsqueeze(-1)).squeeze(-1))
    return torch.cat(results)


def measure(logits: torch.Tensor, labels: torch.Tensor, chunk_size, reps: int):
    """Run forward+backward and return (mean_wall_ms, peak_mem_bytes)."""
    device = logits.device
    times = []
    peak_mems = []

    for _ in range(reps):
        # Fresh leaf tensor each rep so grad accumulation doesn't interfere
        logits_rep = logits.detach().requires_grad_(True)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        out = logprobs_chunked(logits_rep, labels, chunk_size=chunk_size)
        loss = out.sum()
        loss.backward()

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)
        peak_mems.append(torch.cuda.max_memory_allocated(device))

    return sum(times) / len(times), sum(peak_mems) / len(peak_mems)


def main():
    if not torch.cuda.is_available():
        raise SystemError("No CUDA device found. Set CUDA_VISIBLE_DEVICES=0.")

    device = torch.device("cuda", 0)
    print(f"Device : {torch.cuda.get_device_name(device)}")
    print(f"Vocab  : {VOCAB_SIZE:,}  |  chunk_sizes={CHUNK_SIZES}  |  warmup={WARMUP_REPS}  bench={BENCH_REPS}\n")

    col_w = 14
    header = (
        f"{'seq_len':>10}  "
        f"{'chunk_size':>10}  "
        f"{'time ms':>{col_w}}  "
        f"{'peak MB':>{col_w}}  "
        f"{'vs no-chunk':>{col_w}}  "
        f"{'mem saved MB':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for seq_len in SEQ_LENS:
        # Allocate logits in bfloat16 (typical LLM dtype) with gradient tracking
        logits = torch.randn(seq_len, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
        labels = torch.randint(0, VOCAB_SIZE, (seq_len,), device=device)

        # ----- warmup -----
        for cs in CHUNK_SIZES:
            for _ in range(WARMUP_REPS):
                _ = measure(logits, labels, chunk_size=cs, reps=1)

        # ----- benchmark: collect baseline (no-chunk) first -----
        t_baseline, mem_baseline = measure(logits, labels, chunk_size=None, reps=BENCH_REPS)

        # ----- print one row per chunk_size -----
        for cs in CHUNK_SIZES:
            if cs is None:
                t_cs, mem_cs = t_baseline, mem_baseline
            else:
                t_cs, mem_cs = measure(logits, labels, chunk_size=cs, reps=BENCH_REPS)

            speedup = t_baseline / t_cs if t_cs > 0 else float("inf")
            mem_cs_mb = mem_cs / (1024**2)
            mem_baseline_mb = mem_baseline / (1024**2)
            mem_saved_mb = mem_baseline_mb - mem_cs_mb
            cs_label = "None" if cs is None else str(cs)

            print(
                f"{seq_len:>10,}  "
                f"{cs_label:>10}  "
                f"{t_cs:>{col_w}.1f}  "
                f"{mem_cs_mb:>{col_w}.0f}  "
                f"{speedup:>{col_w}.2f}x  "
                f"{mem_saved_mb:>{col_w}.0f}"
            )

        print(sep)

        # Free memory before next seq_len
        del logits, labels
        torch.cuda.empty_cache()

    print("All times are mean wall-clock (ms) over forward+backward passes.")
    print("vs no-chunk: speedup relative to chunk_size=None (>1 = faster).")


if __name__ == "__main__":
    main()
