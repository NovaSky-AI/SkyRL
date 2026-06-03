"""
Benchmark: chunked vs non-chunked logprob computation.

Tests ChunkedDistributedLogprob and DistributedLogprob from
skyrl/backends/skyrl_train/distributed/megatron/model_utils.py,
which is the actual code path used in SkyRL's MegatronModelWrapper.

Usage (single GPU, torchrun required for distributed init):
    uv run --isolated --extra megatron torchrun --nproc_per_node=1 \\
        skyrl/benchmarks/bench_chunked_logprobs.py
"""

import os
import time
from typing import Optional

import torch
import torch.distributed as dist

VOCAB_SIZES = [32000, 64000, 128000]
SEQ_LENS = [32768, 65536, 131072]
# chunk_size=None routes through DistributedLogprob (no chunking); all others use
# ChunkedDistributedLogprob with the given chunk size.
CHUNK_SIZES = [None, 32, 1024, 4096, 8192, 16384]
WARMUP_REPS = 2
BENCH_REPS = 5


def measure(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    chunk_size: Optional[int],
    tp_group: torch.distributed.ProcessGroup,
    reps: int,
):
    """Run forward+backward through the real SkyRL logprob kernel.

    Returns (mean_wall_ms, mean_peak_mem_bytes).
    """
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    device = vocab_parallel_logits.device
    times = []
    peak_mems = []

    for _ in range(reps):
        # Fresh leaf tensor each rep so grad accumulation does not interfere.
        logits_rep = vocab_parallel_logits.detach().requires_grad_(True)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        if chunk_size is None:
            # Non-chunked real implementation (DistributedLogprob)
            out = DistributedLogprob.apply(
                logits_rep,
                target,
                vocab_start_index,
                vocab_end_index,
                tp_group,
                False,  # inference_only=False -> saves tensors for backward
            )
        else:
            # Chunked real implementation (ChunkedDistributedLogprob)
            out = ChunkedDistributedLogprob.apply(
                logits_rep,
                target,
                vocab_start_index,
                vocab_end_index,
                chunk_size,
                tp_group,
                False,  # inference_only=False -> saves tensors for backward
            )

        loss = out.sum()
        loss.backward()

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)
        peak_mems.append(torch.cuda.max_memory_allocated(device))

    return sum(times) / len(times), sum(peak_mems) / len(peak_mems)


def main():
    # --- Distributed init (required by the real SkyRL kernel) ---
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Initialise Megatron model-parallel state (TP=1, single GPU).
    import megatron.core.parallel_state as mpu

    mpu.initialize_model_parallel(tensor_model_parallel_size=1)

    tp_group = dist.group.WORLD  # TP=1, so the whole-world group is the TP group

    device = torch.device("cuda", local_rank)

    if dist.get_rank() == 0:
        print(f"Device      : {torch.cuda.get_device_name(device)}")
        print(f"World size  : {dist.get_world_size()}")
        print(
            f"Vocab sizes : {VOCAB_SIZES}  |  chunk_sizes={CHUNK_SIZES}  "
            f"|  warmup={WARMUP_REPS}  bench={BENCH_REPS}"
        )
        print("Implementation: real SkyRL ChunkedDistributedLogprob / DistributedLogprob\n")

    col_w = 14
    header = (
        f"{'vocab_size':>10}  "
        f"{'seq_len':>10}  "
        f"{'chunk_size':>10}  "
        f"{'time ms':>{col_w}}  "
        f"{'peak MB':>{col_w}}  "
        f"{'vs no-chunk':>{col_w}}  "
        f"{'mem saved MB':>{col_w}}"
    )
    sep = "-" * len(header)

    for vocab_size in VOCAB_SIZES:
        if dist.get_rank() == 0:
            print(f"\n=== vocab_size={vocab_size:,} ===")
            print(header)
            print(sep)

        for seq_len in SEQ_LENS:
            # With TP=1 the full vocab lives on this rank.
            vocab_start_index = 0
            vocab_end_index = vocab_size

            # Shape expected by the real kernel: [batch, seq, vocab // TP]
            # We use batch=1 to keep allocations comparable to a single-sequence workload.
            try:
                logits = torch.randn(1, seq_len, vocab_size, dtype=torch.bfloat16, device=device)
                # targets: [batch, seq], values in [0, vocab_size)
                target = torch.randint(0, vocab_size, (1, seq_len), device=device)
            except torch.OutOfMemoryError:
                if dist.get_rank() == 0:
                    oom_row = (
                        f"{vocab_size:>10,}  "
                        f"{seq_len:>10,}  "
                        f"{'(all)':>10}  "
                        f"{'OOM':>{col_w}}  "
                        f"{'OOM':>{col_w}}  "
                        f"{'OOM':>{col_w}}  "
                        f"{'OOM':>{col_w}}"
                    )
                    print(oom_row)
                    print(sep)
                torch.cuda.empty_cache()
                continue

            # ----- single pass: warmup + benchmark inline per chunk size -----
            results: dict[Optional[int], tuple[Optional[float], Optional[float]]] = {}
            for cs in CHUNK_SIZES:
                try:
                    for _ in range(WARMUP_REPS):
                        measure(logits, target, vocab_start_index, vocab_end_index, cs, tp_group, reps=1)
                    t_cs, mem_cs = measure(
                        logits, target, vocab_start_index, vocab_end_index, cs, tp_group, reps=BENCH_REPS
                    )
                    results[cs] = (t_cs, mem_cs)
                except torch.OutOfMemoryError:
                    results[cs] = (None, None)
                finally:
                    torch.cuda.empty_cache()  # isolate between chunk sizes

            t_baseline, mem_baseline = results[None]

            # ----- print one row per chunk_size -----
            for cs in CHUNK_SIZES:
                cs_label = "None" if cs is None else str(cs)
                t_cs, mem_cs = results[cs]

                if t_cs is None:
                    if dist.get_rank() == 0:
                        print(
                            f"{vocab_size:>10,}  "
                            f"{seq_len:>10,}  "
                            f"{cs_label:>10}  "
                            f"{'OOM':>{col_w}}  "
                            f"{'OOM':>{col_w}}  "
                            f"{'OOM':>{col_w}}  "
                            f"{'OOM':>{col_w}}"
                        )
                    continue

                mem_cs_mb = mem_cs / (1024**2)
                if t_baseline is not None and t_cs > 0:
                    speedup_str = f"{t_baseline / t_cs:>{col_w}.2f}x"
                    mem_saved_str = f"{mem_baseline / (1024**2) - mem_cs_mb:>{col_w}.0f}"
                else:
                    speedup_str = f"{'N/A':>{col_w}}"
                    mem_saved_str = f"{'N/A':>{col_w}}"

                if dist.get_rank() == 0:
                    print(
                        f"{vocab_size:>10,}  "
                        f"{seq_len:>10,}  "
                        f"{cs_label:>10}  "
                        f"{t_cs:>{col_w}.1f}  "
                        f"{mem_cs_mb:>{col_w}.0f}  "
                        f"{speedup_str}  "
                        f"{mem_saved_str}"
                    )

            if dist.get_rank() == 0:
                print(sep)

            # Free memory before next seq_len
            del logits, target
            torch.cuda.empty_cache()

    if dist.get_rank() == 0:
        print("\nAll times are mean wall-clock (ms) over forward+backward passes.")
        print("vs no-chunk: speedup relative to chunk_size=None (>1 = faster).")
        print("chunk_size=None uses DistributedLogprob; all others use ChunkedDistributedLogprob.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
