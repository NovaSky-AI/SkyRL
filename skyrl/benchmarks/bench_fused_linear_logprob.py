"""
Benchmark: LM-head training signal — baseline vs fused LM-head paths.

SkyRL's Megatron backend can turn decoder hidden states + the
output-layer weight into the per-token cross-entropy / log-prob used by the
RL/SFT loss (all run forward+backward so grads flow to hidden *and* weight):

  baseline : logits = hidden @ weightᵀ, then megatron-core's eager
             ``vocab_parallel_cross_entropy`` (materializes [B,S,vocab//TP]
             logits + an fp32 grad of the same shape).
  nvidia   : same logits, then megatron-core's ``fused_vocab_parallel_cross_entropy``
             (the @jit_fuser / TorchScript "NVIDIA stack" fused CE — fuses the
             softmax/CE *stages* but still materializes the logits).
  liger    : FusedLinearChunkedDistributedLogprob — folds the projection into the
             chunked, TP-parallel log-prob so the logits are *never* materialized.
  triton   : FusedLinearLogprobTriton — the Triton backend for the same log-prob
             contract.

Takeaway the table makes concrete: the NVIDIA fused CE is a *compute* optimization
(faster than eager) but not a *memory* one — like the baseline it materializes the
logits and OOMs at long context. The fused LM-head paths are the *memory*
optimization: they fit long context (at a modest compute cost), which the NVIDIA
path cannot.

Usage (single GPU; per-rank vocab shard = `vocab`):
    uv run --isolated --extra megatron torchrun --nproc_per_node=1 \\
        skyrl/benchmarks/bench_fused_linear_logprob.py
Run with --nproc_per_node=N for a real TP=N measurement (pass --vocab as the full
vocab; each rank then holds vocab//N).

Use ``--autotune-sweep --output-dir PATH`` to benchmark the Triton mainloop's
schedule search over representative model, token, TP, and CP shapes. With eight
processes, each GPU independently receives one eighth of the unique local shapes.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

# Qwen3.6-35B-A3B: hidden=2048, vocab=248320. Defaults show the per-rank shard
# for TP=4 (62080) and the full vocab (248320).
HIDDEN = 2048
VOCAB_SHARDS = [62080, 248320]
SEQ_LENS = [8192, 32768, 65536, 131072, 262144]
CHUNK_SIZE = 1024
WARMUP_REPS = 1
BENCH_REPS = 3
MODES = ["baseline", "nvidia", "liger", "triton"]

AUTOTUNE_MODELS = {
    "nemotron-super-120b": (4096, 131072),
    "nemotron-ultra-550b": (8192, 131072),
    "glm-5.3-flash": (4096, 154880),
    "kimi-k3": (7168, 163840),
}
AUTOTUNE_TOTAL_TOKEN_COUNTS = (16384, 32768, 65536, 131072, 262144)
AUTOTUNE_SHARDING_SCHEMES = tuple((tp, cp) for tp in (1, 2, 4, 8) for cp in (1, 2, 4, 8) if tp * cp <= 8)
AUTOTUNE_VOCAB_PER_SPLIT = 1024
FIXED_STAGE3 = (128, 256, 32, 3, 8)
FIXED_STAGE5 = (128, 256, 32, 5, 8)


@dataclass(frozen=True)
class AutotuneLocalShape:
    model: str
    hidden_size: int
    vocab_size: int
    num_tokens: int


def _vocab_bounds(vocab_local, tp_group):
    tp_rank = dist.get_rank(tp_group)
    vocab_start = tp_rank * vocab_local
    return vocab_start, vocab_start + vocab_local


def _loss(mode, hidden, weight, target, vocab_local, chunk_size, tp_group):
    """Per-token CE summed to a scalar (so all modes produce identical grads)."""
    from megatron.core.fusions.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        FusedLinearChunkedDistributedLogprob,
    )

    vocab_start, vocab_end = _vocab_bounds(vocab_local, tp_group)
    if mode == "liger":
        lp = FusedLinearChunkedDistributedLogprob.apply(
            hidden, weight, target, vocab_start, vocab_end, chunk_size, tp_group, False
        )
        return (-lp).sum()
    if mode == "triton":
        from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
            FusedLinearLogprobTriton,
        )

        lp = FusedLinearLogprobTriton.apply(hidden, weight, target, vocab_start, vocab_end, chunk_size, tp_group, False)
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
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    for repetition in range(reps):
        hidden = weight = target = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            shared_generator = torch.Generator(device=device).manual_seed(1000 + repetition)
            weight_generator = torch.Generator(device=device).manual_seed(2000 + tp_rank + repetition * tp_size)
            hidden = torch.randn(
                1,
                seq_len,
                HIDDEN,
                dtype=torch.bfloat16,
                device=device,
                generator=shared_generator,
                requires_grad=True,
            )
            weight = (
                torch.randn(
                    vocab_local,
                    HIDDEN,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=weight_generator,
                )
                * (HIDDEN**-0.5)
            ).requires_grad_(True)
            target = torch.randint(
                0,
                vocab_local * tp_size,
                (1, seq_len),
                device=device,
                generator=shared_generator,
            )
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


def _active_modes():
    modes = list(MODES)
    from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
        TRITON_AVAILABLE,
        is_cuda_available,
    )

    if not (TRITON_AVAILABLE and is_cuda_available):
        modes.remove("triton")
    return modes


def _correctness(modes, vocab_local, tp_group, device):
    """Validate loss and both gradients on every TP rank before timing."""
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)
    S = 64
    shared_generator = torch.Generator(device=device).manual_seed(0)
    weight_generator = torch.Generator(device=device).manual_seed(100 + tp_rank)
    h0 = torch.randn(1, S, HIDDEN, dtype=torch.bfloat16, device=device, generator=shared_generator)
    w0 = torch.randn(
        vocab_local,
        HIDDEN,
        dtype=torch.bfloat16,
        device=device,
        generator=weight_generator,
    ) * (HIDDEN**-0.5)
    tgt = torch.randint(
        0,
        vocab_local * tp_size,
        (1, S),
        device=device,
        generator=shared_generator,
    )
    ref = {}
    for mode in modes:
        h = h0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        loss = _loss(mode, h, w, tgt, vocab_local, CHUNK_SIZE, tp_group)
        loss.backward()
        ref[mode] = (
            loss.detach().float(),
            h.grad.detach().float(),
            w.grad.detach().float(),
        )
    base_loss, base_gh, base_gw = ref["baseline"]
    out = []
    for mode in modes:
        loss, gh, gw = ref[mode]
        loss_delta = (loss - base_loss).abs().item()
        gh_delta = (gh - base_gh).abs().max().item()
        gw_delta = (gw - base_gw).abs().max().item()
        matches = (
            torch.allclose(loss, base_loss, atol=2e-2, rtol=2e-2)
            and torch.allclose(gh, base_gh, atol=2e-2, rtol=2e-2)
            and torch.allclose(gw, base_gw, atol=2e-2, rtol=2e-2)
        )
        if not matches:
            raise RuntimeError(
                f"{mode} failed TP correctness on rank {tp_rank}: "
                f"loss_delta={loss_delta:.2e}, hidden_grad_delta={gh_delta:.2e}, "
                f"weight_grad_delta={gw_delta:.2e}"
            )
        out.append(
            f"{mode}: max|loss, dgrad_hidden, dgrad_weight deltas|=" f"{loss_delta:.2e}, {gh_delta:.2e}, {gw_delta:.2e}"
        )
    return out


def _load_kernel_module():
    path = Path(__file__).parents[1] / "backends/skyrl_train/distributed/megatron/fused_linear_logprob_triton.py"
    name = "skyrl_fused_linear_logprob_triton_benchmark"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_triton():
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError("--autotune-sweep requires Triton") from exc
    return triton


def _config_dict(spec: tuple[int, int, int, int, int]) -> dict[str, int]:
    block_m, block_n, block_k, num_stages, num_warps = spec
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_stages": num_stages,
        "num_warps": num_warps,
    }


def _triton_config(spec: tuple[int, int, int, int, int]):
    triton = _require_triton()
    values = _config_dict(spec)
    return triton.Config(
        {
            "BLOCK_SIZE_M": values["block_m"],
            "BLOCK_SIZE_N": values["block_n"],
            "BLOCK_SIZE_K": values["block_k"],
        },
        num_stages=values["num_stages"],
        num_warps=values["num_warps"],
    )


def _spec_from_config(config) -> tuple[int, int, int, int, int]:
    return (
        config.kwargs["BLOCK_SIZE_M"],
        config.kwargs["BLOCK_SIZE_N"],
        config.kwargs["BLOCK_SIZE_K"],
        config.num_stages,
        config.num_warps,
    )


def _logical_autotune_matrix() -> tuple[dict[str, Any], ...]:
    cases = []
    for model, (hidden_size, full_vocab_size) in AUTOTUNE_MODELS.items():
        for total_tokens in AUTOTUNE_TOTAL_TOKEN_COUNTS:
            for tp, cp in AUTOTUNE_SHARDING_SCHEMES:
                cases.append(
                    {
                        "model": model,
                        "hidden_size": hidden_size,
                        "full_vocab_size": full_vocab_size,
                        "total_tokens": total_tokens,
                        "tp": tp,
                        "cp": cp,
                        "local_shape": AutotuneLocalShape(
                            model=model,
                            hidden_size=hidden_size,
                            vocab_size=full_vocab_size // tp,
                            num_tokens=total_tokens // cp,
                        ),
                    }
                )
    return tuple(cases)


def _allocate_autotune_buffers(shape: AutotuneLocalShape, device: torch.device):
    num_splits = math.ceil(shape.vocab_size / AUTOTUNE_VOCAB_PER_SPLIT)
    return {
        "num_splits": num_splits,
        "hidden": torch.zeros((shape.num_tokens, shape.hidden_size), dtype=torch.bfloat16, device=device),
        "weight": torch.zeros((shape.vocab_size, shape.hidden_size), dtype=torch.bfloat16, device=device),
        "labels": torch.zeros((shape.num_tokens,), dtype=torch.int64, device=device),
        "maximum": torch.empty((shape.num_tokens, num_splits), dtype=torch.float32, device=device),
        "accumulate": torch.empty((shape.num_tokens, num_splits), dtype=torch.float32, device=device),
        "entropy_b": torch.empty((shape.num_tokens, num_splits), dtype=torch.float32, device=device),
        "logprobs": torch.empty((shape.num_tokens,), dtype=torch.float32, device=device),
    }


def _kernel_args(module, shape: AutotuneLocalShape, buffers):
    hidden = buffers["hidden"]
    weight = buffers["weight"]
    labels = buffers["labels"]
    maximum = buffers["maximum"]
    accumulate = buffers["accumulate"]
    entropy_b = buffers["entropy_b"]
    logprobs = buffers["logprobs"]
    return (
        0,
        hidden,
        weight,
        labels,
        shape.num_tokens,
        module._autotune_token_bucket(shape.num_tokens),
        shape.hidden_size,
        shape.vocab_size,
        AUTOTUNE_VOCAB_PER_SPLIT,
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        maximum,
        maximum.stride(0),
        maximum.stride(1),
        accumulate,
        accumulate.stride(0),
        accumulate.stride(1),
        entropy_b,
        entropy_b.stride(0),
        entropy_b.stride(1),
        logprobs,
        logprobs.stride(0),
        1.0,
    )


def _measure_cuda(launch: Callable[[], Any], warmup: int, repetitions: int) -> float:
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _raw_launch(module, shape: AutotuneLocalShape, buffers, spec):
    triton = _require_triton()
    block_m, block_n, block_k, num_stages, num_warps = spec
    grid = (triton.cdiv(shape.num_tokens, block_m) * buffers["num_splits"],)
    module.efficient_entropy_kernel_general_mainloop.fn[grid](
        *_kernel_args(module, shape, buffers),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        USE_TMA=module.SUPPORT_CUDA_TMA,
        INPUT_PRECISION="tf32",
        num_stages=num_stages,
        num_warps=num_warps,
    )


def _autotuned_launch(module, shape: AutotuneLocalShape, buffers):
    triton = _require_triton()

    def grid(meta):
        return (triton.cdiv(shape.num_tokens, meta["BLOCK_SIZE_M"]) * buffers["num_splits"],)

    module.efficient_entropy_kernel_general_mainloop[grid](
        *_kernel_args(module, shape, buffers),
        USE_TMA=module.SUPPORT_CUDA_TMA,
        INPUT_PRECISION="tf32",
    )


def _measure_full_forward(module, shape, buffers, spec, repetitions):
    autotuner = module.efficient_entropy_kernel_general_mainloop
    original_configs, original_cache = autotuner.configs, autotuner.cache
    autotuner.configs, autotuner.cache = [_triton_config(spec)], {}
    try:
        return _measure_cuda(
            lambda: module.efficient_entropy_forward(
                buffers["hidden"],
                buffers["weight"],
                buffers["labels"],
                dist_process_group=None,
            ),
            warmup=1,
            repetitions=repetitions,
        )
    finally:
        autotuner.configs, autotuner.cache = original_configs, original_cache


def _benchmark_autotune_shape(
    module,
    shape: AutotuneLocalShape,
    device: torch.device,
    repetitions: int,
):
    buffers = _allocate_autotune_buffers(shape, device)
    autotuner = module.efficient_entropy_kernel_general_mainloop
    torch.cuda.synchronize()
    started = time.perf_counter()
    _autotuned_launch(module, shape, buffers)
    torch.cuda.synchronize()
    first_autotuned_call_ms = (time.perf_counter() - started) * 1000.0
    selected_spec = _spec_from_config(autotuner.best_config)
    cached_autotuned_ms = _measure_cuda(
        lambda: _autotuned_launch(module, shape, buffers),
        warmup=1,
        repetitions=repetitions,
    )

    timings = {}
    failures = {}
    for spec in module._FORWARD_MAINLOOP_CONFIG_SPECS:
        try:
            timings[spec] = _measure_cuda(
                lambda spec=spec: _raw_launch(module, shape, buffers, spec),
                warmup=1,
                repetitions=repetitions,
            )
        except Exception as exc:  # Invalid-resource configs are reported here.
            failures[str(spec)] = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()

    best_spec = min(timings, key=timings.__getitem__)
    full_repetitions = max(1, repetitions // 2)
    result = {
        "kind": "local_shape",
        "shape": asdict(shape),
        "token_bucket": module._autotune_token_bucket(shape.num_tokens),
        "selected_config": _config_dict(selected_spec),
        "measured_best_config": _config_dict(best_spec),
        "first_autotuned_call_ms": first_autotuned_call_ms,
        "cached_autotuned_ms": cached_autotuned_ms,
        "fixed_stage3_ms": timings[FIXED_STAGE3],
        "fixed_stage5_ms": timings[FIXED_STAGE5],
        "measured_best_ms": timings[best_spec],
        "full_fixed_stage5_ms": _measure_full_forward(module, shape, buffers, FIXED_STAGE5, full_repetitions),
        "full_measured_best_ms": _measure_full_forward(module, shape, buffers, best_spec, full_repetitions),
        "all_config_ms": {json.dumps(_config_dict(spec), sort_keys=True): value for spec, value in timings.items()},
        "failures": failures,
    }
    torch.cuda.empty_cache()
    return result


def _recompile_probe(module, device: torch.device):
    records = []
    for num_tokens in (20000, 20001, 40000):
        shape = AutotuneLocalShape("specialization-probe", 4096, 16000, num_tokens)
        buffers = _allocate_autotune_buffers(shape, device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        _autotuned_launch(module, shape, buffers)
        torch.cuda.synchronize()
        best_config = module.efficient_entropy_kernel_general_mainloop.best_config
        records.append(
            {
                "num_tokens": num_tokens,
                "bucket": module._autotune_token_bucket(num_tokens),
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "selected_config": _config_dict(_spec_from_config(best_config)),
            }
        )
        torch.cuda.empty_cache()
    return {"kind": "specialization_probe", "calls": records}


def _run_autotune_sweep(output_dir: Path, repetitions: int) -> None:
    triton = _require_triton()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    module = _load_kernel_module()

    logical = _logical_autotune_matrix()
    shapes = tuple(dict.fromkeys(case["local_shape"] for case in logical))
    assigned = tuple(shape for index, shape in enumerate(shapes) if index % world_size == rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rank-{rank}.jsonl"
    metadata = {
        "kind": "metadata",
        "rank": rank,
        "world_size": world_size,
        "device": torch.cuda.get_device_name(device),
        "device_capability": torch.cuda.get_device_capability(device),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "kernel_file": module.__file__,
        "support_cuda_tma": module.SUPPORT_CUDA_TMA,
        "logical_case_count": len(logical),
        "unique_local_shape_count": len(shapes),
        "assigned_shape_count": len(assigned),
    }
    with output_path.open("w") as output:
        print(json.dumps(metadata, sort_keys=True), file=output, flush=True)
        if rank == 0:
            print(
                json.dumps(_recompile_probe(module, device), sort_keys=True),
                file=output,
                flush=True,
            )
        for index, shape in enumerate(assigned, start=1):
            result = _benchmark_autotune_shape(module, shape, device, repetitions)
            print(json.dumps(result, sort_keys=True), file=output, flush=True)
            print(
                f"rank={rank} shape={index}/{len(assigned)} model={shape.model} "
                f"M={shape.num_tokens} H={shape.hidden_size} V={shape.vocab_size} "
                f"stage5={result['fixed_stage5_ms']:.3f}ms "
                f"best={result['measured_best_ms']:.3f}ms",
                flush=True,
            )


def _run_backend_comparison(args) -> None:

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
    modes = _active_modes()
    if rank0:
        print(
            f"Device {torch.cuda.get_device_name(device)} | TP(world)={world} | hidden={HIDDEN} | chunk={args.chunk_size}"
        )
        print(
            "baseline = megatron vocab_parallel_cross_entropy (eager) | "
            "nvidia = fused_vocab_parallel_cross_entropy (@jit_fuser) | "
            "liger = FusedLinearChunkedDistributedLogprob (no logits) | "
            "triton = FusedLinearLogprobTriton (no logits)"
        )
        if "triton" not in modes:
            print("triton mode unavailable; skipping")
        print("all: hidden+weight -> per-token CE -> sum -> backward\n")
        print("correctness (TP=%d, vocab=%d):" % (world, vocab_shards[0] * world))
    correctness = _correctness(modes, vocab_shards[0], tp_group, device)
    if rank0:
        for line in correctness:
            print("  " + line)
        print()

    cw = 12
    for vlocal in vocab_shards:
        if rank0:
            print(f"=== per-rank vocab shard = {vlocal:,} ===")
            hdr = (
                f"{'seq_len':>9} |"
                + "".join(f" {m + ' MB':>{cw}} |" for m in modes)
                + "".join(f" {m + ' ms':>{cw}} |" for m in modes)
            )
            print(hdr)
            print("-" * len(hdr))
        for s in SEQ_LENS:
            res = {}
            for mode in modes:
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
                    + "".join(f" {mb(m):>{cw}} |" for m in modes)
                    + "".join(f" {ms(m):>{cw}} |" for m in modes)
                )
                print(row)
        if rank0:
            print()

    dist.destroy_process_group()


def _entropy_specialization_call(
    module,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    grad_logprobs: torch.Tensor,
    tp_group,
    *,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logprobs, _entropy, maximum, accumulate, entropy_b = module.efficient_entropy_forward(
        hidden,
        weight,
        labels,
        dist_process_group=tp_group,
        compute_entropy=compute_entropy,
    )
    dentropy = torch.zeros_like(grad_logprobs) if compute_entropy else None
    d_hidden, d_weight = module.efficient_entropy_backward(
        grad_logprobs,
        dentropy,
        hidden,
        weight,
        labels,
        maximum,
        accumulate,
        entropy_b,
        dist_process_group=tp_group,
        compute_entropy=compute_entropy,
    )
    return logprobs, d_hidden, d_weight


def _run_entropy_specialization_probe(args) -> None:
    """Compare the general entropy path with the dense log-prob-only path."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    import megatron.core.parallel_state as mpu

    world = dist.get_world_size()
    if world != args.tensor_parallel * args.context_parallel:
        raise ValueError(
            f"torchrun world size {world} must equal TP*CP=" f"{args.tensor_parallel * args.context_parallel}"
        )
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_parallel,
        context_parallel_size=args.context_parallel,
    )
    module = _load_kernel_module()
    tp_group = mpu.get_tensor_model_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    device = torch.device("cuda", local_rank)
    rank0 = dist.get_rank() == 0

    check_tokens, check_hidden, check_vocab_local = 257, 256, 2048
    shared_generator = torch.Generator(device=device).manual_seed(4700 + cp_rank)
    weight_generator = torch.Generator(device=device).manual_seed(4800 + dist.get_rank())
    check_hidden_tensor = torch.randn(
        check_tokens,
        check_hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )
    check_weight = torch.randn(
        check_vocab_local,
        check_hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=weight_generator,
    ) * (check_hidden**-0.5)
    check_labels = torch.randint(
        0,
        check_vocab_local * args.tensor_parallel,
        (check_tokens,),
        device=device,
        generator=shared_generator,
    )
    check_grad = torch.linspace(0.5, 1.5, check_tokens, dtype=torch.float32, device=device)
    with_entropy = _entropy_specialization_call(
        module,
        check_hidden_tensor,
        check_weight,
        check_labels,
        check_grad,
        tp_group,
        compute_entropy=True,
    )
    logprob_only = _entropy_specialization_call(
        module,
        check_hidden_tensor,
        check_weight,
        check_labels,
        check_grad,
        tp_group,
        compute_entropy=False,
    )
    deltas = []
    for actual, expected in zip(logprob_only, with_entropy):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        deltas.append((actual.float() - expected.float()).abs().max().item())
    if rank0:
        print(
            "ENTROPY_SPECIALIZATION_CORRECTNESS="
            + json.dumps(
                {
                    "logprobs_max_abs": deltas[0],
                    "dhidden_max_abs": deltas[1],
                    "dweight_max_abs": deltas[2],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    vocab_local = args.vocab // args.tensor_parallel
    weight = torch.randn(
        vocab_local,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=weight_generator,
    ) * (args.hidden**-0.5)
    for total_tokens in args.seq_lens:
        if total_tokens % args.context_parallel:
            raise ValueError(f"seq_len={total_tokens} must be divisible by CP={args.context_parallel}")
        local_tokens = total_tokens // args.context_parallel
        hidden = torch.randn(
            local_tokens,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_generator,
        )
        labels = torch.randint(
            0,
            args.vocab,
            (local_tokens,),
            device=device,
            generator=shared_generator,
        )
        grad_logprobs = torch.linspace(0.5, 1.5, local_tokens, dtype=torch.float32, device=device)
        launch = functools.partial(
            _entropy_specialization_call,
            module,
            hidden,
            weight,
            labels,
            grad_logprobs,
            tp_group,
        )

        variants = {"entropy": True, "logprob-only": False}
        for compute_entropy in variants.values():
            launch(compute_entropy=compute_entropy)
        torch.cuda.synchronize(device)
        samples = {name: [] for name in variants}
        peaks = {name: [] for name in variants}
        ordered = list(variants)
        for repetition in range(args.repetitions):
            offset = repetition % len(ordered)
            for name in ordered[offset:] + ordered[:offset]:
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                launch(compute_entropy=variants[name])
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
                peaks[name].append(torch.cuda.max_memory_allocated(device))

        local_metrics = torch.tensor(
            [
                metric
                for name in ordered
                for metric in (statistics.median(samples[name]), statistics.median(peaks[name]))
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.reduce(local_metrics, dst=0, op=dist.ReduceOp.MAX)
        if rank0:
            reduced = local_metrics.tolist()
            metrics = {
                name: {"ms": reduced[index * 2], "peak_bytes": int(reduced[index * 2 + 1])}
                for index, name in enumerate(ordered)
            }
            metrics["speedup"] = metrics["entropy"]["ms"] / metrics["logprob-only"]["ms"]
            metrics["peak_saved_bytes"] = metrics["entropy"]["peak_bytes"] - metrics["logprob-only"]["peak_bytes"]
            print(
                "ENTROPY_SPECIALIZATION_RESULT="
                + json.dumps(
                    {
                        "total_tokens": total_tokens,
                        "local_tokens": local_tokens,
                        "hidden": args.hidden,
                        "vocab_local": vocab_local,
                        "tp": args.tensor_parallel,
                        "cp": args.context_parallel,
                        **metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="full vocab; per-rank shard = vocab // world_size",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--context-parallel", type=int, default=1)
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument(
        "--autotune-sweep",
        action="store_true",
        help="sweep Triton schedules over representative local shapes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="JSONL output directory required by --autotune-sweep",
    )
    parser.add_argument("--repetitions", type=int, default=BENCH_REPS)
    parser.add_argument(
        "--entropy-specialization-probe",
        action="store_true",
        help="compare entropy and dense log-prob-only kernel paths",
    )
    args = parser.parse_args()
    if args.entropy_specialization_probe:
        if args.vocab is None:
            parser.error("--entropy-specialization-probe requires --vocab")
        _run_entropy_specialization_probe(args)
    elif args.autotune_sweep:
        if args.output_dir is None:
            parser.error("--autotune-sweep requires --output-dir")
        _run_autotune_sweep(args.output_dir, args.repetitions)
    else:
        _run_backend_comparison(args)


if __name__ == "__main__":
    main()
