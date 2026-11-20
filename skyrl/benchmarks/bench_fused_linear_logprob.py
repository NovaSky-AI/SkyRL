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
schedule search over production model, token, TP, and CP shapes. With eight
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
BLOCK_MASK_ACTIVE_FRACTIONS = (1.0, 0.95, 0.75, 0.66, 0.25)

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


def _loss(
    mode,
    hidden,
    weight,
    target,
    vocab_local,
    chunk_size,
    tp_group,
    active_mask=None,
    mask_in_kernel=True,
    active_spans=None,
):
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

        args = (
            hidden,
            weight,
            target,
            vocab_start,
            vocab_end,
            chunk_size,
            tp_group,
            False,
        )
        kernel_mask = active_mask if mask_in_kernel else None
        if active_spans is not None:
            lp = FusedLinearLogprobTriton.apply(*args, kernel_mask, active_spans)
        elif kernel_mask is not None:
            lp = FusedLinearLogprobTriton.apply(*args, kernel_mask)
        else:
            lp = FusedLinearLogprobTriton.apply(*args)
        if active_mask is not None and not mask_in_kernel:
            return -(lp * active_mask).sum()
        return -lp.sum()
    logits = torch.matmul(hidden, weight.t())  # [B, S, vocab//TP]
    if mode == "baseline":
        ce = vocab_parallel_cross_entropy(logits, target, 0.0, tp_group)
    elif mode == "nvidia":
        ce = fused_vocab_parallel_cross_entropy(logits, target, tp_group)
    else:
        raise ValueError(mode)
    return ce.sum()


def _measure(
    mode,
    seq_len,
    vocab_local,
    chunk_size,
    tp_group,
    device,
    reps,
    active_fraction=None,
    hidden_size=HIDDEN,
    mask_in_kernel=True,
):
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
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
                generator=shared_generator,
                requires_grad=True,
            )
            weight = (
                torch.randn(
                    vocab_local,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=weight_generator,
                )
                * (hidden_size**-0.5)
            ).requires_grad_(True)
            target = torch.randint(
                0,
                vocab_local * tp_size,
                (1, seq_len),
                device=device,
                generator=shared_generator,
            )
            active_mask = None
            if active_fraction is not None:
                active_tokens = math.ceil(seq_len * active_fraction)
                active_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                active_mask[:, seq_len - active_tokens :] = True
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            loss = _loss(
                mode,
                hidden,
                weight,
                target,
                vocab_local,
                chunk_size,
                tp_group,
                active_mask,
                mask_in_kernel,
            )
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
        HAS_ACTIVE_MASK=False,
        COMPUTE_ENTROPY=True,
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
        HAS_ACTIVE_MASK=False,
        COMPUTE_ENTROPY=True,
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


def _load_block_mask_trace(path: Path, dataset: str, num_tokens: int, tp: int, cp: int, cp_rank: int) -> torch.Tensor:
    """Rebuild an ordered CP-local mask trace without token data."""
    payload = json.loads(path.read_text())
    datasets = {item["dataset"]: item for item in payload["datasets"]}
    if dataset not in datasets:
        raise ValueError(f"Unknown mask trace {dataset!r}; choose from {sorted(datasets)}")

    records = []
    for encoded in datasets[dataset]["records"]:
        value = bool(encoded["starts_active"])
        mask = []
        for run in encoded["runs"]:
            mask.extend([value] * int(run))
            value = not value
        if len(mask) != int(encoded["length"]):
            raise ValueError(f"Malformed mask trace record with length {encoded['length']}")
        records.append(mask)

    # Match production first-fit-decreasing segment ordering. Segment padding
    # is part of the kernel mask; hidden rows themselves remain untouched.
    bins: list[list[list[bool]]] = []
    remaining: list[int] = []
    for mask in sorted(records, key=len, reverse=True):
        for index, capacity in enumerate(remaining):
            if capacity >= len(mask):
                bins[index].append(mask)
                remaining[index] -= len(mask)
                break
        else:
            bins.append([mask])
            remaining.append(32768 - len(mask))

    alignment = tp if cp == 1 else tp * cp * 2
    # Build one full local trace period. Taking the first N rows is badly biased
    # for short benchmark windows because FFD puts the longest records first.
    # Instead, start at the real record boundary whose cyclic window most closely
    # matches this CP rank's full-trace density. This preserves run lengths and
    # natural row order without sorting or moving rows in the measured kernel.
    local_stream: list[bool] = []
    record_starts: list[int] = []
    for packed_bin in bins:
        for mask in packed_bin:
            record_starts.append(len(local_stream))
            padded_length = math.ceil(len(mask) / alignment) * alignment
            padded = mask + [False] * (padded_length - len(mask))
            if cp == 1:
                local_stream.extend(padded)
            else:
                chunk = padded_length // (2 * cp)
                local_stream.extend(padded[cp_rank * chunk : (cp_rank + 1) * chunk])
                mirrored = 2 * cp - cp_rank - 1
                local_stream.extend(padded[mirrored * chunk : (mirrored + 1) * chunk])

    local_tokens = num_tokens // cp
    repeats = math.ceil((len(local_stream) + local_tokens) / len(local_stream))
    cyclic_stream = local_stream * repeats
    prefix = [0]
    for active in cyclic_stream:
        prefix.append(prefix[-1] + int(active))
    target_fraction = sum(local_stream) / len(local_stream)
    start = min(
        record_starts,
        key=lambda offset: abs((prefix[offset + local_tokens] - prefix[offset]) / local_tokens - target_fraction),
    )
    window = cyclic_stream[start : start + local_tokens]
    return torch.tensor(window, dtype=torch.bool).unsqueeze(0)


def _active_spans(active_mask: torch.Tensor, max_inactive_gap: int) -> tuple[tuple[int, int], ...]:
    """Return ordered active spans, coalescing short zero gaps without moving rows."""
    flat_mask = active_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1).tolist()
    runs = []
    start = None
    for index, active in enumerate(flat_mask + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None

    spans = []
    for start, end in runs:
        if spans and start - spans[-1][1] <= max_inactive_gap:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return tuple(spans)


def _benchmark_narrow_dlogits_tail(
    args,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    tp_group,
) -> dict[str, Any]:
    """Compare the incumbent padded final split with direct narrow staging."""
    from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
        _OOV_LABEL_SENTINEL,
        efficient_entropy_backward,
        efficient_entropy_forward,
    )

    hidden_2d = hidden.detach().reshape(-1, hidden.shape[-1]).contiguous()
    weight_2d = weight.detach().contiguous()
    mask_1d = active_mask.reshape(-1).contiguous()
    target_1d = target.reshape(-1)
    vocab_start, vocab_end = _vocab_bounds(weight_2d.shape[0], tp_group)
    rank = dist.get_rank(tp_group)
    target_mask = (target_1d < vocab_start) | (target_1d >= vocab_end)
    labels = (target_1d - vocab_start) + rank * weight_2d.shape[0]
    labels = torch.where(target_mask, _OOV_LABEL_SENTINEL, labels).to(torch.int64).contiguous()
    _, _, maximum, accumulate, entropy_b = efficient_entropy_forward(
        hidden_2d,
        weight_2d,
        labels,
        1.0,
        tp_group,
        mask_1d,
    )
    dlogprobs = torch.ones(hidden_2d.shape[0], dtype=torch.float32, device=hidden.device)
    dentropy = torch.zeros_like(dlogprobs)
    spans = _active_spans(mask_1d, 128)

    def launch(narrow: bool):
        return efficient_entropy_backward(
            dlogprobs,
            dentropy,
            hidden_2d,
            weight_2d,
            labels,
            maximum,
            accumulate,
            entropy_b,
            False,
            1.0,
            tp_group,
            mask_1d,
            spans,
            narrow,
        )

    outputs = {name: launch(narrow) for name, narrow in (("wide-tail", False), ("narrow-tail", True))}
    torch.cuda.synchronize(hidden.device)
    torch.testing.assert_close(outputs["narrow-tail"][0], outputs["wide-tail"][0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(outputs["narrow-tail"][1], outputs["wide-tail"][1], atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(outputs["narrow-tail"][0][~mask_1d]) == 0

    samples = {name: [] for name in outputs}
    peaks = {name: [] for name in outputs}
    variants = list(outputs)
    for repetition in range(args.repetitions):
        offset = repetition % len(variants)
        for name in variants[offset:] + variants[:offset]:
            torch.cuda.reset_peak_memory_stats(hidden.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            launch(name == "narrow-tail")
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end))
            peaks[name].append(torch.cuda.max_memory_allocated(hidden.device))

    return {
        "tail_width": weight_2d.shape[0] % 9504 or 9504,
        "span_count": len(spans),
        "metrics": {
            name: {
                "ms": statistics.median(values),
                "peak_bytes": statistics.median(peaks[name]),
                "dhidden_max_abs_vs_wide": (outputs[name][0].float() - outputs["wide-tail"][0].float())
                .abs()
                .max()
                .item(),
                "dweight_max_abs_vs_wide": (outputs[name][1].float() - outputs["wide-tail"][1].float())
                .abs()
                .max()
                .item(),
            }
            for name, values in samples.items()
        },
    }


def _run_block_mask_probe(args) -> None:
    """Measure the end-to-end value and dense-call overhead of tile skipping."""
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
    tp_group = mpu.get_tensor_model_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    device = torch.device("cuda", local_rank)
    rank0 = dist.get_rank() == 0
    vocab_local = (args.vocab or VOCAB_SHARDS[-1]) // args.tensor_parallel

    # Compare the masked kernel against the existing dense kernel with the same
    # mask applied to its output. This checks the scalar loss and both gradients
    # before any performance result is accepted.
    from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
        FusedLinearLogprobTriton,
    )

    check_tokens = 257
    check_hidden = 256
    # Cross the 9,504-column backward split boundary and leave a partial tail.
    check_vocab_local = 10_000
    shared_generator = torch.Generator(device=device).manual_seed(761)
    weight_generator = torch.Generator(device=device).manual_seed(991 + dist.get_rank(tp_group))
    hidden_base = torch.randn(
        1,
        check_tokens,
        check_hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )
    weight_base = torch.randn(
        check_vocab_local,
        check_hidden,
        dtype=torch.bfloat16,
        device=device,
        generator=weight_generator,
    ) * (check_hidden**-0.5)
    target = torch.randint(
        0,
        check_vocab_local * args.tensor_parallel,
        (1, check_tokens),
        device=device,
        generator=shared_generator,
    )
    vocab_start, vocab_end = _vocab_bounds(check_vocab_local, tp_group)
    blocky66 = torch.arange(check_tokens, device=device).unsqueeze(0) >= 55
    # Gap-128 deliberately retains this hole inside one dHidden GEMM span.
    blocky66[:, 128:160] = False
    masks = {
        "all": torch.ones((1, check_tokens), dtype=torch.bool, device=device),
        "blocky95": torch.arange(check_tokens, device=device).unsqueeze(0) >= 13,
        "blocky66": blocky66,
        "blocky25": torch.arange(check_tokens, device=device).unsqueeze(0) >= 193,
        "none": torch.zeros((1, check_tokens), dtype=torch.bool, device=device),
    }
    correctness = []
    fused_args = (
        target,
        vocab_start,
        vocab_end,
        check_tokens,
        tp_group,
        False,
    )
    span_gaps = args.block_sparse_backward_span_gaps or []
    for mask_name, mask in masks.items():
        dense_hidden = hidden_base.clone().requires_grad_(True)
        dense_weight = weight_base.clone().requires_grad_(True)
        dense_logprobs = FusedLinearLogprobTriton.apply(dense_hidden, dense_weight, *fused_args)
        dense_loss = -(dense_logprobs * mask).sum()
        dense_loss.backward()

        masked_hidden = hidden_base.clone().requires_grad_(True)
        masked_weight = weight_base.clone().requires_grad_(True)
        masked_logprobs = FusedLinearLogprobTriton.apply(masked_hidden, masked_weight, *fused_args, mask)
        masked_loss = -masked_logprobs.sum()
        masked_loss.backward()

        expected_logprobs = dense_logprobs.masked_fill(~mask, 0.0)
        logprob_delta = (masked_logprobs - expected_logprobs).abs().max().item()
        loss_delta = (masked_loss - dense_loss).abs().item()
        hidden_delta = (masked_hidden.grad - dense_hidden.grad).abs().max().item()
        weight_delta = (masked_weight.grad - dense_weight.grad).abs().max().item()
        torch.testing.assert_close(masked_logprobs, expected_logprobs, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(masked_loss, dense_loss, atol=1e-4, rtol=1e-6)
        torch.testing.assert_close(masked_hidden.grad, dense_hidden.grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(masked_weight.grad, dense_weight.grad, atol=2e-2, rtol=2e-2)
        correctness.append(
            {
                "mask": mask_name,
                "logprob_max_abs": logprob_delta,
                "loss_max_abs": loss_delta,
                "hidden_grad_max_abs": hidden_delta,
                "weight_grad_max_abs": weight_delta,
            }
        )
        for gap in span_gaps:
            span_hidden = hidden_base.clone().requires_grad_(True)
            span_weight = weight_base.clone().requires_grad_(True)
            spans = _active_spans(mask, gap)
            span_logprobs = FusedLinearLogprobTriton.apply(
                span_hidden,
                span_weight,
                *fused_args,
                mask,
                spans,
            )
            span_loss = -span_logprobs.sum()
            span_loss.backward()
            torch.testing.assert_close(span_logprobs, expected_logprobs, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(span_loss, dense_loss, atol=1e-4, rtol=1e-6)
            torch.testing.assert_close(span_hidden.grad, dense_hidden.grad, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(span_weight.grad, dense_weight.grad, atol=2e-2, rtol=2e-2)
            assert torch.count_nonzero(span_logprobs[~mask]) == 0
            assert torch.count_nonzero(span_hidden.grad[~mask]) == 0
            correctness.append(
                {
                    "mask": f"{mask_name}-span-gap-{gap}",
                    "logprob_max_abs": (span_logprobs - expected_logprobs).abs().max().item(),
                    "loss_max_abs": (span_loss - dense_loss).abs().item(),
                    "hidden_grad_max_abs": (span_hidden.grad - dense_hidden.grad).abs().max().item(),
                    "weight_grad_max_abs": (span_weight.grad - dense_weight.grad).abs().max().item(),
                }
            )

    if rank0:
        print(
            "BLOCK_MASK_CORRECTNESS=" + json.dumps(correctness, sort_keys=True),
            flush=True,
        )

    if rank0:
        print(
            f"block-mask probe | {torch.cuda.get_device_name(device)} "
            f"| TP={args.tensor_parallel} CP={args.context_parallel} "
            f"| hidden={args.hidden} | vocab/rank={vocab_local}"
        )
        print("synthetic masks are contiguous active suffixes; trace masks preserve real run structure")

    for seq_len in args.seq_lens:
        shared_generator = torch.Generator(device=device).manual_seed(1000 + seq_len)
        weight_generator = torch.Generator(device=device).manual_seed(2000 + dist.get_rank(tp_group))
        weight = (
            torch.randn(
                vocab_local,
                args.hidden,
                dtype=torch.bfloat16,
                device=device,
                generator=weight_generator,
            )
            * (args.hidden**-0.5)
        ).requires_grad_(True)
        local_seq_len = seq_len // args.context_parallel
        if local_seq_len * args.context_parallel != seq_len:
            raise ValueError(f"seq_len={seq_len} must be divisible by CP={args.context_parallel}")
        hidden = torch.randn(
            1,
            local_seq_len,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_generator,
            requires_grad=True,
        )
        target = torch.randint(
            0,
            vocab_local * args.tensor_parallel,
            (1, local_seq_len),
            device=device,
            generator=shared_generator,
        )

        mask_cases = []
        if not args.block_mask_trace_only:
            for fraction in BLOCK_MASK_ACTIVE_FRACTIONS:
                active_tokens = math.ceil(seq_len * fraction)
                global_active_mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
                global_active_mask[:, seq_len - active_tokens :] = True
                if args.context_parallel > 1:
                    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
                        _get_tokens_on_this_cp_rank,
                    )

                    active_mask = _get_tokens_on_this_cp_rank(
                        global_active_mask,
                        cp_rank,
                        args.context_parallel,
                        seq_dim=1,
                    )
                else:
                    active_mask = global_active_mask
                mask_cases.append((f"suffix-{fraction:.0%}", fraction, active_mask))

        if args.mask_trace is not None:
            for trace_dataset in args.mask_trace_datasets:
                trace_mask = _load_block_mask_trace(
                    args.mask_trace,
                    trace_dataset,
                    seq_len,
                    args.tensor_parallel,
                    args.context_parallel,
                    cp_rank,
                ).to(device)
                trace_active = trace_mask.sum(dtype=torch.float64)
                dist.all_reduce(trace_active, op=dist.ReduceOp.SUM)
                trace_fraction = trace_active.item() / (trace_mask.numel() * world)
                mask_cases.append(
                    (
                        trace_dataset,
                        trace_fraction,
                        trace_mask,
                    )
                )

        for pattern, fraction, active_mask in mask_cases:

            span_variants = {f"span-gap-{gap}": _active_spans(active_mask, gap) for gap in span_gaps}

            def launch(variant, hidden_input, weight_input, target_input, mask_input):
                hidden_input.grad = None
                weight_input.grad = None
                mask_in_kernel = variant != "dense"
                loss = _loss(
                    "triton",
                    hidden_input,
                    weight_input,
                    target_input,
                    vocab_local,
                    args.chunk_size,
                    tp_group,
                    mask_input,
                    mask_in_kernel,
                    span_variants.get(variant),
                )
                loss.backward()

            variants = ["dense", "sparse", *span_variants]
            for variant in variants:
                launch(variant, hidden, weight, target, active_mask)
            torch.cuda.synchronize(device)
            samples = {variant: [] for variant in variants}
            peaks = {variant: [] for variant in variants}
            for repetition in range(args.repetitions):
                offset = repetition % len(variants)
                order = variants[offset:] + variants[:offset]
                for variant in order:
                    torch.cuda.reset_peak_memory_stats(device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch(variant, hidden, weight, target, active_mask)
                    end.record()
                    end.synchronize()
                    samples[variant].append(start.elapsed_time(end))
                    peaks[variant].append(torch.cuda.max_memory_allocated(device))

            local_metrics = torch.tensor(
                [
                    metric
                    for variant in variants
                    for metric in (
                        statistics.median(samples[variant]),
                        statistics.median(peaks[variant]),
                    )
                ],
                dtype=torch.float64,
                device=device,
            )
            # Training step latency is set by the slowest TP/CP rank.
            dist.reduce(local_metrics, dst=0, op=dist.ReduceOp.MAX)
            reduced = local_metrics.tolist()
            if rank0:
                metrics = {
                    variant: {"ms": reduced[index * 2], "peak": reduced[index * 2 + 1]}
                    for index, variant in enumerate(variants)
                }
                dense_ms = metrics["dense"]["ms"]
                summary = " ".join(
                    f"{variant}={values['ms']:.3f}ms/{dense_ms / values['ms']:.3f}x/"
                    f"{values['peak'] / 1024**2:.0f}MB"
                    for variant, values in metrics.items()
                )
                print(f"tokens={seq_len} pattern={pattern} active={fraction:>5.1%} {summary}")

            if args.narrow_dlogits_tail_probe:
                tail_result = _benchmark_narrow_dlogits_tail(
                    args,
                    hidden,
                    weight,
                    target,
                    active_mask,
                    tp_group,
                )
                tail_variants = tuple(tail_result["metrics"])
                local_tail_metrics = torch.tensor(
                    [tail_result["metrics"][variant]["ms"] for variant in tail_variants]
                    + [tail_result["metrics"][variant]["peak_bytes"] for variant in tail_variants]
                    + [tail_result["metrics"][variant]["dhidden_max_abs_vs_wide"] for variant in tail_variants]
                    + [tail_result["metrics"][variant]["dweight_max_abs_vs_wide"] for variant in tail_variants],
                    dtype=torch.float64,
                    device=device,
                )
                dist.reduce(local_tail_metrics, dst=0, op=dist.ReduceOp.MAX)
                if rank0:
                    values = local_tail_metrics.tolist()
                    variant_count = len(tail_variants)
                    wide_ms, narrow_ms = values[:variant_count]
                    wide_peak, narrow_peak = values[variant_count : 2 * variant_count]
                    print(
                        f"NARROW_DLOGITS_TAIL tokens={seq_len} pattern={pattern} "
                        f"active={fraction:>5.1%} width={tail_result['tail_width']} "
                        f"spans={tail_result['span_count']} "
                        f"wide={wide_ms:.3f}ms narrow={narrow_ms:.3f}ms "
                        f"speedup={wide_ms / narrow_ms:.4f}x "
                        f"peak_delta={(narrow_peak - wide_peak) / 1024**2:.1f}MiB "
                        f"dhidden_max_abs={max(values[2 * variant_count:3 * variant_count]):.4g} "
                        f"dweight_max_abs={max(values[3 * variant_count:]):.4g}",
                        flush=True,
                    )

        del hidden, weight, target
        torch.cuda.empty_cache()

    dist.destroy_process_group()


def _entropy_specialization_call(
    module,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    grad_logprobs: torch.Tensor,
    tp_group,
    active_mask: torch.Tensor | None,
    *,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logprobs, _entropy, maximum, accumulate, entropy_b = module.efficient_entropy_forward(
        hidden,
        weight,
        labels,
        dist_process_group=tp_group,
        active_mask=active_mask,
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
        active_mask=active_mask,
        compute_entropy=compute_entropy,
    )
    return logprobs, d_hidden, d_weight


def _run_entropy_specialization_probe(args) -> None:
    """Compare the general entropy path with SkyRL's logprob-only specialization."""
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
    rows = torch.arange(check_tokens, device=device)
    check_masks = {
        "dense": None,
        "blocky": ((rows >= 13) & (rows < 117)) | (rows >= 193),
    }
    correctness = []
    for name, active_mask in check_masks.items():
        with_entropy = _entropy_specialization_call(
            module,
            check_hidden_tensor,
            check_weight,
            check_labels,
            check_grad,
            tp_group,
            active_mask,
            compute_entropy=True,
        )
        logprob_only = _entropy_specialization_call(
            module,
            check_hidden_tensor,
            check_weight,
            check_labels,
            check_grad,
            tp_group,
            active_mask,
            compute_entropy=False,
        )
        deltas = []
        for actual, expected in zip(logprob_only, with_entropy):
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            deltas.append((actual.float() - expected.float()).abs().max().item())
        if active_mask is not None:
            torch.testing.assert_close(
                logprob_only[0][~active_mask],
                torch.zeros_like(logprob_only[0][~active_mask]),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                logprob_only[1][~active_mask],
                torch.zeros_like(logprob_only[1][~active_mask]),
                atol=0,
                rtol=0,
            )
        correctness.append(
            {
                "mask": name,
                "logprobs_max_abs": deltas[0],
                "dhidden_max_abs": deltas[1],
                "dweight_max_abs": deltas[2],
            }
        )
    if rank0:
        print("ENTROPY_SPECIALIZATION_CORRECTNESS=" + json.dumps(correctness, sort_keys=True), flush=True)

    vocab_local = (args.vocab or VOCAB_SHARDS[-1]) // args.tensor_parallel
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
            vocab_local * args.tensor_parallel,
            (local_tokens,),
            device=device,
            generator=shared_generator,
        )
        grad_logprobs = torch.linspace(0.5, 1.5, local_tokens, dtype=torch.float32, device=device)
        active_mask = torch.zeros(local_tokens, dtype=torch.bool, device=device)
        active_mask[local_tokens - math.ceil(local_tokens * 0.66) :] = True

        launch = functools.partial(
            _entropy_specialization_call,
            module,
            hidden,
            weight,
            labels,
            grad_logprobs,
            tp_group,
            active_mask,
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
                        "metrics": metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del launch, hidden, labels, grad_logprobs, active_mask
        torch.cuda.empty_cache()

    dist.destroy_process_group()


def _run_rl_entropy_probe(args) -> None:
    """Compare separate and combined RL entropy, optionally including its gradient."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    world = dist.get_world_size()
    if world != args.tensor_parallel * args.context_parallel:
        raise ValueError(f"torchrun world size {world} must equal TP*CP={args.tensor_parallel * args.context_parallel}")
    kernel_path = (
        Path(__file__).resolve().parents[1] / "backends/skyrl_train/distributed/megatron/fused_linear_logprob_triton.py"
    )
    spec = importlib.util.spec_from_file_location("skyrl_fused_linear_logprob_triton", kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Triton kernel from {kernel_path}")
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)
    FusedLinearLogprobTriton = kernel_module.FusedLinearLogprobTriton

    tp_groups = [
        dist.new_group(ranks=list(range(cp_rank * args.tensor_parallel, (cp_rank + 1) * args.tensor_parallel)))
        for cp_rank in range(args.context_parallel)
    ]
    tp_group = tp_groups[dist.get_rank() // args.tensor_parallel]
    tp_rank = dist.get_rank(tp_group)
    cp_rank = dist.get_rank() // args.tensor_parallel
    device = torch.device("cuda", local_rank)
    rank0 = dist.get_rank() == 0
    vocab_local = (args.vocab or VOCAB_SHARDS[-1]) // args.tensor_parallel

    try:
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            FusedLinearChunkedDistributedLogprob,
        )
    except ModuleNotFoundError:
        # The standalone Triton probe intentionally runs in a minimal image;
        # omit the optional Torch/Megatron comparison when its package deps are absent.
        FusedLinearChunkedDistributedLogprob = None

    class DistributedEntropy(torch.autograd.Function):
        """Materialized-logits entropy baseline with a TP-correct backward."""

        @staticmethod
        def forward(ctx, logits, group):
            maximum = logits.amax(dim=-1, keepdim=True)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            exponentials = (logits - maximum).exp()
            denominator = exponentials.sum(dim=-1, keepdim=True)
            dist.all_reduce(denominator, group=group)
            probabilities = exponentials / denominator
            expected_logit = (probabilities * logits).sum(dim=-1, keepdim=True)
            dist.all_reduce(expected_logit, group=group)
            ctx.save_for_backward(logits, probabilities, expected_logit)
            return (maximum + denominator.log() - expected_logit).squeeze(-1)

        @staticmethod
        def backward(ctx, grad_output):
            logits, probabilities, expected_logit = ctx.saved_tensors
            return probabilities * (expected_logit - logits) * grad_output.unsqueeze(-1), None

    def legacy_entropy(hidden, weight):
        output = torch.empty(hidden.shape[:2], dtype=torch.float32, device=hidden.device)
        for start in range(0, hidden.shape[1], args.chunk_size):
            end = min(hidden.shape[1], start + args.chunk_size)
            logits = (hidden[:, start:end].to(weight.dtype) @ weight.T).float()
            output[:, start:end] = DistributedEntropy.apply(logits, tp_group)
        return output

    def launch(
        backend,
        combined,
        sparse,
        legacy_entropy_start,
        hidden,
        weight,
        target,
        active_mask,
        active_spans,
        response_start,
    ):
        hidden.grad = None
        weight.grad = None
        op_args = (
            hidden,
            weight,
            target,
            tp_rank * vocab_local,
            (tp_rank + 1) * vocab_local,
            args.chunk_size,
            tp_group,
            False,
        )
        if backend == "triton":
            if combined:
                kernel_mask = active_mask if sparse else None
                logprobs, entropy = FusedLinearLogprobTriton.apply(
                    *op_args,
                    kernel_mask,
                    active_spans if sparse else None,
                    True,
                    args.rl_entropy_requires_grad,
                )
            else:
                logprobs = FusedLinearLogprobTriton.apply(*op_args)
        else:
            if FusedLinearChunkedDistributedLogprob is None:
                raise RuntimeError("torch fused backend requires megatron-core in the benchmark environment")
            if combined:
                logprobs, entropy = FusedLinearChunkedDistributedLogprob.apply(
                    *op_args, True, args.rl_entropy_requires_grad
                )
            else:
                logprobs = FusedLinearChunkedDistributedLogprob.apply(*op_args)
        if not combined:
            with torch.set_grad_enabled(args.rl_entropy_requires_grad):
                entropy = legacy_entropy(hidden[:, legacy_entropy_start:], weight)
            entropy = entropy[:, response_start - legacy_entropy_start :]
        else:
            entropy = entropy[:, response_start:]
        loss = -(logprobs * active_mask).sum()
        # Force the no-grad metric reduction into the measured stream.
        response_mask = active_mask[:, response_start:]
        metric = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1)
        if args.rl_entropy_requires_grad:
            loss = loss - metric
        loss.backward()
        return logprobs.masked_fill(~active_mask, 0.0), entropy.masked_fill(~response_mask, 0.0), metric

    @torch.no_grad()
    def fp64_forward_preflight():
        num_tokens, hidden_size, full_vocab = 257, 128, 2048
        local_vocab = full_vocab // args.tensor_parallel
        common_generator = torch.Generator(device=device).manual_seed(122)
        hidden = torch.randn(
            1, num_tokens, hidden_size, dtype=torch.bfloat16, device=device, generator=common_generator
        )
        target = torch.randint(0, full_vocab, (1, num_tokens), device=device, generator=common_generator)
        shard_generator = torch.Generator(device=device).manual_seed(123 + tp_rank)
        weight = (
            torch.randn(local_vocab, hidden_size, dtype=torch.bfloat16, device=device, generator=shard_generator)
            * hidden_size**-0.5
        )
        active_mask = torch.zeros((1, num_tokens), dtype=torch.bool, device=device)
        active_mask[:, 37:151] = True
        active_mask[:, 219:] = True
        actual_logprobs, actual_entropy = FusedLinearLogprobTriton.apply(
            hidden,
            weight,
            target,
            tp_rank * local_vocab,
            (tp_rank + 1) * local_vocab,
            num_tokens,
            tp_group,
            True,
            active_mask,
            None,
            True,
        )

        logits = hidden.double() @ weight.double().T
        maximum = logits.amax(dim=-1, keepdim=True)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=tp_group)
        exponentials = (logits - maximum).exp()
        denominator = exponentials.sum(dim=-1, keepdim=True)
        dist.all_reduce(denominator, group=tp_group)
        local_log_probs = logits - maximum - denominator.log()
        local_probabilities = exponentials / denominator
        expected_entropy = -(local_probabilities * local_log_probs).sum(dim=-1)
        dist.all_reduce(expected_entropy, group=tp_group)

        local_target = target - tp_rank * local_vocab
        owned = (local_target >= 0) & (local_target < local_vocab)
        expected_logprobs = local_log_probs.gather(-1, local_target.clamp(0, local_vocab - 1).unsqueeze(-1)).squeeze(-1)
        expected_logprobs.masked_fill_(~owned, 0.0)
        dist.all_reduce(expected_logprobs, group=tp_group)
        expected_logprobs.masked_fill_(~active_mask, 0.0)
        expected_entropy.masked_fill_(~active_mask, 0.0)

        torch.testing.assert_close(actual_logprobs.double(), expected_logprobs, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(actual_entropy.double(), expected_entropy, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(actual_logprobs[~active_mask]) == 0
        assert torch.count_nonzero(actual_entropy[~active_mask]) == 0
        return {
            "logprob_max_abs": (actual_logprobs.double() - expected_logprobs).abs().max().item(),
            "entropy_max_abs": (actual_entropy.double() - expected_entropy).abs().max().item(),
        }

    fp64_errors = fp64_forward_preflight()

    if rank0:
        print(
            f"RL entropy probe | {torch.cuda.get_device_name(device)} | TP={args.tensor_parallel} "
            f"CP={args.context_parallel} | H={args.hidden} | V/rank={vocab_local} | "
            f"entropy_grad={args.rl_entropy_requires_grad}",
            flush=True,
        )
        print("RL_ENTROPY_FP64=" + json.dumps(fp64_errors, sort_keys=True), flush=True)

    for total_tokens in args.seq_lens:
        if total_tokens % args.context_parallel:
            raise ValueError(f"seq_len={total_tokens} must be divisible by CP={args.context_parallel}")
        local_tokens = total_tokens // args.context_parallel
        generator = torch.Generator(device=device).manual_seed(1220 + total_tokens)
        hidden = torch.randn(
            1,
            local_tokens,
            args.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            requires_grad=True,
        )
        weight = (
            torch.randn(
                vocab_local,
                args.hidden,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            * (args.hidden**-0.5)
        ).requires_grad_(True)
        target = torch.randint(
            0,
            vocab_local * args.tensor_parallel,
            (1, local_tokens),
            device=device,
            generator=generator,
        )

        for fraction in BLOCK_MASK_ACTIVE_FRACTIONS:
            global_active_mask = torch.zeros((1, total_tokens), dtype=torch.bool, device=device)
            active_rows = math.ceil(total_tokens * fraction)
            global_active_mask[:, total_tokens - active_rows :] = True
            # Split one response-like suffix into turns without sorting or shuffling.
            if fraction <= 0.75 and active_rows >= 1024:
                global_active_mask[:, total_tokens - active_rows // 2 - 256 : total_tokens - active_rows // 2] = False
            if args.context_parallel > 1:
                cp_chunk = total_tokens // (2 * args.context_parallel)
                mirror_rank = 2 * args.context_parallel - cp_rank - 1
                active_mask = torch.cat(
                    (
                        global_active_mask[:, cp_rank * cp_chunk : (cp_rank + 1) * cp_chunk],
                        global_active_mask[:, mirror_rank * cp_chunk : (mirror_rank + 1) * cp_chunk],
                    ),
                    dim=1,
                )
                response_start = 0
            else:
                active_mask = global_active_mask
                response_start = total_tokens - active_rows
            active_spans = _active_spans(active_mask, 128)
            variants = {
                "legacy-triton-packed": ("triton", False, False, 0),
                "combined-triton": ("triton", True, False, 0),
                "sparse-triton": ("triton", True, True, 0),
            }
            if args.context_parallel == 1:
                variants["legacy-triton-unpacked"] = ("triton", False, False, response_start)
            if FusedLinearChunkedDistributedLogprob is not None:
                variants["legacy-torch-packed"] = ("torch", False, False, 0)
                variants["combined-torch"] = ("torch", True, False, 0)
                if args.context_parallel == 1:
                    variants["legacy-torch-unpacked"] = ("torch", False, False, response_start)

            launch_args = (hidden, weight, target, active_mask, active_spans, response_start)
            reference = launch(*variants["legacy-triton-packed"], *launch_args)
            reference_hidden_grad = hidden.grad.detach().clone()
            reference_weight_grad = weight.grad.detach().clone()
            reference_hidden_grad_max = reference_hidden_grad.abs().max().item()
            reference_weight_grad_max = reference_weight_grad.abs().max().item()
            correctness = {}
            for name, variant in variants.items():
                result = launch(*variant, *launch_args)
                torch.testing.assert_close(result[0], reference[0], atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(result[1], reference[1], atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(hidden.grad, reference_hidden_grad, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(weight.grad, reference_weight_grad, atol=2e-2, rtol=2e-2)
                correctness[name] = {
                    "logprob_max_abs": (result[0] - reference[0]).abs().max().item(),
                    "entropy_max_abs": (result[1] - reference[1]).abs().max().item(),
                    "dhidden_max_abs": (hidden.grad - reference_hidden_grad).abs().max().item(),
                    "dweight_max_abs": (weight.grad - reference_weight_grad).abs().max().item(),
                    "dhidden_reference_max_abs": reference_hidden_grad_max,
                    "dweight_reference_max_abs": reference_weight_grad_max,
                }
            if rank0:
                print("RL_ENTROPY_CORRECTNESS=" + json.dumps(correctness, sort_keys=True), flush=True)

            for variant in variants.values():
                launch(*variant, *launch_args)
            torch.cuda.synchronize(device)
            samples = {name: [] for name in variants}
            ordered = list(variants)
            for repetition in range(args.repetitions):
                offset = repetition % len(ordered)
                for name in ordered[offset:] + ordered[:offset]:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    launch(*variants[name], *launch_args)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end))

            local_metrics = torch.tensor(
                [statistics.median(samples[name]) for name in ordered],
                dtype=torch.float64,
                device=device,
            )
            dist.reduce(local_metrics, dst=0, op=dist.ReduceOp.MAX)
            if rank0:
                timings = dict(zip(ordered, local_metrics.tolist()))
                result = {
                    "tokens": total_tokens,
                    "active_fraction": float(global_active_mask.float().mean()),
                    "tp": args.tensor_parallel,
                    "cp": args.context_parallel,
                    "entropy_requires_grad": args.rl_entropy_requires_grad,
                    "timings_ms": timings,
                    "packed_dense_triton_speedup": timings["legacy-triton-packed"] / timings["combined-triton"],
                    "packed_sparse_triton_speedup": timings["legacy-triton-packed"] / timings["sparse-triton"],
                }
                if "legacy-triton-unpacked" in timings:
                    result["unpacked_dense_triton_speedup"] = (
                        timings["legacy-triton-unpacked"] / timings["combined-triton"]
                    )
                    result["unpacked_sparse_triton_speedup"] = (
                        timings["legacy-triton-unpacked"] / timings["sparse-triton"]
                    )
                if "legacy-torch-packed" in timings:
                    result["packed_torch_speedup"] = timings["legacy-torch-packed"] / timings["combined-torch"]
                    if "legacy-torch-unpacked" in timings:
                        result["unpacked_torch_speedup"] = timings["legacy-torch-unpacked"] / timings["combined-torch"]
                print("RL_ENTROPY_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
        del hidden, weight, target
        torch.cuda.empty_cache()

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab",
        type=int,
        default=None,
        help="full vocab; per-rank shard = vocab // tensor-parallel size",
    )
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--tensor-parallel", type=int, default=None)
    parser.add_argument("--context-parallel", type=int, default=1)
    parser.add_argument("--mask-trace", type=Path)
    parser.add_argument("--mask-trace-datasets", nargs="+")
    parser.add_argument(
        "--autotune-sweep",
        action="store_true",
        help="sweep Triton schedules over production local shapes",
    )
    parser.add_argument(
        "--block-mask-probe",
        action="store_true",
        help="compare dense Triton with 100/95/75/66/25%-active suffix masks",
    )
    parser.add_argument(
        "--entropy-specialization-probe",
        action="store_true",
        help="compare general entropy with SkyRL's logprob-only specialization",
    )
    parser.add_argument(
        "--rl-entropy-probe",
        action="store_true",
        help="compare separate and combined fused log-prob/entropy for RL masks",
    )
    parser.add_argument(
        "--rl-entropy-requires-grad",
        action="store_true",
        help="include the entropy bonus in backward for --rl-entropy-probe",
    )
    parser.add_argument(
        "--narrow-dlogits-tail-probe",
        action="store_true",
        help="compare padded and direct-width final d-logits staging",
    )
    parser.add_argument(
        "--block-sparse-backward-span-gaps",
        type=int,
        nargs="+",
        help="benchmark ordered contiguous-span dHidden with these maximum inactive gaps",
    )
    parser.add_argument(
        "--block-mask-trace-only",
        action="store_true",
        help="omit synthetic suffix masks from the block-mask probe",
    )
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="JSONL output directory required by --autotune-sweep",
    )
    parser.add_argument("--repetitions", type=int, default=BENCH_REPS)
    args = parser.parse_args()
    if args.tensor_parallel is None:
        args.tensor_parallel = int(os.environ.get("WORLD_SIZE", "1")) // args.context_parallel
    if (args.mask_trace is None) != (args.mask_trace_datasets is None):
        parser.error("--mask-trace and --mask-trace-datasets must be provided together")
    if args.rl_entropy_requires_grad and not args.rl_entropy_probe:
        parser.error("--rl-entropy-requires-grad requires --rl-entropy-probe")
    probes = (args.autotune_sweep, args.block_mask_probe, args.entropy_specialization_probe, args.rl_entropy_probe)
    if sum(probes) > 1:
        parser.error("choose only one benchmark probe mode")
    if args.narrow_dlogits_tail_probe and not args.block_mask_probe:
        parser.error("--narrow-dlogits-tail-probe requires --block-mask-probe")
    if args.autotune_sweep:
        if args.output_dir is None:
            parser.error("--autotune-sweep requires --output-dir")
        _run_autotune_sweep(args.output_dir, args.repetitions)
    elif args.block_mask_probe:
        _run_block_mask_probe(args)
    elif args.entropy_specialization_probe:
        _run_entropy_specialization_probe(args)
    elif args.rl_entropy_probe:
        _run_rl_entropy_probe(args)
    else:
        _run_backend_comparison(args)


if __name__ == "__main__":
    main()
