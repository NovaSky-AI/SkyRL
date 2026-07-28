"""Benchmark resident FSDP multi-LoRA grouped-GEMM compute.

Example:
    python -m skyrl.benchmarks.bench_fsdp_multi_lora \
        --batch-size 16 --sequence-length 2048 --hidden-size 4096 \
        --output-size 4096 --rank 16 --adapters 8
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.workers.fsdp.multi_lora import MultiLoRALinear


def _run_iteration(layer: MultiLoRALinear, inputs: torch.Tensor, operation: Callable[[], torch.Tensor]) -> None:
    layer.zero_grad(set_to_none=True)
    inputs.grad = None
    output = operation()
    output.float().square().mean().backward()


def _measure(
    layer: MultiLoRALinear,
    inputs: torch.Tensor,
    operation: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _run_iteration(layer, inputs, operation)
    torch.cuda.synchronize()

    layer.zero_grad(set_to_none=True)
    inputs.grad = None
    torch.cuda.empty_cache()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _run_iteration(layer, inputs, operation)
    torch.cuda.synchronize()
    peak_temporary_bytes = torch.cuda.max_memory_allocated() - baseline_memory

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _run_iteration(layer, inputs, operation)
    end.record()
    torch.cuda.synchronize()
    elapsed_seconds = start.elapsed_time(end) / 1_000

    tokens = inputs.shape[0] * inputs.shape[1] * iterations
    return {
        "milliseconds_per_iteration": elapsed_seconds * 1_000 / iterations,
        "tokens_per_second": tokens / elapsed_seconds,
        "peak_temporary_mib": peak_temporary_bytes / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--output-size", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--adapters", type=int, default=8)
    parser.add_argument("--active-adapters", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    if not 1 <= args.active_adapters <= args.adapters:
        raise ValueError("active-adapters must be between 1 and adapters")

    dtype = torch.bfloat16
    device = torch.device("cuda")
    torch.manual_seed(0)
    base_layer = nn.Linear(args.hidden_size, args.output_size, bias=False, device=device, dtype=dtype)
    layer = MultiLoRALinear(
        base_layer,
        max_adapters=args.adapters,
        rank=args.rank,
        alpha=args.rank,
        dropout=0,
    ).train()
    with torch.no_grad():
        for adapter in layer.adapters:
            adapter.lora_A.weight.normal_(std=0.02)
            adapter.lora_B.weight.normal_(std=0.02)

    inputs = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    adapter_indices = torch.arange(args.batch_size, device=device) % args.active_adapters
    active_slots, compact_indices = layer.compact_routing(adapter_indices)
    layer.set_adapter_indices(adapter_indices, active_slots=active_slots, compact_indices=compact_indices)

    def grouped_operation() -> torch.Tensor:
        return layer._apply_lora_grouped(inputs, compact_indices, active_slots)

    measurement = _measure(
        layer,
        inputs,
        grouped_operation,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    result = {
        "shape": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "hidden_size": args.hidden_size,
            "output_size": args.output_size,
            "rank": args.rank,
            "resident_adapters": args.adapters,
            "active_adapters": len(active_slots),
            "dtype": "bfloat16",
        },
        "scope": "lora_delta_forward_backward",
        "kernel": "grouped_mm",
        "measurement": measurement,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
