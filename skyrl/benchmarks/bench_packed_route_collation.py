"""Compare padded and packed route collation with serial and pooled fills.

Production shapes need ~110 GiB of host RAM, so drive this from a cluster harness::

    uv run --isolated --extra skyrl-train python -m \
        skyrl.benchmarks.bench_packed_route_collation --num-moe-layers 40

Scaled-down smoke::

    uv run --isolated --extra skyrl-train python -m \
        skyrl.benchmarks.bench_packed_route_collation \
        --num-sequences 64 --max-seqlen 4096 --num-moe-layers 4 --iterations 1
"""

import argparse
import functools
import gc
import os
import resource
import statistics
import time

import numpy as np
import torch

from skyrl.backends.skyrl_train.utils.packed_tensor import cu_seqlens_from_lengths
from skyrl.backends.skyrl_train.utils.replay_utils import replay_padding_row
from skyrl.train.dataset.parallel_fill import default_fill_workers, fill_batch_rows

DEFAULT_NUM_MOE_LAYERS = 40
DEFAULT_TOPK = 22
DEFAULT_NUM_SEQUENCES = 1024
DEFAULT_MAX_SEQLEN = 32768
ROUTE_DTYPE = torch.int16

# Fraction of ``max_seqlen`` each distribution draws its shortest sequence from. "uniform"
# has no padding at all; "typical_rl" is the measured production spread.
LENGTH_DISTRIBUTIONS = {
    "uniform": 1.0,
    "mild_ragged": 0.5,
    "typical_rl": 1 / 16,
    "heavy_tail": 1 / 64,
}


def _sequence_lengths(distribution: str, num_sequences: int, max_seqlen: int, seed: int) -> np.ndarray:
    """Draw per-trajectory total lengths, always including one full-length sequence."""
    minimum = max(1, round(max_seqlen * LENGTH_DISTRIBUTIONS[distribution]))
    if minimum >= max_seqlen:
        return np.full(num_sequences, max_seqlen, dtype=np.int64)
    rng = np.random.default_rng(seed)
    lengths = rng.integers(minimum, max_seqlen + 1, size=num_sequences).astype(np.int64)
    # max_total is set by the longest trajectory, so pin one to the cap for a stable rectangle.
    lengths[0] = max_seqlen
    return lengths


def _make_trajectories(lengths: np.ndarray, num_layers: int, topk: int, seed: int) -> list[np.ndarray]:
    """One route array per trajectory, sized to its full sequence length.

    Arrays are views over one template so source allocation does not dominate the benchmark.
    """
    rng = np.random.default_rng(seed + 1)
    template = rng.integers(0, 128, size=(int(lengths.max()), num_layers, topk), dtype=np.int16)
    return [template[: int(length)] for length in lengths]


def _write_padded_row(
    padded: torch.Tensor,
    trajectories: list[np.ndarray],
    lengths: np.ndarray,
    sample_index: int,
) -> None:
    """One trajectory's slot in the rectangle: left dummy rows, routes, trailing dummy rows."""
    padding_row = replay_padding_row(padded.shape[-1], dtype=padded.dtype)
    sample_indices = trajectories[sample_index]
    left_pad = padded.shape[1] - int(lengths[sample_index])
    route_end = left_pad + sample_indices.shape[0]
    padded[sample_index, :left_pad] = padding_row
    padded[sample_index, left_pad:route_end] = torch.from_numpy(sample_indices)
    padded[sample_index, route_end:] = padding_row


def _write_packed_segment(
    packed: torch.Tensor,
    cu_seqlens: torch.Tensor,
    trajectories: list[np.ndarray],
    sample_index: int,
) -> None:
    """One trajectory's segment of the packed buffer: routes, then any trailing dummy rows."""
    sample_indices = trajectories[sample_index]
    segment = packed[int(cu_seqlens[sample_index]) : int(cu_seqlens[sample_index + 1])]
    captured = sample_indices.shape[0]
    segment[:captured] = torch.from_numpy(sample_indices)
    segment[captured:] = replay_padding_row(segment.shape[-1], dtype=packed.dtype)


def _make_fill(packed: bool, workers: int):
    """Build a fill callable over the trainer's own pool helper."""

    def fill(buffer: torch.Tensor, trajectories: list[np.ndarray], lengths: np.ndarray) -> None:
        if packed:
            cu_seqlens = cu_seqlens_from_lengths(lengths)
            write = functools.partial(_write_packed_segment, buffer, cu_seqlens, trajectories)
        else:
            write = functools.partial(_write_padded_row, buffer, trajectories, lengths)
        fill_batch_rows(write, len(trajectories), workers=workers)

    return fill


def _padded_shape(lengths: np.ndarray, num_layers: int, topk: int) -> tuple[int, ...]:
    return (len(lengths), int(lengths.max()), num_layers, topk)


def _packed_shape(lengths: np.ndarray, num_layers: int, topk: int) -> tuple[int, ...]:
    return (int(lengths.sum()), num_layers, topk)


def _peak_rss_bytes() -> int:
    """Process high-water RSS. Monotone, so only ever read as a whole-run ceiling."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _time_cold(fill, shape, trajectories, lengths, iterations: int) -> tuple[float, int]:
    """Median wall clock over a freshly allocated buffer each iteration.

    Fresh buffers retain the first-touch allocation cost measured in production.
    """
    durations = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        buffer = torch.empty(shape, dtype=ROUTE_DTYPE)
        fill(buffer, trajectories, lengths)
        durations.append(time.perf_counter() - start)
        buffer_bytes = buffer.numel() * buffer.element_size()
        del buffer
    return statistics.median(durations), buffer_bytes


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:8.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES)
    parser.add_argument("--max-seqlen", type=int, default=DEFAULT_MAX_SEQLEN)
    parser.add_argument("--num-moe-layers", type=int, default=DEFAULT_NUM_MOE_LAYERS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_fill_workers())
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=list(LENGTH_DISTRIBUTIONS),
        choices=list(LENGTH_DISTRIBUTIONS),
    )
    args = parser.parse_args()

    print(
        f"num_sequences={args.num_sequences} max_seqlen={args.max_seqlen} "
        f"moe_layers={args.num_moe_layers} topk={args.topk} dtype={ROUTE_DTYPE} "
        f"iterations={args.iterations}"
    )
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')} torch_threads={torch.get_num_threads()}")
    print(f"pooled arms use {args.workers} workers (the trainer's autoscaled pool size)")
    header = (
        f"{'distribution':<20} {'pad_1t':>8} {'pad_pool':>9} {'pack_1t':>8} {'pack_pool':>10} "
        f"{'best_pad':>9} {'vs_best':>8} {'pad_GiB':>9} {'packed_GiB':>11} {'saved':>7}"
    )
    print(header)
    print("-" * len(header))

    for distribution in args.distributions:
        lengths = _sequence_lengths(distribution, args.num_sequences, args.max_seqlen, args.seed)
        trajectories = _make_trajectories(lengths, args.num_moe_layers, args.topk, args.seed)

        padded_shape = _padded_shape(lengths, args.num_moe_layers, args.topk)
        packed_shape = _packed_shape(lengths, args.num_moe_layers, args.topk)
        arms = {
            "pad_1t": (_make_fill(False, 1), padded_shape),
            "pad_pool": (_make_fill(False, args.workers), padded_shape),
            "pack_1t": (_make_fill(True, 1), packed_shape),
            "pack_pool": (_make_fill(True, args.workers), packed_shape),
        }
        timings = {}
        buffer_bytes = {}
        for name, (fill, shape) in arms.items():
            timings[name], buffer_bytes[name] = _time_cold(fill, shape, trajectories, lengths, args.iterations)
        del trajectories
        gc.collect()

        best_padded = min(timings["pad_1t"], timings["pad_pool"])
        best_packed = min(timings["pack_1t"], timings["pack_pool"])
        print(
            f"{distribution:<20} {timings['pad_1t'] * 1000:8.1f} {timings['pad_pool'] * 1000:9.1f} "
            f"{timings['pack_1t'] * 1000:8.1f} {timings['pack_pool'] * 1000:10.1f} "
            f"{best_padded * 1000:9.1f} {best_padded / best_packed:7.2f}x "
            f"{_format_gib(buffer_bytes['pad_1t'])} {_format_gib(buffer_bytes['pack_1t']):>11} "
            f"{1 - buffer_bytes['pack_1t'] / buffer_bytes['pad_1t']:6.1%}"
        )

    print(f"process peak RSS: {_format_gib(_peak_rss_bytes()).strip()} GiB")


if __name__ == "__main__":
    main()
