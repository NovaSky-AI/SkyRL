"""Sweep tile sizes for ragged_dot CUTLASS kernel optimization."""

import subprocess
import sys
import re
from pathlib import Path

CUDA_FILE = Path(__file__).parent.parent / "tx/ffi/ragged_dot_ffi.cu"
SO_FILE = Path(__file__).parent.parent / "tx/ffi/libragged_dot_ffi.so"

# Tile configurations to test: (M, N, K)
# Constraints: dimensions should be powers of 2 or multiples that work with SM90
TILE_CONFIGS = [
    (128, 256, 64),   # current
    (128, 128, 64),
    (64, 128, 64),
    (64, 256, 64),
    (256, 128, 64),
    (128, 64, 128),
    (64, 64, 128),
    (128, 128, 128),
]

# Workload presets for benchmarking
WORKLOAD_PRESETS = {
    "moe": {
        "description": "MoE expert layer (Qwen3-30B-A3B)",
        "args": ["--preset", "moe"],
    },
    "lora": {
        "description": "LoRA adapter layer (rank=8)",
        "args": ["--preset", "lora"],
    },
    "lora-moe": {
        "description": "LoRA on MoE experts",
        "args": ["--preset", "lora-moe"],
    },
}


def set_tile_shape(m: int, n: int, k: int) -> None:
    """Update the TileShape in the CUDA file."""
    content = CUDA_FILE.read_text()

    # Replace the TileShape definition
    pattern = r'using TileShape = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using TileShape = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)
    print(f"Set TileShape to ({m}, {n}, {k})")


def rebuild_kernel() -> bool:
    """Rebuild the CUTLASS kernel."""
    # Remove old .so to force rebuild
    if SO_FILE.exists():
        SO_FILE.unlink()

    # Rebuild using uv with hatchling
    result = subprocess.run(
        ["uv", "run", "--with", "hatchling", "python", "-c",
         "from tx.ffi.build import build_ragged_dot; build_ragged_dot()"],
        capture_output=True,
        text=True,
        cwd=CUDA_FILE.parent.parent.parent,
    )

    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False

    return SO_FILE.exists()


def run_benchmark(workload: str = "moe", num_tokens: int = 8192) -> dict | None:
    """Run the benchmark and parse results."""
    # Build command with workload preset or custom args
    cmd = [sys.executable, "benchmarks/bench_ragged_dot.py", "--num-warmup", "3", "--num-iters", "10"]

    if workload in WORKLOAD_PRESETS:
        cmd.extend(WORKLOAD_PRESETS[workload]["args"])
    else:
        # Legacy: custom num_tokens only
        cmd.extend(["--num-tokens", str(num_tokens)])

    # Need to run in a fresh process to reload the .so
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=CUDA_FILE.parent.parent.parent,
    )

    output = result.stdout
    if result.returncode != 0:
        print(f"Benchmark failed: {result.stderr}")
        return None

    # Parse CUTLASS FFI results
    results = {}

    # Forward pass
    match = re.search(r'CUTLASS FFI:\s+([\d.]+)\s+ms\s+([\d.]+)\s+TFLOPS', output)
    if match:
        results['fwd_ms'] = float(match.group(1))
        results['fwd_tflops'] = float(match.group(2))

    # Find backward pass (second occurrence)
    matches = list(re.finditer(r'CUTLASS FFI:\s+([\d.]+)\s+ms\s+([\d.]+)\s+TFLOPS', output))
    if len(matches) >= 2:
        results['bwd_ms'] = float(matches[1].group(1))
        results['bwd_tflops'] = float(matches[1].group(2))

    return results if results else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep tile sizes for CUTLASS kernel")
    parser.add_argument("--workload", choices=list(WORKLOAD_PRESETS.keys()), default="moe",
                        help="Workload preset to benchmark (moe, lora, lora-moe)")
    parser.add_argument("--all-workloads", action="store_true", help="Sweep all workloads")
    parser.add_argument("--num-tokens", type=int, default=8192, help="Number of tokens (for custom workload)")
    parser.add_argument("--dry-run", action="store_true", help="Only show configs, don't run")
    args = parser.parse_args()

    print("CUTLASS Tile Size Sweep")
    print("=" * 70)

    workloads = list(WORKLOAD_PRESETS.keys()) if args.all_workloads else [args.workload]

    if args.dry_run:
        print("Tile configurations to test:")
        for m, n, k in TILE_CONFIGS:
            print(f"  ({m}, {n}, {k})")
        print()
        print("Workloads to test:")
        for w in workloads:
            print(f"  {w}: {WORKLOAD_PRESETS[w]['description']}")
        return

    # Store results per workload
    all_results: dict[str, list] = {w: [] for w in workloads}

    for m, n, k in TILE_CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing TileShape({m}, {n}, {k})")
        print("-" * 70)

        set_tile_shape(m, n, k)

        if not rebuild_kernel():
            print("  FAILED to build")
            for w in workloads:
                all_results[w].append((m, n, k, None))
            continue

        for workload in workloads:
            print(f"\n  Workload: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
            bench_results = run_benchmark(workload)
            if bench_results:
                print(f"    Forward:  {bench_results.get('fwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('fwd_tflops', 'N/A'):>8.2f} TFLOPS")
                print(f"    Backward: {bench_results.get('bwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('bwd_tflops', 'N/A'):>8.2f} TFLOPS")
                all_results[workload].append((m, n, k, bench_results))
            else:
                print("    FAILED to benchmark")
                all_results[workload].append((m, n, k, None))

    # Summary per workload
    for workload in workloads:
        results = all_results[workload]
        print(f"\n{'='*70}")
        print(f"SUMMARY: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
        print("=" * 70)
        print(f"{'TileShape':<20} {'Fwd (ms)':>10} {'Fwd TFLOPS':>12} {'Bwd (ms)':>10} {'Bwd TFLOPS':>12}")
        print("-" * 70)

        for m, n, k, res in results:
            tile_str = f"({m}, {n}, {k})"
            if res:
                fwd_ms = f"{res.get('fwd_ms', 0):.3f}"
                fwd_tf = f"{res.get('fwd_tflops', 0):.2f}"
                bwd_ms = f"{res.get('bwd_ms', 0):.3f}"
                bwd_tf = f"{res.get('bwd_tflops', 0):.2f}"
                print(f"{tile_str:<20} {fwd_ms:>10} {fwd_tf:>12} {bwd_ms:>10} {bwd_tf:>12}")
            else:
                print(f"{tile_str:<20} {'FAILED':>10}")

        # Find best
        valid_results = [(m, n, k, r) for m, n, k, r in results if r and 'fwd_tflops' in r]
        if valid_results:
            best_fwd = max(valid_results, key=lambda x: x[3]['fwd_tflops'])
            best_bwd = max(valid_results, key=lambda x: x[3].get('bwd_tflops', 0))
            print()
            print(f"Best forward:  ({best_fwd[0]}, {best_fwd[1]}, {best_fwd[2]}) - {best_fwd[3]['fwd_tflops']:.2f} TFLOPS")
            print(f"Best backward: ({best_bwd[0]}, {best_bwd[1]}, {best_bwd[2]}) - {best_bwd[3].get('bwd_tflops', 0):.2f} TFLOPS")


if __name__ == "__main__":
    main()
