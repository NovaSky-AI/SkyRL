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
    (128, 256, 64),
    (128, 128, 64),
    (64, 128, 64),
    (64, 256, 64),
    (256, 128, 64),
    (128, 64, 128),
    (64, 64, 128),
    (128, 128, 128),
]

# Qwen3-30B-A3B MoE shapes
MOE_SHAPES = {
    "gate_up": {"hidden_size": 2048, "intermediate_size": 768},   # K=2048, N=768
    "down":    {"hidden_size": 768,  "intermediate_size": 2048},  # K=768, N=2048
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


def run_benchmark(num_tokens: int, hidden_size: int, intermediate_size: int) -> dict | None:
    """Run the benchmark and parse results."""
    # Need to run in a fresh process to reload the .so
    result = subprocess.run(
        [sys.executable, "benchmarks/bench_ragged_dot.py",
         "--num-tokens", str(num_tokens),
         "--hidden-size", str(hidden_size),
         "--intermediate-size", str(intermediate_size),
         "--num-warmup", "3",
         "--num-iters", "10",
         "--forward-only"],
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

    return results if results else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep tile sizes for CUTLASS kernel")
    parser.add_argument("--num-tokens", type=int, default=8192, help="Number of tokens")
    parser.add_argument("--dry-run", action="store_true", help="Only show configs, don't run")
    args = parser.parse_args()

    print("CUTLASS Tile Size Sweep")
    print("=" * 70)
    print(f"Qwen3-30B-A3B MoE shapes, M={args.num_tokens}")
    print(f"  gate_up: K=2048, N=768")
    print(f"  down:    K=768,  N=2048")
    print()

    if args.dry_run:
        print("Tile configurations to test:")
        for m, n, k in TILE_CONFIGS:
            print(f"  ({m}, {n}, {k})")
        return

    results = []

    for m, n, k in TILE_CONFIGS:
        print(f"\n{'='*70}")
        print(f"Testing TileShape({m}, {n}, {k})")
        print("-" * 70)

        set_tile_shape(m, n, k)

        if not rebuild_kernel():
            print("  FAILED to build")
            results.append((m, n, k, None, None))
            continue

        tile_results = {}
        for shape_name, shape_cfg in MOE_SHAPES.items():
            bench_result = run_benchmark(
                args.num_tokens,
                shape_cfg["hidden_size"],
                shape_cfg["intermediate_size"],
            )
            tile_results[shape_name] = bench_result
            if bench_result:
                print(f"  {shape_name:>8}: {bench_result['fwd_ms']:>8.3f} ms  {bench_result['fwd_tflops']:>8.2f} TFLOPS")
            else:
                print(f"  {shape_name:>8}: FAILED")

        results.append((m, n, k, tile_results.get("gate_up"), tile_results.get("down")))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'TileShape':<20} {'gate_up ms':>12} {'TFLOPS':>8} {'down ms':>12} {'TFLOPS':>8} {'Total ms':>10}")
    print("-" * 70)

    for m, n, k, gate_up, down in results:
        tile_str = f"({m}, {n}, {k})"
        if gate_up and down:
            gu_ms = gate_up['fwd_ms']
            gu_tf = gate_up['fwd_tflops']
            dn_ms = down['fwd_ms']
            dn_tf = down['fwd_tflops']
            total_ms = gu_ms + dn_ms
            print(f"{tile_str:<20} {gu_ms:>12.3f} {gu_tf:>8.2f} {dn_ms:>12.3f} {dn_tf:>8.2f} {total_ms:>10.3f}")
        else:
            print(f"{tile_str:<20} {'FAILED':>12}")

    # Find best
    valid_results = [(m, n, k, gu, dn) for m, n, k, gu, dn in results if gu and dn]
    if valid_results:
        best_gate_up = max(valid_results, key=lambda x: x[3]['fwd_tflops'])
        best_down = max(valid_results, key=lambda x: x[4]['fwd_tflops'])
        best_total = min(valid_results, key=lambda x: x[3]['fwd_ms'] + x[4]['fwd_ms'])
        print()
        print(f"Best gate_up: ({best_gate_up[0]}, {best_gate_up[1]}, {best_gate_up[2]}) - {best_gate_up[3]['fwd_tflops']:.2f} TFLOPS")
        print(f"Best down:    ({best_down[0]}, {best_down[1]}, {best_down[2]}) - {best_down[4]['fwd_tflops']:.2f} TFLOPS")
        print(f"Best total:   ({best_total[0]}, {best_total[1]}, {best_total[2]}) - {best_total[3]['fwd_ms'] + best_total[4]['fwd_ms']:.3f} ms")


if __name__ == "__main__":
    main()
