"""Sweep tile sizes and kernel parameters for ragged_dot CUTLASS kernel optimization."""

import os
import subprocess
import sys
import re
from pathlib import Path
from itertools import product

CUDA_FILE = Path(__file__).parent.parent / "tx/ffi/ragged_dot_ffi.cu"
SO_FILE = Path(__file__).parent.parent / "tx/ffi/libragged_dot_ffi.so"

# Tile configurations to test: (M, N, K)
# Constraints: dimensions should be powers of 2 or multiples that work with SM90
# Note: Cooperative schedule requires M >= 128
TILE_CONFIGS = [
    (64, 256, 64),    # best from previous sweep
    (128, 128, 64),
    (64, 128, 64),
    (128, 256, 64),
    (128, 128, 128),
]

# Smaller tile configs that may work better for LoRA (small K dimension)
TILE_CONFIGS_LORA = [
    (128, 128, 32),   # smaller K tile
    (128, 256, 32),
    (128, 64, 32),
    (64, 128, 32),
    (64, 64, 32),
    (128, 128, 64),   # reference
]

# Decode-specific tile configs for large K (LoRA A, MoE: K=hidden_size)
DECODE_TILE_CONFIGS = [
    (128, 64, 64),   # current Decode config (Cooperative)
    (128, 128, 64),  # wider N
    (128, 256, 64),  # very wide N
    (128, 64, 128),  # deeper K reduction
    (128, 128, 128), # wider N + deeper K
    (64, 128, 64),   # smaller M (Pingpong only)
    (64, 256, 64),   # smaller M, wide N (Pingpong only)
    (64, 64, 64),    # small tiles (Pingpong only)
    (64, 128, 128),  # smaller M, deep K (Pingpong only)
]

# Decode-specific tile configs for small K (LoRA B: K=rank, typically 8-64)
DECODE_SMALLK_TILE_CONFIGS = [
    (128, 128, 32),  # current DecodeSmallK config (Cooperative)
    (128, 256, 32),  # wider N
    (128, 64, 32),   # narrower N
    (128, 128, 16),  # very small K tile
    (128, 256, 16),  # wide N, tiny K
    (64, 128, 32),   # smaller M (Pingpong only)
    (64, 256, 32),   # smaller M, wide N (Pingpong only)
    (64, 64, 32),    # small tiles (Pingpong only)
]

# Phase 2: best decode configs for cluster sweep
DECODE_TILE_CONFIGS_PHASE2 = [
    (128, 128, 64),  # best decode-moe fwd (Cooperative)
    (64, 256, 32),   # best decode-lora fwd (Pingpong)
    (128, 128, 32),  # strong on both workloads (Cooperative)
    (64, 128, 64),   # current config baseline (Pingpong)
]

# Cluster shapes to test: (M, N, K)
CLUSTER_CONFIGS = [
    (1, 1, 1),        # current
    (2, 1, 1),
    (1, 2, 1),
    (2, 2, 1),
]

# Kernel schedules to test
SCHEDULE_CONFIGS = [
    ("KernelPtrArrayTmaWarpSpecializedPingpong", "PtrArrayTmaWarpSpecializedPingpong"),      # current
    ("KernelPtrArrayTmaWarpSpecializedCooperative", "PtrArrayTmaWarpSpecializedCooperative"),
]

# Compiler flag configurations to sweep
# Each entry is (label, extra_flags_string)
COMPILER_FLAG_CONFIGS = [
    ("baseline", ""),
    ("fast_math", "--use_fast_math"),
    ("ndebug", "-DNDEBUG"),
    ("extra_vectorization", "--extra-device-vectorization"),
    ("ptxas_O3", "-Xptxas -O3"),
    ("ptxas_expensive_opts", "-Xptxas -O3 -Xptxas --allow-expensive-optimizations=true"),
    ("fast_math+ndebug", "--use_fast_math -DNDEBUG"),
    ("fast_math+ndebug+extra_vec", "--use_fast_math -DNDEBUG --extra-device-vectorization"),
    ("all_opts", "--use_fast_math -DNDEBUG --extra-device-vectorization -Xptxas -O3 -Xptxas --allow-expensive-optimizations=true"),
]

# Workload presets for benchmarking
WORKLOAD_PRESETS = {
    "decode-moe": {
        "description": "Decode-time MoE expert layer",
        "args": ["--preset", "decode-moe"],
    },
    "decode-lora": {
        "description": "Decode-time LoRA adapter layer",
        "args": ["--preset", "decode-lora"],
    },
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
    "decode-lora-A-qwen4b": {
        "description": "Decode LoRA A pass Qwen3-4B attn (K=2560, N=32)",
        "args": ["--preset", "decode-lora-A-qwen4b"],
    },
    "decode-lora-B-qwen4b": {
        "description": "Decode LoRA B pass Qwen3-4B attn (K=32, N=2560)",
        "args": ["--preset", "decode-lora-B-qwen4b"],
    },
    "decode-lora-B-mlp-qwen4b": {
        "description": "Decode LoRA B pass Qwen3-4B MLP (K=32, N=9728)",
        "args": ["--preset", "decode-lora-B-mlp-qwen4b"],
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


def set_cluster_shape(m: int, n: int, k: int) -> None:
    """Update the ClusterShape in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using ClusterShape = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using ClusterShape = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)


def set_schedule(kernel_schedule: str, epilogue_schedule: str) -> None:
    """Update the KernelSchedule and EpilogueSchedule in the CUDA file."""
    content = CUDA_FILE.read_text()

    # Replace KernelSchedule
    pattern = r'using KernelSchedule = cutlass::gemm::\w+;'
    replacement = f'using KernelSchedule = cutlass::gemm::{kernel_schedule};'
    content = re.sub(pattern, replacement, content)

    # Replace EpilogueSchedule
    pattern = r'using EpilogueSchedule = cutlass::epilogue::\w+;'
    replacement = f'using EpilogueSchedule = cutlass::epilogue::{epilogue_schedule};'
    content = re.sub(pattern, replacement, content)

    CUDA_FILE.write_text(content)


def set_kernel_config(tile: tuple, cluster: tuple, schedule: tuple) -> str:
    """Set all kernel configuration parameters. Returns a config description string."""
    set_tile_shape(*tile)
    set_cluster_shape(*cluster)
    set_schedule(*schedule)

    schedule_name = schedule[0].replace("KernelPtrArrayTmaWarpSpecialized", "")
    return f"Tile{tile} Cluster{cluster} {schedule_name}"


def set_decode_tile_shape(m: int, n: int, k: int) -> None:
    """Update the TileShapeDecode in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using TileShapeDecode = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using TileShapeDecode = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)


def set_decode_cluster_shape(m: int, n: int, k: int) -> None:
    """Update the ClusterShapeDecode in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using ClusterShapeDecode = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using ClusterShapeDecode = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)


def set_decode_schedule(kernel_schedule: str, epilogue_schedule: str) -> None:
    """Update the KernelScheduleDecode and EpilogueScheduleDecode in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using KernelScheduleDecode = cutlass::gemm::\w+;'
    replacement = f'using KernelScheduleDecode = cutlass::gemm::{kernel_schedule};'
    content = re.sub(pattern, replacement, content)

    pattern = r'using EpilogueScheduleDecode = cutlass::epilogue::\w+;'
    replacement = f'using EpilogueScheduleDecode = cutlass::epilogue::{epilogue_schedule};'
    content = re.sub(pattern, replacement, content)

    CUDA_FILE.write_text(content)


def set_decode_kernel_config(tile: tuple, cluster: tuple, schedule: tuple) -> str:
    """Set all decode kernel configuration parameters. Returns a config description string."""
    set_decode_tile_shape(*tile)
    set_decode_cluster_shape(*cluster)
    set_decode_schedule(*schedule)

    schedule_name = schedule[0].replace("KernelPtrArrayTmaWarpSpecialized", "")
    return f"Tile{tile} Cluster{cluster} {schedule_name}"


def set_decode_smallk_tile_shape(m: int, n: int, k: int) -> None:
    """Update the TileShapeDecodeSmallK in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using TileShapeDecodeSmallK = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using TileShapeDecodeSmallK = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)


def set_decode_smallk_cluster_shape(m: int, n: int, k: int) -> None:
    """Update the ClusterShapeDecodeSmallK in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using ClusterShapeDecodeSmallK = cute::Shape<cute::_\d+, cute::_\d+, cute::_\d+>;'
    replacement = f'using ClusterShapeDecodeSmallK = cute::Shape<cute::_{m}, cute::_{n}, cute::_{k}>;'

    new_content = re.sub(pattern, replacement, content)
    CUDA_FILE.write_text(new_content)


def set_decode_smallk_schedule(kernel_schedule: str, epilogue_schedule: str) -> None:
    """Update the KernelScheduleDecodeSmallK and EpilogueScheduleDecodeSmallK in the CUDA file."""
    content = CUDA_FILE.read_text()

    pattern = r'using KernelScheduleDecodeSmallK = cutlass::gemm::\w+;'
    replacement = f'using KernelScheduleDecodeSmallK = cutlass::gemm::{kernel_schedule};'
    content = re.sub(pattern, replacement, content)

    pattern = r'using EpilogueScheduleDecodeSmallK = cutlass::epilogue::\w+;'
    replacement = f'using EpilogueScheduleDecodeSmallK = cutlass::epilogue::{epilogue_schedule};'
    content = re.sub(pattern, replacement, content)

    CUDA_FILE.write_text(content)


def set_decode_smallk_kernel_config(tile: tuple, cluster: tuple, schedule: tuple) -> str:
    """Set all DecodeSmallK kernel configuration parameters. Returns a config description string."""
    set_decode_smallk_tile_shape(*tile)
    set_decode_smallk_cluster_shape(*cluster)
    set_decode_smallk_schedule(*schedule)

    schedule_name = schedule[0].replace("KernelPtrArrayTmaWarpSpecialized", "")
    return f"Tile{tile} Cluster{cluster} {schedule_name}"


def rebuild_kernel(extra_flags: str = "") -> bool:
    """Rebuild the CUTLASS kernel."""
    # Remove old .so to force rebuild
    if SO_FILE.exists():
        SO_FILE.unlink()

    env = os.environ.copy()
    if extra_flags:
        env["EXTRA_NVCC_FLAGS"] = extra_flags

    # Rebuild: --extra gpu for JAX, --with hatchling for build.py module-level import
    result = subprocess.run(
        ["uv", "run", "--extra", "gpu", "--with", "hatchling", "python", "-c",
         "from tx.ffi.build import build_ragged_dot; build_ragged_dot()"],
        capture_output=True,
        text=True,
        cwd=CUDA_FILE.parent.parent.parent,
        env=env,
    )

    if result.returncode != 0:
        print(f"  Build stderr: {result.stderr[-500:]}" if result.stderr else "  (no stderr)")
        return False

    return SO_FILE.exists()


def run_benchmark(workload: str = "moe", num_tokens: int = 8192) -> dict | None:
    """Run the benchmark and parse results."""
    # Build command with workload preset or custom args
    # Use uv run so the benchmark executes inside the project venv (where JAX lives)
    cmd = ["uv", "run", "--extra", "gpu", "python", "benchmarks/bench_ragged_dot.py", "--num-warmup", "3", "--num-iters", "10"]

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


def sweep_compiler_flags(args) -> None:
    """Sweep compiler flag configurations while keeping kernel config fixed."""
    if args.all_workloads:
        decode_workloads = ["decode-lora-A-qwen4b", "decode-lora-B-qwen4b"]
        prefill_workloads = ["moe", "lora", "lora-moe"]
        workloads = decode_workloads if args.decode else prefill_workloads
    else:
        workloads = [args.workload]

    print("CUTLASS Compiler Flag Sweep")
    print("=" * 80)
    print(f"Testing {len(COMPILER_FLAG_CONFIGS)} flag configurations across {len(workloads)} workload(s)")
    print()

    if args.dry_run:
        print("Flag configurations to test:")
        for label, flags in COMPILER_FLAG_CONFIGS:
            print(f"  {label}: {flags or '(none)'}")
        print()
        print("Workloads to test:")
        for w in workloads:
            print(f"  {w}: {WORKLOAD_PRESETS[w]['description']}")
        return

    all_results: dict[str, list] = {w: [] for w in workloads}

    for label, extra_flags in COMPILER_FLAG_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Testing flags: {label}")
        if extra_flags:
            print(f"  EXTRA_NVCC_FLAGS={extra_flags}")
        else:
            print("  (no extra flags)")
        print("-" * 80)

        if not rebuild_kernel(extra_flags):
            print("  FAILED to build")
            for w in workloads:
                all_results[w].append((label, None))
            continue

        for workload in workloads:
            print(f"\n  Workload: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
            bench_results = run_benchmark(workload)
            if bench_results:
                print(f"    Forward:  {bench_results.get('fwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('fwd_tflops', 'N/A'):>8.2f} TFLOPS")
                print(f"    Backward: {bench_results.get('bwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('bwd_tflops', 'N/A'):>8.2f} TFLOPS")
                all_results[workload].append((label, bench_results))
            else:
                print("    FAILED to benchmark")
                all_results[workload].append((label, None))

    # Summary per workload
    for workload in workloads:
        results = all_results[workload]
        print(f"\n{'='*80}")
        print(f"SUMMARY: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
        print("=" * 80)
        print(f"{'Flags':<45} {'Fwd (ms)':>10} {'Fwd TF':>8} {'Bwd (ms)':>10} {'Bwd TF':>8}")
        print("-" * 80)

        for label, res in results:
            if res:
                fwd_ms = f"{res.get('fwd_ms', 0):.3f}"
                fwd_tf = f"{res.get('fwd_tflops', 0):.1f}"
                bwd_ms = f"{res.get('bwd_ms', 0):.3f}"
                bwd_tf = f"{res.get('bwd_tflops', 0):.1f}"
                print(f"{label:<45} {fwd_ms:>10} {fwd_tf:>8} {bwd_ms:>10} {bwd_tf:>8}")
            else:
                print(f"{label:<45} {'FAILED':>10}")

        valid_results = [(cfg, r) for cfg, r in results if r and 'fwd_tflops' in r]
        if valid_results:
            best_fwd = max(valid_results, key=lambda x: x[1]['fwd_tflops'])
            best_bwd = max(valid_results, key=lambda x: x[1].get('bwd_tflops', 0))
            print()
            print(f"Best forward:  {best_fwd[0]} - {best_fwd[1]['fwd_tflops']:.2f} TFLOPS")
            print(f"Best backward: {best_bwd[0]} - {best_bwd[1].get('bwd_tflops', 0):.2f} TFLOPS")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep kernel parameters for CUTLASS ragged_dot")
    parser.add_argument("--workload", choices=list(WORKLOAD_PRESETS.keys()), default=None,
                        help="Workload preset to benchmark")
    parser.add_argument("--all-workloads", action="store_true", help="Sweep all workloads")
    parser.add_argument("--sweep-tiles", action="store_true", help="Sweep tile shapes")
    parser.add_argument("--sweep-clusters", action="store_true", help="Sweep cluster shapes")
    parser.add_argument("--sweep-schedules", action="store_true", help="Sweep kernel schedules")
    parser.add_argument("--sweep-all", action="store_true", help="Sweep all parameters")
    parser.add_argument("--lora-tiles", action="store_true", help="Use LoRA-optimized tile configs (smaller K)")
    parser.add_argument("--decode", action="store_true",
                        help="Sweep decode kernel parameters for large K (TileShapeDecode)")
    parser.add_argument("--decode-smallk", action="store_true",
                        help="Sweep decode kernel parameters for small K (TileShapeDecodeSmallK)")
    parser.add_argument("--sweep-flags", action="store_true",
                        help="Sweep nvcc compiler flags (keeps tile/cluster/schedule fixed)")
    parser.add_argument("--phase2", action="store_true",
                        help="Use phase2 tile configs (best from phase1) for cluster sweep")
    parser.add_argument("--dry-run", action="store_true", help="Only show configs, don't run")
    args = parser.parse_args()

    # Default workload depends on mode
    if args.workload is None:
        if args.decode_smallk:
            args.workload = "decode-lora-B-qwen4b"
        elif args.decode:
            args.workload = "decode-moe"
        else:
            args.workload = "moe"

    # Handle --sweep-flags mode separately
    if args.sweep_flags:
        sweep_compiler_flags(args)
        return

    # Default to sweeping tiles only if nothing specified
    if not (args.sweep_tiles or args.sweep_clusters or args.sweep_schedules or args.sweep_all):
        args.sweep_tiles = True

    if args.sweep_all:
        args.sweep_tiles = args.sweep_clusters = args.sweep_schedules = True

    # Workload groups for --all-workloads
    decode_largek_workloads = ["decode-lora-A-qwen4b", "decode-moe"]
    decode_smallk_workloads = ["decode-lora-B-qwen4b", "decode-lora-B-mlp-qwen4b"]
    prefill_workloads = ["moe", "lora", "lora-moe"]

    if args.decode_smallk:
        mode_label = "Decode SmallK"
    elif args.decode:
        mode_label = "Decode"
    else:
        mode_label = "Prefill"
    print(f"CUTLASS {mode_label} Kernel Parameter Sweep")
    print("=" * 80)

    if args.all_workloads:
        if args.decode_smallk:
            workloads = decode_smallk_workloads
        elif args.decode:
            workloads = decode_largek_workloads
        else:
            workloads = prefill_workloads
    else:
        workloads = [args.workload]

    # Build parameter combinations
    if args.decode_smallk:
        base_tiles = DECODE_SMALLK_TILE_CONFIGS
    elif args.decode and args.phase2:
        base_tiles = DECODE_TILE_CONFIGS_PHASE2
    elif args.decode:
        base_tiles = DECODE_TILE_CONFIGS
    else:
        base_tiles = TILE_CONFIGS

    tiles = base_tiles if args.sweep_tiles else [base_tiles[0]]
    clusters = CLUSTER_CONFIGS if args.sweep_clusters else [CLUSTER_CONFIGS[0]]
    schedules = SCHEDULE_CONFIGS if args.sweep_schedules else [SCHEDULE_CONFIGS[0]]

    configs = list(product(tiles, clusters, schedules))
    print(f"Testing {len(configs)} configurations across {len(workloads)} workload(s)")
    print()

    if args.dry_run:
        print("Configurations to test:")
        for tile, cluster, schedule in configs:
            schedule_name = schedule[0].replace("KernelPtrArrayTmaWarpSpecialized", "")
            print(f"  Tile{tile} Cluster{cluster} {schedule_name}")
        print()
        print("Workloads to test:")
        for w in workloads:
            print(f"  {w}: {WORKLOAD_PRESETS[w]['description']}")
        return

    # Select the right config setter for prefill vs decode vs decode-smallk
    if args.decode_smallk:
        config_setter = set_decode_smallk_kernel_config
    elif args.decode:
        config_setter = set_decode_kernel_config
    else:
        config_setter = set_kernel_config

    # Store results: {workload: [(config_str, bench_results), ...]}
    all_results: dict[str, list] = {w: [] for w in workloads}

    for tile, cluster, schedule in configs:
        print(f"\n{'='*80}")
        config_str = config_setter(tile, cluster, schedule)
        print(f"Testing: {config_str}")
        print("-" * 80)

        if not rebuild_kernel():
            print("  FAILED to build")
            for w in workloads:
                all_results[w].append((config_str, None))
            continue

        for workload in workloads:
            print(f"\n  Workload: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
            bench_results = run_benchmark(workload)
            if bench_results:
                print(f"    Forward:  {bench_results.get('fwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('fwd_tflops', 'N/A'):>8.2f} TFLOPS")
                print(f"    Backward: {bench_results.get('bwd_ms', 'N/A'):>8.3f} ms  {bench_results.get('bwd_tflops', 'N/A'):>8.2f} TFLOPS")
                all_results[workload].append((config_str, bench_results))
            else:
                print("    FAILED to benchmark")
                all_results[workload].append((config_str, None))

    # Summary per workload
    for workload in workloads:
        results = all_results[workload]
        print(f"\n{'='*80}")
        print(f"SUMMARY: {workload} ({WORKLOAD_PRESETS[workload]['description']})")
        print("=" * 80)
        print(f"{'Configuration':<50} {'Fwd (ms)':>10} {'Fwd TF':>8} {'Bwd (ms)':>10} {'Bwd TF':>8}")
        print("-" * 80)

        for config_str, res in results:
            if res:
                fwd_ms = f"{res.get('fwd_ms', 0):.3f}"
                fwd_tf = f"{res.get('fwd_tflops', 0):.1f}"
                bwd_ms = f"{res.get('bwd_ms', 0):.3f}"
                bwd_tf = f"{res.get('bwd_tflops', 0):.1f}"
                print(f"{config_str:<50} {fwd_ms:>10} {fwd_tf:>8} {bwd_ms:>10} {bwd_tf:>8}")
            else:
                print(f"{config_str:<50} {'FAILED':>10}")

        # Find best
        valid_results = [(cfg, r) for cfg, r in results if r and 'fwd_tflops' in r]
        if valid_results:
            best_fwd = max(valid_results, key=lambda x: x[1]['fwd_tflops'])
            best_bwd = max(valid_results, key=lambda x: x[1].get('bwd_tflops', 0))
            print()
            print(f"Best forward:  {best_fwd[0]} - {best_fwd[1]['fwd_tflops']:.2f} TFLOPS")
            print(f"Best backward: {best_bwd[0]} - {best_bwd[1].get('bwd_tflops', 0):.2f} TFLOPS")


if __name__ == "__main__":
    main()
