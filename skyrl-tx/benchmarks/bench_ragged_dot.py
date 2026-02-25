"""Benchmark ragged_dot CUTLASS kernel with Qwen3-30B-A3B MoE and LoRA shapes."""

import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax

from tx.ffi import ragged_dot_ffi, ragged_dot_ffi_available


# Preset configurations for different workloads
PRESETS = {
    "decode-moe": {
        "description": "Decode-time MoE expert layer (small/sparse token batches)",
        "num_tokens": 64,
        "num_groups": 128,  # num_experts
        "k_dim": 2048,      # hidden_size
        "n_dim": 768,       # intermediate_size
    },
    "decode-lora": {
        "description": "Decode-time LoRA adapter layer (small token batches)",
        "num_tokens": 64,
        "num_groups": 32,   # max_lora_adapters
        "k_dim": 8,         # lora_rank
        "n_dim": 4096,      # output features
    },
    "decode-lora-A-qwen4b": {
        "description": "Decode LoRA A pass on Qwen3-4B attn (x @ lora_A)",
        "num_tokens": 256,
        "num_groups": 3,    # max_lora_adapters (RL workload)
        "k_dim": 2560,      # hidden_size
        "n_dim": 32,        # lora_rank
    },
    "decode-lora-B-qwen4b": {
        "description": "Decode LoRA B pass on Qwen3-4B attn (intermediate @ lora_B)",
        "num_tokens": 256,
        "num_groups": 3,    # max_lora_adapters (RL workload)
        "k_dim": 32,        # lora_rank
        "n_dim": 2560,      # hidden_size
    },
    "decode-lora-B-mlp-qwen4b": {
        "description": "Decode LoRA B pass on Qwen3-4B MLP (intermediate @ lora_B)",
        "num_tokens": 256,
        "num_groups": 3,    # max_lora_adapters (RL workload)
        "k_dim": 32,        # lora_rank
        "n_dim": 9728,      # intermediate_size
    },
    "moe": {
        "description": "MoE expert layer (Qwen3-30B-A3B)",
        "num_tokens": 8192,
        "num_groups": 128,  # num_experts
        "k_dim": 2048,      # hidden_size
        "n_dim": 768,       # intermediate_size
    },
    "lora": {
        "description": "LoRA adapter layer",
        "num_tokens": 8192,
        "num_groups": 32,   # max_lora_adapters
        "k_dim": 8,         # lora_rank
        "n_dim": 4096,      # output features
    },
    "lora-moe": {
        "description": "LoRA on MoE experts (combined groups)",
        "num_tokens": 8192,
        "num_groups": 1024, # num_experts * max_lora_adapters (128 * 8, capped at kernel limit)
        "k_dim": 8,         # lora_rank
        "n_dim": 768,       # intermediate_size
    },
}


def generate_group_sizes(num_tokens: int, num_groups: int, key: jax.Array) -> jax.Array:
    """Generate random group sizes that sum to num_tokens."""
    # Random assignment of tokens to groups
    assignments = jax.random.randint(key, (num_tokens,), 0, num_groups)
    return jnp.bincount(assignments, length=num_groups).astype(jnp.int32)


def benchmark_forward(
    num_tokens: int,
    k_dim: int,
    n_dim: int,
    num_groups: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    use_ffi: bool = True,
):
    """Benchmark forward pass: lhs[M, K] @ rhs[G, K, N] -> out[M, N]."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (num_tokens, k_dim), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_groups, k_dim, n_dim), dtype=jnp.bfloat16)
    group_sizes = generate_group_sizes(num_tokens, num_groups, k3)
    group_offset = jnp.array([0], dtype=jnp.int32)

    if use_ffi:
        @jax.jit
        def fn():
            return ragged_dot_ffi(lhs, rhs, group_sizes, group_offset)
    else:
        @jax.jit
        def fn():
            return lax.ragged_dot(lhs, rhs, group_sizes)

    # Warmup (includes JIT compilation on first call)
    for _ in range(num_warmup):
        out = fn()
        out.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        out = fn()
        out.block_until_ready()
    elapsed = time.perf_counter() - start

    # FLOPs: 2 * M * K * N (matmul FLOPs)
    flops = 2 * num_tokens * k_dim * n_dim
    tflops = (flops * num_iters / elapsed) / 1e12

    return elapsed / num_iters, tflops


def benchmark_backward(
    num_tokens: int,
    k_dim: int,
    n_dim: int,
    num_groups: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    use_ffi: bool = True,
):
    """Benchmark backward pass through ragged_dot."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (num_tokens, k_dim), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_groups, k_dim, n_dim), dtype=jnp.bfloat16)
    group_sizes = generate_group_sizes(num_tokens, num_groups, k3)
    group_offset = jnp.array([0], dtype=jnp.int32)

    if use_ffi:
        def forward(lhs, rhs):
            return ragged_dot_ffi(lhs, rhs, group_sizes, group_offset).sum()
    else:
        def forward(lhs, rhs):
            return lax.ragged_dot(lhs, rhs, group_sizes).sum()

    grad_fn = jax.jit(jax.grad(forward, argnums=(0, 1)))

    # Warmup (includes JIT compilation on first call)
    for _ in range(num_warmup):
        d_lhs, d_rhs = grad_fn(lhs, rhs)
        d_lhs.block_until_ready()
        d_rhs.block_until_ready()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        d_lhs, d_rhs = grad_fn(lhs, rhs)
        d_lhs.block_until_ready()
        d_rhs.block_until_ready()
    elapsed = time.perf_counter() - start

    # Backward FLOPs: d_lhs = grad @ rhs.T (2*M*N*K) + d_rhs = lhs.T @ grad (2*K*M*N)
    # Total: 4 * M * K * N
    flops = 4 * num_tokens * k_dim * n_dim
    tflops = (flops * num_iters / elapsed) / 1e12

    return elapsed / num_iters, tflops


def run_benchmark_suite(
    num_tokens: int,
    k_dim: int,
    n_dim: int,
    num_groups: int,
    num_warmup: int,
    num_iters: int,
    run_forward: bool,
    run_backward: bool,
):
    """Run the benchmark suite with the given configuration."""
    if run_forward:
        print("Forward Pass (lhs[M,K] @ rhs[G,K,N] -> out[M,N])")
        print("-" * 60)

        if ragged_dot_ffi_available():
            ffi_time, ffi_tflops = benchmark_forward(
                num_tokens, k_dim, n_dim, num_groups, num_warmup, num_iters, use_ffi=True
            )
            print(f"  CUTLASS FFI:  {ffi_time*1000:8.3f} ms  {ffi_tflops:8.2f} TFLOPS")

        jax_time, jax_tflops = benchmark_forward(
            num_tokens, k_dim, n_dim, num_groups, num_warmup, num_iters, use_ffi=False
        )
        print(f"  JAX ragged:   {jax_time*1000:8.3f} ms  {jax_tflops:8.2f} TFLOPS")

        if ragged_dot_ffi_available():
            print(f"  Speedup:      {jax_time/ffi_time:.2f}x")
        print()

    if run_backward:
        print("Backward Pass (grad wrt lhs and rhs)")
        print("-" * 60)

        if ragged_dot_ffi_available():
            ffi_time, ffi_tflops = benchmark_backward(
                num_tokens, k_dim, n_dim, num_groups, num_warmup, num_iters, use_ffi=True
            )
            print(f"  CUTLASS FFI:  {ffi_time*1000:8.3f} ms  {ffi_tflops:8.2f} TFLOPS")

        jax_time, jax_tflops = benchmark_backward(
            num_tokens, k_dim, n_dim, num_groups, num_warmup, num_iters, use_ffi=False
        )
        print(f"  JAX ragged:   {jax_time*1000:8.3f} ms  {jax_tflops:8.2f} TFLOPS")

        if ragged_dot_ffi_available():
            print(f"  Speedup:      {jax_time/ffi_time:.2f}x")
        print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark ragged_dot CUTLASS kernel")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a preset configuration (moe, lora, lora-moe)")
    parser.add_argument("--all-presets", action="store_true", help="Run all preset configurations")
    parser.add_argument("--num-tokens", type=int, default=8192, help="Number of tokens (M)")
    parser.add_argument("--num-groups", type=int, default=128, help="Number of groups (G) - experts or adapters")
    parser.add_argument("--k-dim", type=int, default=2048, help="K dimension - hidden_size (MoE) or lora_rank (LoRA)")
    parser.add_argument("--n-dim", type=int, default=768, help="N dimension - output features")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--backward-only", action="store_true", help="Only benchmark backward pass")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward pass")
    args = parser.parse_args()

    print("Ragged Dot Benchmark")
    print("=" * 60)
    print(f"CUTLASS FFI available: {ragged_dot_ffi_available()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print()

    run_forward = not args.backward_only
    run_backward = not args.forward_only

    if args.all_presets:
        # Run all presets
        for preset_name, preset in PRESETS.items():
            print("=" * 60)
            print(f"Preset: {preset_name} - {preset['description']}")
            print("=" * 60)
            print(f"Config:")
            print(f"  num_tokens (M):  {preset['num_tokens']}")
            print(f"  num_groups (G):  {preset['num_groups']}")
            print(f"  k_dim (K):       {preset['k_dim']}")
            print(f"  n_dim (N):       {preset['n_dim']}")
            print(f"  warmup/iters:    {args.num_warmup}/{args.num_iters}")
            print()
            run_benchmark_suite(
                preset['num_tokens'], preset['k_dim'], preset['n_dim'], preset['num_groups'],
                args.num_warmup, args.num_iters, run_forward, run_backward
            )
    elif args.preset:
        # Use a specific preset
        preset = PRESETS[args.preset]
        print(f"Preset: {args.preset} - {preset['description']}")
        print()
        print(f"Config:")
        print(f"  num_tokens (M):  {preset['num_tokens']}")
        print(f"  num_groups (G):  {preset['num_groups']}")
        print(f"  k_dim (K):       {preset['k_dim']}")
        print(f"  n_dim (N):       {preset['n_dim']}")
        print(f"  warmup/iters:    {args.num_warmup}/{args.num_iters}")
        print()
        run_benchmark_suite(
            preset['num_tokens'], preset['k_dim'], preset['n_dim'], preset['num_groups'],
            args.num_warmup, args.num_iters, run_forward, run_backward
        )
    else:
        # Use custom config from args
        print(f"Config:")
        print(f"  num_tokens (M):  {args.num_tokens}")
        print(f"  num_groups (G):  {args.num_groups}")
        print(f"  k_dim (K):       {args.k_dim}")
        print(f"  n_dim (N):       {args.n_dim}")
        print(f"  warmup/iters:    {args.num_warmup}/{args.num_iters}")
        print()
        run_benchmark_suite(
            args.num_tokens, args.k_dim, args.n_dim, args.num_groups,
            args.num_warmup, args.num_iters, run_forward, run_backward
        )


if __name__ == "__main__":
    main()
