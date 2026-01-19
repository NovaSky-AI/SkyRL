"""Benchmark ragged_dot CUTLASS kernel with Qwen3-30B-A3B MoE shapes."""

import argparse
import time

import jax
import jax.numpy as jnp
from jax import lax

from tx.ffi import ragged_dot_ffi, ragged_dot_ffi_available


def generate_group_sizes(num_tokens: int, num_experts: int, key: jax.Array) -> jax.Array:
    """Generate random group sizes that sum to num_tokens."""
    # Random assignment of tokens to experts
    assignments = jax.random.randint(key, (num_tokens,), 0, num_experts)
    return jnp.bincount(assignments, length=num_experts).astype(jnp.int32)


def benchmark_forward(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    use_ffi: bool = True,
):
    """Benchmark forward pass: lhs[M, K] @ rhs[G, K, N] -> out[M, N]."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16)
    group_sizes = generate_group_sizes(num_tokens, num_experts, k3)
    group_offset = jnp.array([0], dtype=jnp.int32)

    if use_ffi:
        fn = lambda: ragged_dot_ffi(lhs, rhs, group_sizes, group_offset)
    else:
        fn = lambda: lax.ragged_dot(lhs, rhs, group_sizes)

    # Warmup
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
    flops = 2 * num_tokens * hidden_size * intermediate_size
    tflops = (flops * num_iters / elapsed) / 1e12

    return elapsed / num_iters, tflops


def benchmark_backward(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_warmup: int = 5,
    num_iters: int = 20,
    use_ffi: bool = True,
):
    """Benchmark backward pass through ragged_dot."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    lhs = jax.random.normal(k1, (num_tokens, hidden_size), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (num_experts, hidden_size, intermediate_size), dtype=jnp.bfloat16)
    group_sizes = generate_group_sizes(num_tokens, num_experts, k3)
    group_offset = jnp.array([0], dtype=jnp.int32)

    if use_ffi:
        def forward(lhs, rhs):
            return ragged_dot_ffi(lhs, rhs, group_sizes, group_offset).sum()
    else:
        def forward(lhs, rhs):
            return lax.ragged_dot(lhs, rhs, group_sizes).sum()

    grad_fn = jax.grad(forward, argnums=(0, 1))

    # Warmup
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
    flops = 4 * num_tokens * hidden_size * intermediate_size
    tflops = (flops * num_iters / elapsed) / 1e12

    return elapsed / num_iters, tflops


def main():
    parser = argparse.ArgumentParser(description="Benchmark ragged_dot CUTLASS kernel")
    parser.add_argument("--num-tokens", type=int, default=8192, help="Number of tokens (M)")
    parser.add_argument("--num-experts", type=int, default=128, help="Number of experts (G)")
    parser.add_argument("--hidden-size", type=int, default=2048, help="Hidden size (K)")
    parser.add_argument("--intermediate-size", type=int, default=768, help="MoE intermediate size (N)")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--backward-only", action="store_true", help="Only benchmark backward pass")
    parser.add_argument("--forward-only", action="store_true", help="Only benchmark forward pass")
    args = parser.parse_args()

    print("Ragged Dot Benchmark (Qwen3-30B-A3B MoE shapes)")
    print("=" * 60)
    print(f"CUTLASS FFI available: {ragged_dot_ffi_available()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.device_count()}")
    print()
    print(f"Config:")
    print(f"  num_tokens (M):       {args.num_tokens}")
    print(f"  num_experts (G):      {args.num_experts}")
    print(f"  hidden_size (K):      {args.hidden_size}")
    print(f"  intermediate_size (N): {args.intermediate_size}")
    print(f"  warmup/iters:         {args.num_warmup}/{args.num_iters}")
    print()

    run_forward = not args.backward_only
    run_backward = not args.forward_only

    if run_forward:
        print("Forward Pass (lhs[M,K] @ rhs[G,K,N] -> out[M,N])")
        print("-" * 60)

        if ragged_dot_ffi_available():
            ffi_time, ffi_tflops = benchmark_forward(
                args.num_tokens, args.hidden_size, args.intermediate_size,
                args.num_experts, args.num_warmup, args.num_iters, use_ffi=True
            )
            print(f"  CUTLASS FFI:  {ffi_time*1000:8.3f} ms  {ffi_tflops:8.2f} TFLOPS")

        jax_time, jax_tflops = benchmark_forward(
            args.num_tokens, args.hidden_size, args.intermediate_size,
            args.num_experts, args.num_warmup, args.num_iters, use_ffi=False
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
                args.num_tokens, args.hidden_size, args.intermediate_size,
                args.num_experts, args.num_warmup, args.num_iters, use_ffi=True
            )
            print(f"  CUTLASS FFI:  {ffi_time*1000:8.3f} ms  {ffi_tflops:8.2f} TFLOPS")

        jax_time, jax_tflops = benchmark_backward(
            args.num_tokens, args.hidden_size, args.intermediate_size,
            args.num_experts, args.num_warmup, args.num_iters, use_ffi=False
        )
        print(f"  JAX ragged:   {jax_time*1000:8.3f} ms  {jax_tflops:8.2f} TFLOPS")

        if ragged_dot_ffi_available():
            print(f"  Speedup:      {jax_time/ffi_time:.2f}x")
        print()


if __name__ == "__main__":
    main()
