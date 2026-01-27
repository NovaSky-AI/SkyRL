"""
Test equivalence between cutile-based LoRA computation and ragged_dot.

This test suite verifies that the cutile implementation produces numerically
equivalent results to the existing JAX ragged_dot implementation for LoRA
expert parallelism computation.

Phase 1 Scope: Single-GPU forward pass only (no group_offset, no gradients).
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ImportError:
    pytest.skip("JAX not available", allow_module_level=True)

try:
    import torch
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)

try:
    from tx.kernels.cutile_lora import cutile_ragged_dot
    from tx.kernels import CUTILE_AVAILABLE
except ImportError:
    CUTILE_AVAILABLE = False
    cutile_ragged_dot = None
    pytestmark = pytest.mark.skip("Cutile implementation not available")

from tx.layers.util import ragged_dot


# ============================================================================
# Test Utilities
# ============================================================================


def generate_ragged_test_case(
    m: int,
    d: int,
    out_features: int,
    num_experts: int,
    seed: int = 42,
    distribution: str = "balanced",
    dtype=jnp.float32,
):
    """Generate synthetic test case for ragged_dot equivalence testing.

    Args:
        m: Total number of tokens
        d: Hidden dimension (input features)
        out_features: Output dimension
        num_experts: Number of expert groups
        seed: Random seed for reproducibility
        distribution: Token distribution strategy:
            - "balanced": Equal tokens per expert
            - "imbalanced": Random uneven distribution
            - "sparse": Some experts have zero tokens
        dtype: Data type for arrays (jnp.float32, jnp.float16, jnp.bfloat16)

    Returns:
        Tuple of (lhs, rhs, group_sizes) where:
            lhs: [m, d] input tokens
            rhs: [num_experts, d, out_features] expert weights
            group_sizes: [num_experts] number of tokens per expert
    """
    key = random.PRNGKey(seed)
    key_lhs, key_rhs, key_sizes = random.split(key, 3)

    # Generate input tokens
    lhs = random.normal(key_lhs, (m, d), dtype=dtype)

    # Generate expert weights
    rhs = random.normal(key_rhs, (num_experts, d, out_features), dtype=dtype) * 0.02

    # Generate group sizes based on distribution strategy
    if distribution == "balanced":
        base_size = m // num_experts
        remainder = m % num_experts
        group_sizes = jnp.array([base_size + (1 if i < remainder else 0) for i in range(num_experts)], dtype=jnp.int32)
    elif distribution == "imbalanced":
        # Random distribution ensuring sum equals m
        random_weights = random.uniform(key_sizes, (num_experts,))
        random_weights = random_weights / random_weights.sum()
        group_sizes = (random_weights * m).astype(jnp.int32)
        # Adjust to ensure exact sum
        diff = m - group_sizes.sum()
        group_sizes = group_sizes.at[0].add(diff)
    elif distribution == "sparse":
        # Some experts get zero tokens
        active_experts = num_experts // 2
        active_sizes = jnp.array([m // active_experts] * active_experts, dtype=jnp.int32)
        # Adjust for remainder
        active_sizes = active_sizes.at[0].add(m - active_sizes.sum())
        group_sizes = jnp.concatenate([active_sizes, jnp.zeros(num_experts - active_experts, dtype=jnp.int32)])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return lhs, rhs, group_sizes


def compare_outputs(
    cutile_out: jax.Array,
    ragged_out: jax.Array,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
):
    """Compare cutile and ragged_dot outputs with detailed error reporting.

    Args:
        cutile_out: Output from cutile implementation
        ragged_out: Output from ragged_dot
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Print detailed error statistics

    Raises:
        AssertionError: If outputs don't match within tolerance
    """
    # Ensure both are JAX arrays
    if isinstance(cutile_out, torch.Tensor):
        # This shouldn't happen, but handle gracefully
        cutile_out = jnp.array(cutile_out.detach().cpu().numpy())

    # Check shapes match
    assert (
        cutile_out.shape == ragged_out.shape
    ), f"Shape mismatch: cutile {cutile_out.shape} vs ragged {ragged_out.shape}"

    # Check dtypes match
    assert (
        cutile_out.dtype == ragged_out.dtype
    ), f"Dtype mismatch: cutile {cutile_out.dtype} vs ragged {ragged_out.dtype}"

    # Compute element-wise differences
    abs_diff = jnp.abs(cutile_out - ragged_out)
    rel_diff = abs_diff / (jnp.abs(ragged_out) + 1e-8)

    # Statistics
    max_abs_diff = jnp.max(abs_diff)
    max_rel_diff = jnp.max(rel_diff)
    mean_abs_diff = jnp.mean(abs_diff)
    mean_rel_diff = jnp.mean(rel_diff)

    if verbose:
        print("\nNumerical Comparison:")
        print(f"  Max absolute diff: {max_abs_diff:.6e}")
        print(f"  Max relative diff: {max_rel_diff:.6e}")
        print(f"  Mean absolute diff: {mean_abs_diff:.6e}")
        print(f"  Mean relative diff: {mean_rel_diff:.6e}")
        print(f"  Tolerance: rtol={rtol}, atol={atol}")

    # Check if within tolerance
    try:
        np.testing.assert_allclose(
            cutile_out,
            ragged_out,
            rtol=rtol,
            atol=atol,
            err_msg=f"Outputs differ beyond tolerance (max rel diff: {max_rel_diff:.6e})",
        )
        if verbose:
            print("  ✓ PASS: Within tolerance")
    except AssertionError:
        # Find indices of largest errors
        error_mask = rel_diff > rtol
        num_errors = jnp.sum(error_mask)
        print(f"  ✗ FAIL: {num_errors}/{cutile_out.size} elements exceed tolerance")

        # Show a few worst offenders
        flat_rel_diff = rel_diff.flatten()
        worst_indices = jnp.argsort(flat_rel_diff)[-5:]
        print("\n  Worst 5 errors:")
        for idx in worst_indices:
            i = int(idx)
            print(
                f"    Index {i}: cutile={float(cutile_out.flatten()[i]):.6e}, "
                f"ragged={float(ragged_out.flatten()[i]):.6e}, "
                f"rel_diff={float(flat_rel_diff[i]):.6e}"
            )
        raise


def benchmark_both(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    num_runs: int = 100,
    warmup: int = 10,
):
    """Benchmark latency of ragged_dot vs cutile.

    Args:
        lhs: Input tokens
        rhs: Expert weights
        group_sizes: Group sizes
        num_runs: Number of benchmark iterations
        warmup: Number of warmup iterations

    Returns:
        Tuple of (ragged_time_ms, cutile_time_ms, speedup)
    """
    import time

    # Warmup ragged_dot
    for _ in range(warmup):
        _ = ragged_dot(lhs, rhs, group_sizes)
        jax.block_until_ready(_)

    # Benchmark ragged_dot
    start = time.perf_counter()
    for _ in range(num_runs):
        out = ragged_dot(lhs, rhs, group_sizes)
        jax.block_until_ready(out)
    ragged_time = (time.perf_counter() - start) * 1000 / num_runs

    # Warmup cutile
    for _ in range(warmup):
        _ = cutile_ragged_dot(lhs, rhs, group_sizes)
        jax.block_until_ready(_)

    # Benchmark cutile
    start = time.perf_counter()
    for _ in range(num_runs):
        out = cutile_ragged_dot(lhs, rhs, group_sizes)
        jax.block_until_ready(out)
    cutile_time = (time.perf_counter() - start) * 1000 / num_runs

    speedup = ragged_time / cutile_time

    print("\nBenchmark Results:")
    print(f"  ragged_dot: {ragged_time:.3f} ms")
    print(f"  cutile:     {cutile_time:.3f} ms")
    print(f"  Speedup:    {speedup:.2f}x")

    return ragged_time, cutile_time, speedup


# ============================================================================
# Test Cases
# ============================================================================


@pytest.mark.skipif(not CUTILE_AVAILABLE, reason="Cutile not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCutileRaggedDotEquivalence:
    """Test cutile_ragged_dot matches ragged_dot output."""

    def test_basic_single_expert(self):
        """Simple case: all tokens to one expert."""
        m, d, out_features, num_experts = 128, 64, 32, 1

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=42, distribution="balanced"
        )

        # Run both implementations
        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        # Compare
        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_multiple_experts_balanced(self):
        """Multiple experts with equal token distribution."""
        m, d, out_features, num_experts = 256, 128, 128, 4

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=42, distribution="balanced"
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_multiple_experts_imbalanced(self):
        """Realistic case: uneven token distribution."""
        m, d, out_features, num_experts = 512, 256, 256, 8

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=123, distribution="imbalanced"
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_empty_expert_groups(self):
        """Edge case: some experts have zero tokens."""
        m, d, out_features, num_experts = 256, 128, 128, 8

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=456, distribution="sparse"
        )

        # Verify we actually have empty groups
        assert jnp.any(group_sizes == 0), "Test case should have empty groups"

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_many_small_groups(self):
        """Many experts with few tokens each."""
        m, d, out_features, num_experts = 512, 128, 128, 64

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=789, distribution="balanced"
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_few_large_groups(self):
        """Few experts with many tokens each."""
        m, d, out_features, num_experts = 2048, 256, 256, 4

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=101, distribution="balanced"
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        "d,out_features",
        [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ],
    )
    def test_various_dimensions(self, d, out_features):
        """Test different hidden dimensions."""
        m, num_experts = 256, 4

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=202, distribution="balanced"
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        "dtype",
        [
            jnp.float32,
            pytest.param(jnp.float16, marks=pytest.mark.xfail(reason="fp16 may have numerical issues")),
            pytest.param(jnp.bfloat16, marks=pytest.mark.xfail(reason="bf16 may have numerical issues")),
        ],
    )
    def test_dtype_preservation(self, dtype):
        """Verify dtype (fp32, fp16, bf16) preserved."""
        m, d, out_features, num_experts = 256, 128, 128, 4

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=303, distribution="balanced", dtype=dtype
        )

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        # Check dtype preserved
        assert (
            cutile_out.dtype == ragged_out.dtype == dtype
        ), f"Dtype not preserved: input={dtype}, ragged={ragged_out.dtype}, cutile={cutile_out.dtype}"

        # Looser tolerance for fp16/bf16
        rtol = 1e-2 if dtype in [jnp.float16, jnp.bfloat16] else 1e-3
        compare_outputs(cutile_out, ragged_out, rtol=rtol, atol=1e-4)

    def test_device_placement(self):
        """Ensure output stays on same GPU."""
        m, d, out_features, num_experts = 256, 128, 128, 4

        lhs, rhs, group_sizes = generate_ragged_test_case(
            m, d, out_features, num_experts, seed=404, distribution="balanced"
        )

        # Ensure inputs are on GPU
        device = jax.devices("gpu")[0]
        lhs = jax.device_put(lhs, device)
        rhs = jax.device_put(rhs, device)
        group_sizes = jax.device_put(group_sizes, device)

        ragged_out = ragged_dot(lhs, rhs, group_sizes)
        cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

        # Check device placement
        assert (
            cutile_out.device == ragged_out.device == device
        ), f"Device mismatch: input={device}, ragged={ragged_out.device}, cutile={cutile_out.device}"

        compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5)

    def test_benchmark_performance(self, capsys):
        """Benchmark cutile vs ragged_dot performance across multiple realistic configurations."""
        # Realistic configurations based on common LLM architectures
        # Format: (m, d, out_features, num_experts, description)
        configs = [
            (1024, 512, 512, 16, "Small (original)"),
            (2048, 1024, 1024, 16, "Medium (Qwen-0.6B scale)"),
            (4096, 1536, 1536, 32, "Large (Qwen2.5-1.5B scale)"),
            (4096, 2048, 2048, 32, "Large+ (2B scale)"),
            (8192, 4096, 4096, 64, "XLarge (Llama 3 8B scale)"),
        ]

        results = []

        # Temporarily disable output capturing to show benchmark results
        with capsys.disabled():
            print(f"\n{'='*80}")
            print(f"{'CUTILE vs RAGGED_DOT BENCHMARK SUITE':^80}")
            print(f"{'='*80}")

            for m, d, out_features, num_experts, desc in configs:
                lhs, rhs, group_sizes = generate_ragged_test_case(
                    m, d, out_features, num_experts, seed=505, distribution="imbalanced"
                )

                print(f"\n{desc}")
                print(f"  Config: {m} tokens × {d} hidden → {out_features} out, {num_experts} experts")
                print(f"  {'-'*76}")

                ragged_time, cutile_time, speedup = benchmark_both(lhs, rhs, group_sizes, num_runs=50, warmup=5)

                # Store results
                results.append(
                    {
                        "config": desc,
                        "m": m,
                        "d": d,
                        "num_experts": num_experts,
                        "ragged_time": ragged_time,
                        "cutile_time": cutile_time,
                        "speedup": speedup,
                    }
                )

                # Basic sanity check
                assert cutile_time > 0, f"Cutile execution failed for {desc}"
                assert ragged_time > 0, f"Ragged_dot execution failed for {desc}"

            # Print summary table
            print(f"\n{'='*80}")
            print(f"{'SUMMARY':^80}")
            print(f"{'='*80}")
            print(f"{'Config':<20} {'Tokens':>8} {'Hidden':>8} {'Experts':>8} {'Speedup':>10}")
            print(f"{'-'*80}")
            for r in results:
                status = "✓" if r["speedup"] > 1.0 else "⚠"
                print(
                    f"{r['config']:<20} {r['m']:>8} {r['d']:>8} {r['num_experts']:>8} "
                    f"{status} {r['speedup']:>7.2f}x"
                )
            print(f"{'='*80}\n")


# ============================================================================
# CLI for Manual Testing
# ============================================================================

if __name__ == "__main__":
    """Run basic equivalence test from command line."""
    print("Running basic cutile vs ragged_dot equivalence test...")

    # Simple test case
    m, d, out_features, num_experts = 256, 128, 128, 4
    lhs, rhs, group_sizes = generate_ragged_test_case(m, d, out_features, num_experts, seed=42, distribution="balanced")

    print("\nTest configuration:")
    print(f"  Tokens (m): {m}")
    print(f"  Hidden dim (d): {d}")
    print(f"  Output dim: {out_features}")
    print(f"  Num experts: {num_experts}")
    print(f"  Group sizes: {group_sizes}")

    # Run both
    print("\nRunning ragged_dot...")
    ragged_out = ragged_dot(lhs, rhs, group_sizes)

    print("Running cutile...")
    cutile_out = cutile_ragged_dot(lhs, rhs, group_sizes)

    # Compare
    print("\nComparing outputs...")
    compare_outputs(cutile_out, ragged_out, rtol=1e-3, atol=1e-5, verbose=True)

    print("\n✓ Basic test PASSED!")
