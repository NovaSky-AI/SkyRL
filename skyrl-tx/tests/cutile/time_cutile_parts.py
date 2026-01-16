"""
Time breakdown for cutile LoRA path:
- pad groups (_pad_groups_to_tile_m)
- cutile kernel launch (launch_cutile_lora_gemm)
- combined (pad + launch)

Uses CUDA events for accurate GPU timing.
"""

import sys

try:
    import jax
    import jax.numpy as jnp
    from jax import random
except ImportError:
    print("JAX not available")
    sys.exit(0)

try:
    import torch
except ImportError:
    print("PyTorch not available")
    sys.exit(0)

try:
    # IMPORTANT: match your test import style
    from tx.kernels import CUTILE_AVAILABLE
    from tx.kernels.cutile_lora import _pad_groups_to_tile_m
    from tx.kernels.cutile_lora_kernels import launch_cutile_lora_gemm
    from tx.kernels.cutile_config import config as default_config
except ImportError as e:
    print(f"Cutile implementation not available: {e}")
    sys.exit(0)


def cuda_time_ms(fn, iters=200, warmup=20) -> float:
    """Time fn() with CUDA events; fn must enqueue CUDA work."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def make_case(m, d, out_features, num_experts, seed=505, distribution="imbalanced", dtype=jnp.float16):
    key = random.PRNGKey(seed)
    key_lhs, key_rhs, key_sizes = random.split(key, 3)

    lhs = random.normal(key_lhs, (m, d), dtype=dtype)
    rhs = random.normal(key_rhs, (num_experts, d, out_features), dtype=dtype) * 0.02

    if distribution == "balanced":
        base = m // num_experts
        rem = m % num_experts
        gs = jnp.array([base + (1 if i < rem else 0) for i in range(num_experts)], dtype=jnp.int32)
    elif distribution == "imbalanced":
        w = random.uniform(key_sizes, (num_experts,))
        w = w / w.sum()
        gs = (w * m).astype(jnp.int32)
        diff = m - gs.sum()
        gs = gs.at[0].add(diff)
    elif distribution == "sparse":
        active = max(1, num_experts // 2)
        active_sizes = jnp.array([m // active] * active, dtype=jnp.int32)
        active_sizes = active_sizes.at[0].add(m - active_sizes.sum())
        gs = jnp.concatenate([active_sizes, jnp.zeros(num_experts - active, dtype=jnp.int32)])
    else:
        raise ValueError(distribution)

    return lhs, rhs, gs


def main():
    if not CUTILE_AVAILABLE:
        print("CUTILE_AVAILABLE is False")
        return
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    # Pick one config first
    m, d, out_features, num_experts = 2048, 1024, 1024, 16
    dtype = jnp.float16  # try jnp.bfloat16 too if you support it

    lhs_j, rhs_j, gs_j = make_case(m, d, out_features, num_experts, dtype=dtype)

    # Put on GPU (JAX)
    dev = jax.devices("gpu")[0]
    lhs_j = jax.device_put(lhs_j, dev)
    rhs_j = jax.device_put(rhs_j, dev)
    gs_j = jax.device_put(gs_j, dev)

    # Convert to torch via DLPack (same as your wrapper)
    lhs_t = torch.from_dlpack(lhs_j)
    rhs_t = torch.from_dlpack(rhs_j)
    gs_t = torch.from_dlpack(gs_j)

    # Make sure dtypes are what you expect
    if gs_t.dtype not in (torch.int32, torch.int64):
        gs_t = gs_t.to(torch.int32)

    TILE_M = int(getattr(default_config, "tile_m"))
    TILE_N = int(getattr(default_config, "tile_n"))
    TILE_K = int(getattr(default_config, "tile_k"))

    print(f"Config: m={m}, d={d}, out={out_features}, E={num_experts}, dtype={lhs_t.dtype}")
    print(f"TILE_M/N/K = {TILE_M}/{TILE_N}/{TILE_K}")
    print(f"rhs contiguous={rhs_t.is_contiguous()} stride={rhs_t.stride()}")

    # --------- 1) pad-only ----------
    def do_pad():
        _pad_groups_to_tile_m(lhs_t, gs_t, tile_m=TILE_M)

    pad_ms = cuda_time_ms(do_pad)

    # Prepare once for launch-only timing
    lhs_padded, expert_ids_per_tile = _pad_groups_to_tile_m(lhs_t, gs_t, tile_m=TILE_M)
    m_padded = lhs_padded.shape[0]
    out_t = torch.empty((m_padded, out_features), device=lhs_t.device, dtype=lhs_t.dtype)

    # --------- 2) launch-only ----------
    def do_launch():
        launch_cutile_lora_gemm(
            lhs_padded,
            rhs_t,
            out_t,
            expert_ids_per_tile,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
        )

    launch_ms = cuda_time_ms(do_launch)

    # --------- 3) combined ----------
    def do_combined():
        lp, ept = _pad_groups_to_tile_m(lhs_t, gs_t, tile_m=TILE_M)
        ob = torch.empty((lp.shape[0], out_features), device=lhs_t.device, dtype=lhs_t.dtype)
        launch_cutile_lora_gemm(
            lp,
            rhs_t,
            ob,
            ept,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
        )

    combined_ms = cuda_time_ms(do_combined)

    print("\n=== CUDA-event timing breakdown ===")
    print(f"pad_groups:    {pad_ms:.3f} ms")
    print(f"cutile_launch: {launch_ms:.3f} ms")
    print(f"combined:      {combined_ms:.3f} ms")
    print(f"pad fraction:  {100.0 * pad_ms / max(combined_ms, 1e-9):.1f}%")
    print(f"launch frac:   {100.0 * launch_ms / max(combined_ms, 1e-9):.1f}%")
    print(f"(pad+launch):  {pad_ms + launch_ms:.3f} ms (rough expected)")

    # Optional sanity: ensure nothing is silently syncing on CPU
    # If pad_ms is surprisingly large, it’s likely CPU sync from .cpu()/.tolist()/.item().


if __name__ == "__main__":
    main()
