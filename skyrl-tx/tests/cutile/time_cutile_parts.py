import torch

# Import your functions
# Adjust these imports to match your package structure.
from yourpkg.cutile_lora import _pad_groups_to_tile_m
from yourpkg.cutile_lora_kernels import launch_cutile_lora_gemm, CUTILE_AVAILABLE
from yourpkg.cutile_config import config as default_config


def cuda_time_ms(fn, iters=100, warmup=10):
    """Time fn() using CUDA events. fn must enqueue CUDA work."""
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starter.record()
    for _ in range(iters):
        fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not CUTILE_AVAILABLE:
        raise RuntimeError("CUTILE not available")

    device = "cuda"

    # ---- Configure shapes (match one of your benchmarks) ----
    # Example: Medium: 2048 tokens × 1024 hidden → 1024 out, 16 experts
    m = 2048
    d = 1024
    out = 1024
    E = 16

    dtype = torch.float16  # try torch.bfloat16 too
    TILE_M = int(getattr(default_config, "tile_m"))
    TILE_N = int(getattr(default_config, "tile_n"))
    TILE_K = int(getattr(default_config, "tile_k"))

    # ---- Create synthetic inputs ----
    lhs = torch.randn((m, d), device=device, dtype=dtype)

    # weights should be contiguous in the layout your kernel expects
    weights = torch.randn((E, d, out), device=device, dtype=dtype).contiguous()

    # group sizes that sum to m (contiguous groups in lhs order)
    # Example: random-ish routing
    gs = torch.randint(low=0, high=max(1, m // E * 2), size=(E,), device=device, dtype=torch.int32)
    # normalize so sum == m
    s = int(gs.sum().item())
    if s == 0:
        gs[0] = m
    else:
        # scale and fix remainder
        gs = (gs.to(torch.float32) * (m / float(s))).to(torch.int32)
        diff = m - int(gs.sum().item())
        gs[0] += diff
        gs = torch.clamp(gs, min=0)
    assert int(gs.sum().item()) == m, (int(gs.sum().item()), m)

    print(f"TILE_M/N/K = {TILE_M}/{TILE_N}/{TILE_K}")
    print(f"lhs: {tuple(lhs.shape)} {lhs.dtype}")
    print(f"weights: {tuple(weights.shape)} contiguous={weights.is_contiguous()} stride={weights.stride()}")
    print(f"group_sizes sum={int(gs.sum().item())} E={gs.numel()} dtype={gs.dtype}")

    # ---- 1) Time padding + expert_ids construction ----
    def do_pad():
        _pad_groups_to_tile_m(lhs, gs, tile_m=TILE_M)

    pad_ms = cuda_time_ms(do_pad, iters=200, warmup=20)

    # Prepare padded inputs once for kernel-only timing
    lhs_padded, expert_ids_per_tile = _pad_groups_to_tile_m(lhs, gs, tile_m=TILE_M)
    m_padded = lhs_padded.shape[0]
    out_buf = torch.empty((m_padded, out), device=device, dtype=dtype)

    # ---- 2) Time cutile launch only ----
    def do_launch():
        launch_cutile_lora_gemm(
            lhs_padded,
            weights,
            out_buf,
            expert_ids_per_tile,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
        )

    launch_ms = cuda_time_ms(do_launch, iters=200, warmup=20)

    # ---- 3) Time combined (pad + launch) ----
    def do_pad_and_launch():
        lp, ept = _pad_groups_to_tile_m(lhs, gs, tile_m=TILE_M)
        ob = torch.empty((lp.shape[0], out), device=device, dtype=dtype)
        launch_cutile_lora_gemm(
            lp,
            weights,
            ob,
            ept,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
        )

    combined_ms = cuda_time_ms(do_pad_and_launch, iters=200, warmup=20)

    print("\n=== Timing (CUDA events) ===")
    print(f"pad_groups:   {pad_ms:.3f} ms")
    print(f"cutile_launch:{launch_ms:.3f} ms")
    print(f"combined:     {combined_ms:.3f} ms")
    print(f"pad% of combined: {100.0 * pad_ms / max(combined_ms, 1e-9):.1f}%")
    print(f"launch% of combined: {100.0 * launch_ms / max(combined_ms, 1e-9):.1f}%")

    # sanity: combined should be roughly pad+launch (+alloc noise)
    print(f"(pad+launch) ~ {pad_ms + launch_ms:.3f} ms")


if __name__ == "__main__":
    main()
