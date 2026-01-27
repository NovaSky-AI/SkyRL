"""
JAX-PyTorch interop for cutile LoRA kernels (optimized wrapper).

Key changes vs prior version:
- Assumes ragged_dot-style contiguous grouping implied by `group_sizes`
  (tokens are already laid out group-by-group in order).
- Removes sort/unsort and avoids building per-token expert_ids.
- Pads each group to TILE_M boundary (linear-time, no O(m log m) sort).
- Uses fewer allocations and zero-fills only padded tails.

Note:
- This still crosses JAX<->Torch each call (DLPack). For benchmarking kernel-only,
  prefer a pure-torch harness using torch.cuda.Event timing.
"""

from __future__ import annotations

try:
    import jax
except ImportError as e:
    raise ImportError("JAX is required for cutile LoRA") from e

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required for cutile LoRA") from e

from .cutile_lora_kernels import (
    launch_cutile_lora_gemm,
    CUTILE_AVAILABLE,
)
from .cutile_config import config as default_config


# -----------------------------------------------------------------------------
# DLPack helpers
# -----------------------------------------------------------------------------


def jax_to_torch(jax_arr: "jax.Array") -> "torch.Tensor":
    device_str = str(jax_arr.device).lower()
    if "gpu" not in device_str and "cuda" not in device_str:
        raise ValueError(f"Expected GPU array, got device: {jax_arr.device}")
    try:
        return torch.from_dlpack(jax_arr)  # zero-copy when devices match
    except Exception as e:
        raise RuntimeError(f"DLPack conversion JAX->Torch failed: {e}") from e


def torch_to_jax(torch_tensor: "torch.Tensor") -> "jax.Array":
    if not torch_tensor.is_cuda:
        raise ValueError(f"Expected CUDA tensor, got device: {torch_tensor.device}")
    try:
        return jax.dlpack.from_dlpack(torch_tensor)
    except Exception as e:
        raise RuntimeError(f"DLPack conversion Torch->JAX failed: {e}") from e


# -----------------------------------------------------------------------------
# Group padding (no sort/unsort)
# -----------------------------------------------------------------------------

# tx/kernels/cutile_lora.py (near top-level)
_PAD_PLAN_CACHE = {}


def _make_pad_plan(group_sizes_cpu, tile_m):
    ps_cpu = [((g + tile_m - 1) // tile_m) * tile_m for g in group_sizes_cpu]
    m_padded = sum(ps_cpu)

    expert_ids_list = []
    for e, p in enumerate(ps_cpu):
        expert_ids_list.extend([e] * (p // tile_m))

    return ps_cpu, m_padded, expert_ids_list


def _pad_groups_to_tile_m(
    lhs: torch.Tensor,  # [m, d], groups contiguous in order
    group_sizes: torch.Tensor,  # [E], int32/int64 on CUDA
    tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      lhs_padded: [m_padded, d]
      expert_ids_per_tile: [num_tiles_total] int32
    """
    if group_sizes.numel() == 0:
        # Degenerate: no experts
        m, d = lhs.shape
        lhs_padded = lhs.new_empty((0, d))
        expert_ids_per_tile = torch.empty((0,), device=lhs.device, dtype=torch.int32)
        return lhs_padded, expert_ids_per_tile

    # Make sure group_sizes is on same device
    if group_sizes.device != lhs.device:
        group_sizes = group_sizes.to(device=lhs.device)

    # Compute padded sizes: ceil(gs / tile_m) * tile_m
    # Keep this on GPU to avoid sync; we only pull small scalars when looping over E.
    gs = group_sizes.to(dtype=torch.int64)
    padded_sizes = ((gs + tile_m - 1) // tile_m) * tile_m  # [E]
    # E is small; pulling scalar is fine
    m_padded = int(padded_sizes.sum().item())

    m, d = lhs.shape
    lhs_padded = torch.empty((m_padded, d), device=lhs.device, dtype=lhs.dtype)

    # Build expert_ids_per_tile on CPU list then upload (E is small, num_tiles ~ m/tile_m)
    # If this becomes a bottleneck, you can build it on GPU with a small kernel later.
    expert_ids_list: list[int] = []

    in_off = 0
    out_off = 0

    # Loop over experts (small)
    # E = int(gs.numel())
    # Pull to CPU once to avoid many device->host syncs
    gs_cpu = tuple(group_sizes.detach().cpu().tolist())
    key = (tile_m, gs_cpu, lhs.dtype)

    plan = _PAD_PLAN_CACHE.get(key)
    if plan is None:
        ps_cpu, m_padded, expert_ids_list = _make_pad_plan(gs_cpu, tile_m)
        expert_ids_per_tile = torch.tensor(expert_ids_list, device=lhs.device, dtype=torch.int32)
        plan = (ps_cpu, m_padded, expert_ids_per_tile)
        _PAD_PLAN_CACHE[key] = plan

    ps_cpu, m_padded, expert_ids_per_tile = plan

    m, d = lhs.shape
    lhs_padded = torch.empty((m_padded, d), device=lhs.device, dtype=lhs.dtype)

    in_off = 0
    out_off = 0
    for e, (g, p) in enumerate(zip(gs_cpu, ps_cpu)):
        if g:
            lhs_padded[out_off : out_off + g].copy_(lhs[in_off : in_off + g])
        tail = p - g
        if tail:
            lhs_padded[out_off + g : out_off + p].zero_()
        in_off += g
        out_off += p

    return lhs_padded, expert_ids_per_tile


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def cutile_ragged_dot(
    lhs: "jax.Array",  # [m, d]
    rhs: "jax.Array",  # [E, d, out]
    group_sizes: "jax.Array",  # [E]
    precision=None,  # ignored
    preferred_element_type=None,  # ignored
    group_offset: "jax.Array | None" = None,  # not supported in this wrapper
) -> "jax.Array":
    """
    Optimized drop-in replacement for ragged_dot when group_sizes implies
    contiguous groups in lhs order.

    Output order matches input order (no sort/unsort).
    """
    if not CUTILE_AVAILABLE:
        raise RuntimeError(
            "Cutile not available. Install with:\n"
            "  pip install cuda-tile\n"
            "Note: CUDA Toolkit 13.1+ is required (install separately)\n"
            "Or set TX_USE_CUTILE_LORA=0 to use ragged_dot"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Cutile requires NVIDIA GPU.")
    if group_offset is not None:
        raise NotImplementedError("group_offset not supported in this Phase-1 cutile implementation.")

    # Convert JAX arrays to Torch (zero-copy)
    lhs_t = jax_to_torch(lhs)
    rhs_t = jax_to_torch(rhs)
    gs_t = jax_to_torch(group_sizes)

    # Basic validation
    if lhs_t.ndim != 2:
        raise ValueError(f"lhs must be [m,d], got shape {tuple(lhs_t.shape)}")
    if rhs_t.ndim != 3:
        raise ValueError(f"rhs must be [E,d,out], got shape {tuple(rhs_t.shape)}")
    if gs_t.ndim != 1:
        raise ValueError(f"group_sizes must be [E], got shape {tuple(gs_t.shape)}")

    m, d = lhs_t.shape
    E, d2, out_features = rhs_t.shape
    if d2 != d:
        raise ValueError(f"rhs d ({d2}) must match lhs d ({d})")
    if gs_t.numel() != E:
        raise ValueError(f"group_sizes len ({gs_t.numel()}) must equal num experts ({E})")

    # Optional: ensure integer type
    if gs_t.dtype not in (torch.int32, torch.int64):
        gs_t = gs_t.to(torch.int32)

    # Pad groups to TILE_M boundary (no sort/unsort)
    TILE_M = int(getattr(default_config, "tile_m"))
    TILE_N = int(getattr(default_config, "tile_n"))
    TILE_K = int(getattr(default_config, "tile_k"))

    lhs_padded, expert_ids_per_tile = _pad_groups_to_tile_m(lhs_t, gs_t, tile_m=TILE_M)
    m_padded = lhs_padded.shape[0]

    # Allocate output (use empty; kernel will write all valid tiles)
    out_t = torch.empty((m_padded, out_features), device=lhs_t.device, dtype=lhs_t.dtype)

    # Launch kernel
    launch_cutile_lora_gemm(
        lhs_padded,
        rhs_t,
        out_t,
        expert_ids_per_tile,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
    )

    # Slice back to original m (order preserved)
    out_trim = out_t[:m]

    # Convert back to JAX (zero-copy)
    return torch_to_jax(out_trim)
