"""
JAX-PyTorch interop for cutile LoRA kernels.

This module provides the main API for using cutile kernels from JAX code,
handling DLPack conversions and providing a drop-in replacement for ragged_dot.
"""

try:
    import jax
except ImportError:
    raise ImportError("JAX is required for cutile LoRA")

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for cutile LoRA")

from .cutile_lora_kernels import (
    lora_align_tile_size,
    launch_cutile_lora_gemm,
    CUTILE_AVAILABLE,
)
from .cutile_config import config as default_config


# ============================================================================
# DLPack Conversion Utilities
# ============================================================================


def jax_to_torch(jax_arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor (zero-copy via DLPack).

    Args:
        jax_arr: JAX array (should be on GPU)

    Returns:
        PyTorch tensor sharing the same GPU memory

    Raises:
        ValueError: If array is not on GPU
    """
    # Ensure on GPU
    device_str = str(jax_arr.device()).lower()
    if "gpu" not in device_str and "cuda" not in device_str:
        raise ValueError(f"Expected GPU array, got device: {jax_arr.device()}")

    # Convert via DLPack (zero-copy when devices match)
    try:
        dlpack_capsule = jax.dlpack.to_dlpack(jax_arr)
        torch_tensor = torch.from_dlpack(dlpack_capsule)
        return torch_tensor
    except Exception as e:
        raise RuntimeError(f"DLPack conversion failed: {e}") from e


def torch_to_jax(torch_tensor: torch.Tensor) -> jax.Array:
    """Convert PyTorch tensor back to JAX array (zero-copy via DLPack).

    Args:
        torch_tensor: PyTorch tensor (should be on CUDA)

    Returns:
        JAX array sharing the same GPU memory

    Raises:
        ValueError: If tensor is not on CUDA
    """
    # Ensure on CUDA
    if not torch_tensor.is_cuda:
        raise ValueError(f"Expected CUDA tensor, got device: {torch_tensor.device}")

    # Convert via DLPack (zero-copy)
    try:
        jax_arr = jax.dlpack.from_dlpack(torch_tensor)
        return jax_arr
    except Exception as e:
        raise RuntimeError(f"DLPack conversion failed: {e}") from e


# ============================================================================
# Main API
# ============================================================================


def cutile_ragged_dot(
    lhs: jax.Array,  # [m, d]
    rhs: jax.Array,  # [num_groups, d, out_features]
    group_sizes: jax.Array,  # [num_groups]
    precision=None,  # Ignored (cutile uses GPU precision)
    preferred_element_type=None,  # Ignored
    group_offset: jax.Array | None = None,  # Phase 1: Not implemented yet
) -> jax.Array:
    """Drop-in replacement for ragged_dot using cutile kernels.

    Phase 1 Implementation: Single-GPU forward pass only (no group_offset).

    Args:
        lhs: Input tokens [m, d]
        rhs: Expert weights [num_groups, d, out_features]
        group_sizes: Number of tokens per group [num_groups]
        precision: Ignored (cutile uses native GPU precision)
        preferred_element_type: Ignored
        group_offset: NOT IMPLEMENTED YET (Phase 2)

    Returns:
        Output [m, out_features] with expert-specific computation

    Raises:
        RuntimeError: If cutile not available or CUDA not available
        NotImplementedError: If group_offset is provided (Phase 2)
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

    # Phase 1: group_offset not supported yet
    if group_offset is not None:
        raise NotImplementedError(
            "group_offset not supported in Phase 1 cutile implementation. "
            "Set TX_USE_CUTILE_LORA=0 to use ragged_dot for multi-GPU EP."
        )

    # 1. Convert JAX arrays to PyTorch tensors
    try:
        lhs_torch = jax_to_torch(lhs)
        rhs_torch = jax_to_torch(rhs)
        group_sizes_torch = jax_to_torch(group_sizes)
    except Exception as e:
        raise RuntimeError(f"Failed to convert inputs to PyTorch: {e}") from e

    # 2. Derive expert_ids from group_sizes
    # group_sizes = [10, 20, 30] means:
    #   tokens 0-9: expert 0
    #   tokens 10-29: expert 1
    #   tokens 30-59: expert 2
    expert_ids = torch.repeat_interleave(
        torch.arange(len(group_sizes_torch), device=lhs_torch.device),
        group_sizes_torch,
    ).to(torch.int32)

    # 3. Sort tokens by expert and pad to tile boundaries
    sorted_lhs, sorted_token_ids, sorted_expert_ids_per_tile = lora_align_tile_size(
        lhs_torch, expert_ids, tile_m=default_config.tile_m
    )

    # 4. Allocate output buffer
    m_padded = sorted_lhs.shape[0]
    out_features = rhs_torch.shape[2]
    output_torch = torch.zeros(
        m_padded,
        out_features,
        dtype=lhs_torch.dtype,
        device=lhs_torch.device,
    )

    # 5. Launch cutile kernel
    try:
        launch_cutile_lora_gemm(
            sorted_lhs,
            rhs_torch,
            output_torch,
            sorted_expert_ids_per_tile,
            TILE_M=default_config.tile_m,
            TILE_N=default_config.tile_n,
            TILE_K=default_config.tile_k,
        )
    except Exception as e:
        raise RuntimeError(f"Cutile kernel launch failed: {e}") from e

    # 6. Unsort tokens to original order
    m_original = lhs_torch.shape[0]
    output_sorted = output_torch[:m_original]
    output_unsorted = torch.empty_like(output_sorted)
    output_unsorted[sorted_token_ids] = output_sorted

    # 7. Convert back to JAX
    try:
        output_jax = torch_to_jax(output_unsorted)
    except Exception as e:
        raise RuntimeError(f"Failed to convert output to JAX: {e}") from e

    return output_jax
