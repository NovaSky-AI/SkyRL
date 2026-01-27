"""
Cutile CUDA kernel implementations for LoRA expert parallelism.

This module contains the actual CUDA kernels using NVIDIA's cuTile (cuda-tile on PyPI).
"""

from typing import Tuple

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required for cutile kernels")

try:
    from cuda import tile as ct
    from cuda.tile import Constant as ConstInt

    CUTILE_AVAILABLE = True
except ImportError:
    CUTILE_AVAILABLE = False

    # Define dummy types for syntax checking
    class DummyModule:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    ct = DummyModule()
    ConstInt = int


from .cutile_config import config as default_config


# ============================================================================
# Token Sorting Utilities
# ============================================================================


def lora_align_tile_size(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    tile_m: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert assignment and pad to tile boundaries.

    Adapted from moe_align_tile_size_torch in cutile-moe.py.

    Args:
        hidden_states: Input tokens [m, d]
        expert_ids: Expert assignment per token [m]
        tile_m: Tile size for M dimension (default from config)

    Returns:
        Tuple of:
            - sorted_hidden_states: Tokens sorted by expert [m_padded, d]
            - sorted_token_ids: Original indices of sorted tokens [m]
            - sorted_expert_ids: Expert ID per tile [num_tiles]
    """
    if tile_m is None:
        tile_m = default_config.tile_m

    m, d = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Sort tokens by expert assignment
    _, sorted_token_ids = torch.sort(expert_ids)
    sorted_hidden_states = hidden_states[sorted_token_ids]

    # Count tokens per expert
    num_experts = int(expert_ids.max().item()) + 1
    expert_counts = torch.bincount(expert_ids, minlength=num_experts)

    # Compute padding needed per expert to align to tile_m
    expert_counts_padded = torch.ceil(expert_counts.float() / tile_m).long() * tile_m
    total_padded = expert_counts_padded.sum().item()

    # Create padded tensor
    sorted_hidden_states_padded = torch.zeros(total_padded, d, dtype=dtype, device=device)
    sorted_hidden_states_padded[:m] = sorted_hidden_states

    # Create expert ID per tile
    sorted_expert_ids_per_tile = []
    for expert_id in range(num_experts):
        expert_tokens_padded = expert_counts_padded[expert_id].item()
        num_tiles = expert_tokens_padded // tile_m
        sorted_expert_ids_per_tile.extend([expert_id] * num_tiles)

    sorted_expert_ids_per_tile = torch.tensor(sorted_expert_ids_per_tile, dtype=torch.int32, device=device)

    return sorted_hidden_states_padded, sorted_token_ids, sorted_expert_ids_per_tile


# ============================================================================
# 2D Swizzling Utility
# ============================================================================


def swizzle_2d(
    M: int,
    N: int,
    TILE_M: int,
    TILE_N: int,
    GROUP_SIZE_M: int,
) -> Tuple[int, int]:
    """Compute 2D block swizzling for better cache locality.

    This function must be called from within a cutile kernel context.
    Matches the reference implementation from cutile-moe.py.

    Args:
        M: Total rows
        N: Total columns
        TILE_M: Tile size in M dimension
        TILE_N: Tile size in N dimension
        GROUP_SIZE_M: Number of M blocks to group together

    Returns:
        Tuple of (bid_m, bid_n) - block indices in M and N dimensions
    """
    bid = ct.bid(axis=0)
    num_bid_m = ct.cdiv(M, TILE_M)
    num_bid_n = ct.cdiv(N, TILE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    # Handle edge case when remaining blocks < GROUP_SIZE_M
    # Match reference implementation: use min() - cutile can handle Python min in this context
    remaining = num_bid_m - first_bid_m
    group_size_m = min(remaining, GROUP_SIZE_M)  # Python min works here
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


# ============================================================================
# Cutile Kernel
# ============================================================================

if CUTILE_AVAILABLE:

    @ct.kernel
    def cutile_lora_gemm_kernel(
        hidden_states: torch.Tensor,  # [M, K]
        # [E, K, N]  (if you can store it this way)
        weights: torch.Tensor,
        output: torch.Tensor,  # [M, N]
        expert_ids_per_tile: torch.Tensor,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        TILE_K: ConstInt,
    ):
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = output.shape[1]

        bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M=8)

        start_m = bid_m * TILE_M
        start_n = bid_n * TILE_N

        expert_id = ct.load(expert_ids_per_tile, index=bid_m, shape=())
        zero = ct.PaddingMode.ZERO

        acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

        # Hoist aranges if you still need them (not needed with ct.load offset+shape)
        for start_k in range(0, K, TILE_K):
            a = ct.load(
                hidden_states,
                (start_m, start_k),
                shape=(TILE_M, TILE_K),
                order=(1, 0),
                padding_mode=zero,
            )
            b = ct.load(
                weights,
                (expert_id, start_k, start_n),
                shape=(1, TILE_K, TILE_N),
                order=(0, 2, 1),
                padding_mode=zero,
            ).reshape((TILE_K, TILE_N))

            acc = ct.mma(a, b, acc)

        out_tile = ct.astype(acc, output.dtype)
        output_m_indices = start_m + ct.arange(TILE_M, dtype=ct.int32)
        output_n_indices = start_n + ct.arange(TILE_N, dtype=ct.int32)
        ct.scatter(
            output,
            (output_m_indices[:, None], output_n_indices[None, :]),
            out_tile,
        )


# ============================================================================
# Kernel Launch
# ============================================================================


def launch_cutile_lora_gemm(
    sorted_hidden_states: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    sorted_expert_ids_per_tile: torch.Tensor,
    TILE_M: int = None,
    TILE_N: int = None,
    TILE_K: int = None,
):
    """Launch cutile kernel for LoRA expert computation.

    Args:
        sorted_hidden_states: Sorted and padded tokens [m_padded, d]
        weights: Expert weights [num_experts, d, out_features]
        output: Output buffer [m_padded, out_features]
        sorted_expert_ids_per_tile: Expert ID per tile [num_tiles]
        TILE_M, TILE_N, TILE_K: Tile sizes (default from config)
    """
    if not CUTILE_AVAILABLE:
        raise RuntimeError("Cutile not available. Cannot run CUDA kernels.")

    # Use config defaults if not specified
    if TILE_M is None:
        TILE_M = default_config.tile_m
    if TILE_N is None:
        TILE_N = default_config.tile_n
    if TILE_K is None:
        TILE_K = default_config.tile_k

    m_padded, d = sorted_hidden_states.shape
    out_features = weights.shape[2]

    # Compute grid dimensions
    grid_m = ct.cdiv(m_padded, TILE_M)
    grid_n = ct.cdiv(out_features, TILE_N)
    grid = (grid_m * grid_n,)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cutile_lora_gemm_kernel,
        (
            sorted_hidden_states,
            weights,
            output,
            sorted_expert_ids_per_tile,
            TILE_M,
            TILE_N,
            TILE_K,
        ),
    )
