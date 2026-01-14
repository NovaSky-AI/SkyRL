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
    grid_m = ct.cdiv(M, TILE_M)
    grid_n = ct.cdiv(N, TILE_N)

    # 2D swizzle pattern
    num_block_n = ct.cdiv(grid_m, GROUP_SIZE_M) * grid_n
    bid_m = (bid % num_block_n) // grid_n * GROUP_SIZE_M + (bid // num_block_n)
    bid_n = (bid % num_block_n) % grid_n

    return bid_m, bid_n


# ============================================================================
# Cutile Kernel
# ============================================================================

if CUTILE_AVAILABLE:

    @ct.kernel
    def cutile_lora_gemm_kernel(
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
        expert_ids_per_tile: torch.Tensor,
        TILE_M: ConstInt,
        TILE_N: ConstInt,
        TILE_K: ConstInt,
    ):
        """Cutile kernel for LoRA expert-specific matrix multiplication.

        Computes: output[i] = hidden_states[i] @ weights[expert_id[i // TILE_M]]

        Args:
            hidden_states: Sorted and padded tokens [m_padded, d]
            weights: Expert weights [num_experts, d, out_features]
            output: Output buffer [m_padded, out_features]
            expert_ids_per_tile: Expert ID for each tile [num_tiles]
            TILE_M, TILE_N, TILE_K: Tile sizes (compile-time constants)
        """
        # Get 2D block coordinates with swizzling
        m_total = hidden_states.shape[0]
        d = hidden_states.shape[1]
        out_features = weights.shape[2]

        bid_m, bid_n = swizzle_2d(m_total, out_features, TILE_M, TILE_N, GROUP_SIZE_M=8)

        # Compute global indices for this tile
        start_m = bid_m * TILE_M
        start_n = bid_n * TILE_N

        # Get expert ID for this tile
        expert_id = ct.gather(expert_ids_per_tile, ct.full((1,), bid_m, dtype=ct.int32))[0]

        # Initialize accumulator
        accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

        # Perform tiled matrix multiplication
        for k_tile in range(ct.cdiv(d, TILE_K)):
            start_k = k_tile * TILE_K

            # Load input tile [TILE_M, TILE_K]
            m_indices = start_m + ct.arange(TILE_M)
            k_indices = start_k + ct.arange(TILE_K)
            input_tile = ct.load(
                hidden_states,
                (m_indices[:, None], k_indices[None, :]),
                shape=(TILE_M, TILE_K),
                padding_mode=ct.PaddingMode.ZERO,
            )

            # Load weight tile [TILE_K, TILE_N] for this expert
            n_indices = start_n + ct.arange(TILE_N)
            expert_indices = ct.full((TILE_K, TILE_N), expert_id, dtype=ct.int32)
            k_indices_2d = k_indices[:, None].broadcast_to((TILE_K, TILE_N))
            n_indices_2d = n_indices[None, :].broadcast_to((TILE_K, TILE_N))

            weight_tile = ct.gather(
                weights,
                (expert_indices, k_indices_2d, n_indices_2d),
                shape=(TILE_K, TILE_N),
            )

            # Matrix multiply-accumulate
            accumulator = ct.mma(input_tile, weight_tile, accumulator)

        # Cast accumulator to output dtype
        accumulator = ct.astype(accumulator, output.dtype)

        # Store output tile
        output_m_indices = start_m + ct.arange(TILE_M)
        output_n_indices = start_n + ct.arange(TILE_N)
        ct.scatter(
            output,
            (output_m_indices[:, None], output_n_indices[None, :]),
            accumulator,
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
