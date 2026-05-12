from __future__ import annotations
import torch
import torch.nn as nn


def Param(
    *shape: int,
    dtype: torch.dtype,
    kernel_init: callable,
    device: torch.device | str,
) -> nn.Parameter:
    """Create an initialized parameter tensor.

    Args:
        *shape: Shape of the parameter tensor
        dtype: Data type of the tensor
        kernel_init: Initialization function that modifies tensor in-place
        device: Device to place tensor on

    Returns:
        Initialized nn.Parameter
    """
    tensor = torch.empty(*shape, dtype=dtype, device=device)
    kernel_init(tensor)
    return nn.Parameter(tensor, requires_grad=True)


def prepare_routing(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_groups: int,
    adapter_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare inputs for ragged operations by sorting tokens by group.

    Args:
        tokens: Tensor of shape (num_tokens, ...) to be sorted by group
        indices: Tensor of shape (num_tokens,) indicating group assignment for each token
        num_groups: Total number of groups
        adapter_indices: Optional tensor of shape (num_tokens,) to be sorted together with tokens

    Returns:
        sorted_tokens: Tokens sorted by group index
        group_sizes: Number of tokens in each group
        unsort_indices: Indices to restore original order after ragged operations
        sorted_adapter_indices: Adapter indices sorted with tokens (or None if not provided)
    """
    # Sort by group index
    sort_idx = torch.argsort(indices)
    sorted_tokens = tokens[sort_idx]
    sorted_adapter_indices = None if adapter_indices is None else adapter_indices[sort_idx]

    # Compute group sizes (minlength guarantees output length)
    group_sizes = torch.bincount(indices, minlength=num_groups)

    # Inverse permutation to restore original order
    unsort_indices = torch.argsort(sort_idx)

    return sorted_tokens, group_sizes, unsort_indices, sorted_adapter_indices
