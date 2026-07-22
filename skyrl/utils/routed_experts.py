from typing import TypeAlias

import numpy as np
import torch

RoutedExpertIndices: TypeAlias = np.ndarray
ROUTED_EXPERT_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32)})


def compact_routed_expert_indices(routed_experts: RoutedExpertIndices) -> RoutedExpertIndices:
    """Validate and compact a routed-expert array to the canonical integer dtype."""
    if not isinstance(routed_experts, np.ndarray):
        raise TypeError("routed expert indices must be a NumPy array")
    if routed_experts.ndim != 3 or not np.issubdtype(routed_experts.dtype, np.integer):
        raise ValueError(
            "routed expert indices must be an integer [tokens, layers, topk] array, "
            f"got shape {routed_experts.shape} and dtype {routed_experts.dtype}"
        )
    if int(routed_experts.min(initial=0)) < 0:
        raise ValueError("routed expert indices must be non-negative")

    max_expert_id = int(routed_experts.max(initial=0))
    if max_expert_id < 2**8:
        dtype = np.dtype(np.uint8)
    elif max_expert_id < 2**15:
        dtype = np.dtype(np.int16)
    elif max_expert_id < 2**31:
        dtype = np.dtype(np.int32)
    else:
        raise ValueError(f"routed expert index exceeds signed int32: {max_expert_id}")

    compact = np.asarray(routed_experts, dtype=dtype, order="C")
    if not compact.flags.writeable:
        compact = compact.copy(order="C")
    return compact


def make_replay_padding_indices(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return dummy routes with ``topk`` distinct experts in every row."""
    if not shape or shape[-1] < 1:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {shape}")
    padding_row = torch.arange(shape[-1], dtype=dtype, device=device)
    return padding_row.expand(shape).clone()
