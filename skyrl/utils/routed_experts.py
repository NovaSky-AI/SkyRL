from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch

from skyrl.utils.token_metadata import TokenMetadataTrace

RoutedExpertIndices: TypeAlias = np.ndarray
ROUTED_EXPERT_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32)})


class RoutedExpertTrace:
    """Accumulate routed experts across incremental generation calls."""

    def __init__(self) -> None:
        self._metadata = TokenMetadataTrace()
        self._schema: tuple[int, int, np.dtype] | None = None

    @property
    def prompt_start(self) -> int:
        return self._metadata.num_rows

    def record_generation(
        self,
        *,
        prompt_token_count: int,
        generated_token_count: int,
        routed_experts: RoutedExpertIndices,
    ) -> None:
        if prompt_token_count < self.prompt_start:
            raise ValueError("routed-expert prompt start exceeds prompt length")
        if generated_token_count < 1:
            raise ValueError("routed-expert generation must produce at least one token")

        expected_rows = prompt_token_count - self.prompt_start + generated_token_count - 1
        compact = compact_routed_expert_indices(routed_experts)
        if self._schema is None:
            self._schema = (*compact.shape[1:], compact.dtype)
        self._metadata.append(compact, expected_rows=expected_rows)

    def finalize(self, *, token_count: int, loss_mask: Sequence[int]) -> RoutedExpertIndices:
        if len(loss_mask) != token_count:
            raise ValueError(f"loss mask has {len(loss_mask)} entries, expected {token_count}")
        if self.prompt_start > token_count:
            raise ValueError(f"routed-expert trace has {self.prompt_start} rows for {token_count} tokens")

        for source_index in range(self.prompt_start, token_count - 1):
            if loss_mask[source_index + 1] != 0:
                raise ValueError(f"missing routed-expert row for loss-active target at token {source_index + 1}")

        padding_count = token_count - self.prompt_start
        if padding_count:
            if self._schema is None:
                raise ValueError("cannot pad routed-expert trace before any routes are captured")
            num_layers, topk, dtype = self._schema
            padding_row = np.arange(topk, dtype=dtype)
            padding = np.broadcast_to(padding_row, (padding_count, num_layers, topk)).copy()
            self._metadata.append(padding, expected_rows=padding_count)

        return self._metadata.finalize(expected_rows=token_count)


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
