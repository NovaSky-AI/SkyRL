from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataTrace,
)

RoutedExpertIndices: TypeAlias = np.ndarray
ROUTED_EXPERT_DTYPES = frozenset({np.dtype(np.uint8), np.dtype(np.int16), np.dtype(np.int32)})


class RoutedExpertTrace:
    """Accumulate routed experts across incremental generation calls."""

    def __init__(self) -> None:
        self._metadata = TokenMetadataTrace()

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
        self._metadata.append(compact_routed_expert_indices(routed_experts), expected_rows=expected_rows)

    def finalize(self, *, token_count: int, loss_mask: Sequence[int]) -> RoutedExpertIndices:
        """Return the captured routes, which cover a prefix of the sequence's real tokens.

        The capture ends short of ``token_count``: the last sampled token has no subsequent
        decode forward to record its route, and a synthetic EOS is never evaluated at all.
        The row count is the trace's only report of where the capture stops -- collation
        dummy-fills the uncovered tail and ``make_router_padding_mask`` excludes exactly those
        rows from router accounting -- so the trace must not pad the tail itself, which would
        report fabricated routes as captured ones.
        """
        if len(loss_mask) != token_count:
            raise ValueError(f"loss mask has {len(loss_mask)} entries, expected {token_count}")
        if self.prompt_start > token_count:
            raise ValueError(f"routed-expert trace has {self.prompt_start} rows for {token_count} tokens")

        if any(loss_mask[self.prompt_start + 1 : token_count]):
            for source_index in range(self.prompt_start, token_count - 1):
                if loss_mask[source_index + 1] != 0:
                    raise ValueError(f"missing routed-expert row for loss-active target at token {source_index + 1}")

        return self._metadata.finalize(expected_rows=self.prompt_start)


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
