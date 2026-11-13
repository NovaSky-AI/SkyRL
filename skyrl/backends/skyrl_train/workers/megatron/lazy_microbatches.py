"""Host microbatches built one at a time, as the schedule reaches them.

A worker that builds every microbatch of its shard before its first forward step holds a second
copy of the shard and leaves the GPU idle meanwhile. ``LazyMicrobatches`` builds microbatch ``i``
on its first read, so the first trains as soon as it is built, and drops it after its last.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.distributed.megatron.rank_microbatch import (
        HostMicrobatch,
    )


@dataclass(frozen=True)
class MicrobatchSummary:
    """What the model wrapper reads of a step's microbatches as a whole, without building them."""

    num_actions: int
    # Microbatches that carry no routed experts (R3).
    num_without_routes: int
    # calculate_per_token_loss's token count of the call, when it is on.
    num_tokens_global: Optional[int] = None


class LazyMicrobatches(Sequence):
    """``length`` host microbatches, microbatch ``i`` built by ``build(i)`` when first read.

    Each of ``readers`` consumers (one per virtual pipeline chunk) reads every microbatch once:
    a microbatch is cached between its first and last read, then dropped.
    """

    def __init__(
        self,
        build: Callable[[int], "HostMicrobatch"],
        length: int,
        summary: MicrobatchSummary,
        *,
        readers: int = 1,
    ) -> None:
        self._build = build
        self._length = length
        self._readers = readers
        self._cache: Dict[int, Tuple["HostMicrobatch", int]] = {}
        self.summary = summary

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> "HostMicrobatch":
        if not 0 <= index < self._length:
            raise IndexError(f"microbatch {index} of {self._length}")
        item, reads = self._cache.pop(index) if index in self._cache else (self._build(index), 0)
        if reads + 1 < self._readers:
            self._cache[index] = (item, reads + 1)
        return item


def summarize_microbatches(micro_batches: Sequence) -> MicrobatchSummary:
    """The summary of ``micro_batches``: a ``LazyMicrobatches``' own, or read off a list of them."""
    if isinstance(micro_batches, LazyMicrobatches):
        return micro_batches.summary
    first = micro_batches[0]
    return MicrobatchSummary(
        num_actions=first.num_actions,
        num_without_routes=sum(batch.rollout_expert_indices is None for batch in micro_batches),
        num_tokens_global=first.loss.num_tokens_global,
    )


def microbatch_seqlens(
    row_seqlens: List[List[int]], rows: List[Optional[List[int]]], padded_size: Optional[int]
) -> List[List[int]]:
    """Each microbatch's sub-sequence lengths, padding rows included, from its rows of the batch.

    ``row_seqlens[r]`` lists row ``r``'s sub-sequence lengths and ``rows`` each microbatch's rows
    (None for a padding microbatch). Microbatches pad to ``padded_size`` rows, when given; a padding
    row attends one token.
    """
    seqlens = []
    for microbatch_rows in rows:
        real = microbatch_rows or []
        padding = (padded_size if padded_size is not None else len(real)) - len(real)
        seqlens.append([n for row in real for n in row_seqlens[row]] + [1] * padding)
    return seqlens


def per_token_loss_counts(seqlens_per_microbatch: List[List[int]], align: int, cp_size: int) -> List[int]:
    """Each microbatch's packed token count on this CP rank, as calculate_per_token_loss divides by.

    ``seqlens_per_microbatch`` holds each microbatch's sub-sequence lengths (``microbatch_seqlens``).
    """
    return [sum(-(-int(n) // align) * align for n in seqlens) // cp_size for seqlens in seqlens_per_microbatch]
