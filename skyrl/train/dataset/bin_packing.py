# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Sequence packing algorithms for SkyRL SFT (Megatron backend).

The structure is based on NemoRL's sequence packing implementation
"""

from __future__ import annotations

import enum
import heapq
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class PackingStrategy(enum.Enum):
    """Supported sequence packing algorithms."""

    FIRST_FIT_DECREASING = "first_fit_decreasing"
    BALANCED = "balanced"


class SeqPacker(ABC):
    """Abstract base class for bin-packing algorithms.

    Sub-classes implement :meth:`_pack_implementation` which returns a list
    of bins (each bin is a list of indices into the original
    ``sequence_lengths`` argument). The base class handles the post-pack
    DP-symmetry adjustment in :meth:`_adjust_bin_count`.
    """

    def __init__(
        self,
        bin_capacity: int,
        min_bin_count: Optional[int] = None,
        bin_count_multiple: Optional[int] = None,
    ):
        if min_bin_count is not None and min_bin_count < 0:
            raise ValueError("min_bin_count must be nonnegative")
        if bin_count_multiple is not None and bin_count_multiple < 1:
            raise ValueError("bin_count_multiple must be positive")

        self.bin_capacity = bin_capacity
        self.min_bin_count = min_bin_count
        self.bin_count_multiple = bin_count_multiple

    @abstractmethod
    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack sequences into bins. Override in sub-class."""

    def _validate_sequence_lengths(self, sequence_lengths: List[int]) -> None:
        for length in sequence_lengths:
            if length > self.bin_capacity:
                raise ValueError(f"Sequence length {length} exceeds bin capacity {self.bin_capacity}")

    def _adjust_bin_count(self, bins: List[List[int]]) -> List[List[int]]:
        """Pad the bin list to a multiple of ``bin_count_multiple``.

        New bins are filled by moving one sequence each from the largest
        existing bins, preserving capacity invariants. This is the
        mechanism that ensures each DP rank ends up with the same number of
        bins (and therefore the same ``num_microbatches``) every step.
        """
        current_bin_count = len(bins)
        target_bin_count = current_bin_count

        if self.min_bin_count is not None:
            target_bin_count = max(target_bin_count, self.min_bin_count)

        if self.bin_count_multiple is not None:
            remainder = target_bin_count % self.bin_count_multiple
            if remainder != 0:
                target_bin_count += self.bin_count_multiple - remainder

        if target_bin_count == current_bin_count:
            return bins

        total_sequences = sum(len(bin_contents) for bin_contents in bins)
        if total_sequences < target_bin_count:
            raise ValueError(
                f"Cannot create {target_bin_count} bins with only {total_sequences} sequences. "
                f"Each bin must contain at least one sequence. "
                f"Either reduce min_bin_count/bin_count_multiple or provide more sequences."
            )

        adjusted_bins = [bin_contents.copy() for bin_contents in bins]
        additional_bins_needed = target_bin_count - current_bin_count
        for _ in range(additional_bins_needed):
            adjusted_bins.append([])

        bin_sizes: List[Tuple[int, int]] = [
            (len(bin_contents), i) for i, bin_contents in enumerate(adjusted_bins[:current_bin_count])
        ]
        bin_sizes.sort(reverse=True)

        source_bin_idx = 0
        for new_bin_idx in range(current_bin_count, target_bin_count):
            while source_bin_idx < len(bin_sizes):
                _, original_bin_idx = bin_sizes[source_bin_idx]
                current_size = len(adjusted_bins[original_bin_idx])
                if current_size > 1:
                    sequence_to_move = adjusted_bins[original_bin_idx].pop()
                    adjusted_bins[new_bin_idx].append(sequence_to_move)
                    break
                else:
                    source_bin_idx += 1
            else:
                raise ValueError("Cannot create additional bins: insufficient sequences to redistribute.")

        return adjusted_bins

    def pack(self, sequence_lengths: List[int]) -> List[List[int]]:
        """Pack ``sequence_lengths`` into bins and apply DP-symmetry adjustment."""
        bins = self._pack_implementation(sequence_lengths)
        bins = self._adjust_bin_count(bins)
        return bins


class FirstFitDecreasing(SeqPacker):
    """First-Fit-Decreasing (FFD).

    Sort sequences by length descending, place each in the first bin with
    enough remaining capacity, open a new bin if none fits. Theoretical
    bound is 11/9 OPT + 6/9 (Johnson 1973). O(n log n) for the sort plus
    O(n * m) for placement.
    """

    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        self._validate_sequence_lengths(sequence_lengths)
        indexed = [(length, i) for i, length in enumerate(sequence_lengths)]
        indexed.sort(reverse=True)

        bins: List[List[int]] = []
        bin_remaining: List[int] = []

        for length, idx in indexed:
            placed = False
            for i, remaining in enumerate(bin_remaining):
                if remaining >= length:
                    bins[i].append(idx)
                    bin_remaining[i] -= length
                    placed = True
                    break
            if not placed:
                bins.append([idx])
                bin_remaining.append(self.bin_capacity - length)

        return bins


class Balanced(SeqPacker):
    """Balanced (least-loaded) bin-packing.

    Sort sequences by length descending, then place each sequence into the
    currently least-loaded bin whenever it fits, otherwise open a new bin.
    Unlike :class:`FirstFitDecreasing` (which minimizes the bin count by
    greedily filling the first bin that fits), this strategy spreads tokens
    evenly so the per-bin token totals stay roughly balanced. That balance
    keeps per-micro-batch compute even, which reduces straggling across
    micro-batches. O(n log n) for the sort plus O(n log m) for the
    heap-based placement.

    ``bin_capacity`` is a *soft* cap here: a sequence longer than the capacity
    is placed alone in its own over-capacity bin rather than raising (it can
    never fit alongside anything else). This differs from
    :class:`FirstFitDecreasing`, which rejects oversized sequences.
    """

    def _pack_implementation(self, sequence_lengths: List[int]) -> List[List[int]]:
        # No _validate_sequence_lengths() call: capacity is a soft cap (see
        # class docstring) so oversized sequences fall through to their own bin.
        # Sort by length only (stable), so equal-length sequences keep their
        # original relative order.
        indexed = [(length, i) for i, length in enumerate(sequence_lengths)]
        indexed.sort(key=lambda x: x[0], reverse=True)

        bins: List[List[int]] = []
        # Min-heap of (current_total_tokens, bin_idx); the root is always the
        # least-loaded bin.
        bin_tokens_heap: List[Tuple[int, int]] = []

        for length, idx in indexed:
            placed = False
            # Only the least-loaded bin needs checking: if the emptiest bin
            # cannot fit the sequence, no fuller bin can either.
            if bin_tokens_heap:
                bin_tokens, bin_idx = bin_tokens_heap[0]
                new_bin_tokens = bin_tokens + length
                if new_bin_tokens <= self.bin_capacity:
                    bins[bin_idx].append(idx)
                    heapq.heapreplace(bin_tokens_heap, (new_bin_tokens, bin_idx))
                    placed = True

            if not placed:
                bins.append([idx])
                heapq.heappush(bin_tokens_heap, (length, len(bins) - 1))

        return bins


_PACKERS = {
    PackingStrategy.FIRST_FIT_DECREASING: FirstFitDecreasing,
    PackingStrategy.BALANCED: Balanced,
}


def make_seq_packer(
    algorithm: PackingStrategy | str,
    bin_capacity: int,
    min_bin_count: Optional[int] = None,
    bin_count_multiple: Optional[int] = None,
) -> SeqPacker:
    """Factory returning a configured :class:`SeqPacker` instance.

    Args:
        algorithm: Either a :class:`PackingStrategy` enum value or a
            case-insensitive string matching an enum name (e.g.
            ``"first_fit_decreasing"``).
        bin_capacity: Maximum tokens per bin.
        min_bin_count: Force at least this many bins (typically
            ``dp_size``).
        bin_count_multiple: Force the total bin count to be a multiple of
            this value (typically ``dp_size``).
    """
    if isinstance(algorithm, str):
        try:
            algorithm = PackingStrategy[algorithm.upper()]
        except KeyError:
            available = ", ".join(a.name for a in PackingStrategy)
            raise ValueError(f"Unknown packing algorithm: {algorithm}. Available: {available}")

    if algorithm not in _PACKERS:
        available = ", ".join(a.name for a in PackingStrategy)
        raise ValueError(f"Unknown packing algorithm: {algorithm}. Available: {available}")

    return _PACKERS[algorithm](
        bin_capacity=bin_capacity,
        min_bin_count=min_bin_count,
        bin_count_multiple=bin_count_multiple,
    )
