"""Sequence packing algorithms for SkyRL SFT (Megatron backend).

Only First-Fit-Decreasing (FFD) is shipped in v1; the abstract
:class:`SequencePacker` base supports adding MFFD / FFShuffle / Concatenative
later without touching call sites. The DP-symmetry knobs
(``min_bin_count`` + ``bin_count_multiple``) and the empty-bin
redistribution algorithm in :meth:`SequencePacker._adjust_bin_count` are the
load-bearing pieces — they let the controller emit a bin count that is
exactly a multiple of ``dp_size`` so every DP rank gets the same
``num_microbatches``.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class PackingAlgorithm(enum.Enum):
    """Supported sequence packing algorithms.

    Only ``FIRST_FIT_DECREASING`` is implemented in v1.
    """

    FIRST_FIT_DECREASING = "first_fit_decreasing"


class SequencePacker(ABC):
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


class FirstFitDecreasingPacker(SequencePacker):
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


_PACKERS = {
    PackingAlgorithm.FIRST_FIT_DECREASING: FirstFitDecreasingPacker,
}


def get_packer(
    algorithm: PackingAlgorithm | str,
    bin_capacity: int,
    min_bin_count: Optional[int] = None,
    bin_count_multiple: Optional[int] = None,
) -> SequencePacker:
    """Factory returning a configured :class:`SequencePacker` instance.

    Args:
        algorithm: Either a :class:`PackingAlgorithm` enum value or a
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
            algorithm = PackingAlgorithm[algorithm.upper()]
        except KeyError:
            available = ", ".join(a.name for a in PackingAlgorithm)
            raise ValueError(f"Unknown packing algorithm: {algorithm}. Available: {available}")

    if algorithm not in _PACKERS:
        available = ", ".join(a.name for a in PackingAlgorithm)
        raise ValueError(f"Unknown packing algorithm: {algorithm}. Available: {available}")

    return _PACKERS[algorithm](
        bin_capacity=bin_capacity,
        min_bin_count=min_bin_count,
        bin_count_multiple=bin_count_multiple,
    )
