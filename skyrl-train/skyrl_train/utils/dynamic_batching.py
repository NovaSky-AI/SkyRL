"""
Dynamic token-based batching utilities using Karmarkar-Karp algorithm.

This module provides the core algorithms for partitioning sequences based on token count,
adapted from the reference implementation to work with skyrl-train's architecture.
"""

import copy
import heapq
from typing import List, Optional, Tuple
import math


def ceildiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def get_reverse_idx(idx_map: List[int]) -> List[int]:
    """Build the inverse of an index mapping."""
    reverse_idx_map = [0] * len(idx_map)
    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i
    return reverse_idx_map


class Set:
    """Helper class for Karmarkar-Karp algorithm."""

    def __init__(self) -> None:
        self.sum = 0
        self.items = []

    def add(self, idx: int, val: int):
        self.items.append((idx, val))
        self.sum += val

    def merge(self, other):
        for idx, val in other.items:
            self.items.append((idx, val))
            self.sum += val

    def __lt__(self, other):
        if self.sum != other.sum:
            return self.sum < other.sum
        if len(self.items) != len(other.items):
            return len(self.items) < len(other.items)
        return self.items < other.items


class State:
    """State for Karmarkar-Karp algorithm."""

    def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
        self.k = k
        self.sets = [Set() for _ in range(k)]
        assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
        for i, (idx, seqlen) in enumerate(items):
            self.sets[i].add(idx=idx, val=seqlen)
        self.sets = sorted(self.sets, reverse=True)

    def get_partitions(self):
        partitions = []
        for i in range(len(self.sets)):
            cur_partition = []
            for idx, _ in self.sets[i].items:
                cur_partition.append(idx)
            partitions.append(cur_partition)
        return partitions

    def merge(self, other):
        for i in range(self.k):
            self.sets[i].merge(other.sets[self.k - 1 - i])
        self.sets = sorted(self.sets, reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].sum - self.sets[-1].sum

    def __lt__(self, other):
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]


def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool = False) -> List[List[int]]:
    """
    Karmarkar-Karp algorithm for balanced partitioning.
    See: https://en.wikipedia.org/wiki/Largest_differencing_method

    Args:
        seqlen_list: List of sequence lengths
        k_partitions: Number of partitions to create
        equal_size: If True, ensure each partition has the same number of items

    Returns:
        List of partitions, where each partition is a list of indices
    """
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []

    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()

    if equal_size:
        for partition in partitions:
            assert len(partition) * k_partitions == len(seqlen_list)

    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: List[int], k_partitions: int, equal_size: bool = False
) -> List[List[int]]:
    """
    Calculate balanced partitions using Karmarkar-Karp algorithm.

    Args:
        seqlen_list: List of sequence lengths for each item
        k_partitions: Desired number of partitions
        equal_size: If True, ensure equal number of items per partition

    Returns:
        List of k_partitions lists, each containing indices
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)

    # Verify and sort partitions
    assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
    seen_idx = set()
    sorted_partitions = [None] * k_partitions

    for i, partition in enumerate(partitions):
        assert len(partition) > 0, f"the {i}-th partition is empty"
        for idx in partition:
            seen_idx.add(idx)
        sorted_partitions[i] = sorted(partition)

    assert seen_idx == set(range(len(seqlen_list)))
    return sorted_partitions


def calculate_num_micro_batches(
    token_counts: List[int], max_token_len: int, min_num_micro_batch: Optional[int] = None
) -> int:
    """
    Calculate the number of micro-batches needed.

    Args:
        token_counts: List of token counts for each sequence
        max_token_len: Maximum tokens per micro-batch
        min_num_micro_batch: Minimum number of micro-batches

    Returns:
        Number of micro-batches needed
    """
    total_tokens = sum(token_counts)
    num_sequences = len(token_counts)

    num_micro_batches = min(num_sequences, ceildiv(total_tokens, max_token_len))

    if min_num_micro_batch is not None:
        num_micro_batches = max(min_num_micro_batch, num_micro_batches)

    return num_micro_batches

def create_fixed_partitions(batch_size: int, num_partitions: int) -> List[List[int]]:
    """Create fixed-size partitions of indices."""
    indices = list(range(batch_size))
    partition_size = math.ceil(batch_size / num_partitions)

    partitions = []
    for i in range(0, batch_size, partition_size):
        partition = indices[i : i + partition_size]
        if partition:
            partitions.append(partition)

    return partitions