"""Unit tests for the FFD bin-packing module.

Run with:
  uv run --extra dev -- pytest tests/backends/skyrl_train/distributed/test_bin_packing.py
"""

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.bin_packing import (
    FirstFitDecreasingPacker,
    PackingAlgorithm,
    get_packer,
)


class TestFirstFitDecreasingPacker:
    def test_deterministic(self):
        # Same input must produce the same bin assignment across runs.
        lengths = [820, 410, 520, 110, 250, 700, 50]
        p1 = FirstFitDecreasingPacker(bin_capacity=1024)
        p2 = FirstFitDecreasingPacker(bin_capacity=1024)
        assert p1.pack(lengths) == p2.pack(lengths)

    def test_no_overflow(self):
        # No bin's content may exceed the bin capacity.
        capacity = 100
        lengths = [60, 50, 40, 30, 20, 10, 80, 70, 90, 100, 5]
        packer = FirstFitDecreasingPacker(bin_capacity=capacity)
        bins = packer.pack(lengths)
        for bin_indices in bins:
            assert sum(lengths[i] for i in bin_indices) <= capacity

    def test_all_indices_present(self):
        # Every original index must appear in exactly one bin.
        lengths = [10, 20, 30, 40, 50, 60, 70]
        packer = FirstFitDecreasingPacker(bin_capacity=100)
        bins = packer.pack(lengths)
        flat = [i for b in bins for i in b]
        assert sorted(flat) == list(range(len(lengths)))

    def test_single_seq_per_bin_when_too_big(self):
        # If every sequence is half of capacity, FFD pairs them.
        # If every sequence is more than half, each must get its own bin.
        capacity = 100
        lengths = [60, 70, 80]
        packer = FirstFitDecreasingPacker(bin_capacity=capacity)
        bins = packer.pack(lengths)
        assert len(bins) == 3

    def test_overflow_raises(self):
        with pytest.raises(ValueError, match="exceeds bin capacity"):
            FirstFitDecreasingPacker(bin_capacity=100).pack([150])

    def test_min_bin_count(self):
        # min_bin_count forces extra empty (then redistributed) bins.
        capacity = 100
        lengths = [10, 10, 10]  # natural: 1 bin
        packer = FirstFitDecreasingPacker(bin_capacity=capacity, min_bin_count=3)
        bins = packer.pack(lengths)
        assert len(bins) == 3
        flat = [i for b in bins for i in b]
        assert sorted(flat) == [0, 1, 2]

    def test_bin_count_multiple(self):
        # bin_count_multiple rounds up to the next multiple. Need enough
        # sequences for empty-bin redistribution to succeed.
        capacity = 100
        lengths = [40, 30, 20, 10, 5]  # FFD packs into 1 bin (total 105 > 100, so 2)
        packer = FirstFitDecreasingPacker(bin_capacity=capacity, bin_count_multiple=4)
        bins = packer.pack(lengths)
        # 2 bins -> rounds up to 4
        assert len(bins) == 4
        flat = [i for b in bins for i in b]
        assert sorted(flat) == [0, 1, 2, 3, 4]

    def test_combined_min_and_multiple(self):
        # When both knobs apply, take the larger one and round up to the multiple.
        capacity = 100
        lengths = [10, 20, 30, 40]
        packer = FirstFitDecreasingPacker(bin_capacity=capacity, min_bin_count=3, bin_count_multiple=4)
        bins = packer.pack(lengths)
        # natural FFD on 4 seqs of total 100 -> 1 bin; min=3 -> 3; multiple=4 -> 4.
        assert len(bins) == 4

    def test_redistribute_preserves_capacity(self):
        # Empty-bin redistribution must not push any bin over capacity.
        capacity = 100
        lengths = [40, 30, 20, 10]  # FFD: 1 bin (total 100)
        packer = FirstFitDecreasingPacker(bin_capacity=capacity, min_bin_count=2)
        bins = packer.pack(lengths)
        for b in bins:
            assert sum(lengths[i] for i in b) <= capacity

    def test_redistribute_fails_when_too_few_seqs(self):
        # Cannot create more bins than sequences.
        with pytest.raises(ValueError, match="Cannot create"):
            FirstFitDecreasingPacker(bin_capacity=100, min_bin_count=5).pack([10, 20])


class TestGetPackerFactory:
    def test_enum_value(self):
        packer = get_packer(PackingAlgorithm.FIRST_FIT_DECREASING, bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasingPacker)

    def test_string_value(self):
        packer = get_packer("first_fit_decreasing", bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasingPacker)

    def test_string_case_insensitive(self):
        packer = get_packer("FIRST_FIT_DECREASING", bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasingPacker)

    def test_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            get_packer("nonexistent", bin_capacity=100)

    def test_factory_forwards_kwargs(self):
        packer = get_packer(
            "first_fit_decreasing",
            bin_capacity=100,
            min_bin_count=4,
            bin_count_multiple=2,
        )
        assert packer.bin_capacity == 100
        assert packer.min_bin_count == 4
        assert packer.bin_count_multiple == 2
