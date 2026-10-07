"""Unit tests for the bin-packing module.

Run with:
  uv run --extra dev -- pytest tests/backends/skyrl_train/distributed/test_bin_packing.py
"""

import pytest

from skyrl.train.dataset.bin_packing import (
    FirstFitDecreasing,
    ModifiedFirstFitDecreasing,
    PackingStrategy,
    make_seq_packer,
)


@pytest.mark.parametrize("packer_cls", [FirstFitDecreasing, ModifiedFirstFitDecreasing], ids=["ffd", "mffd"])
class TestFirstFitPackers:
    def test_deterministic(self, packer_cls):
        # Same input must produce the same bin assignment across runs.
        lengths = [820, 410, 520, 110, 250, 700, 50]
        p1 = packer_cls(bin_capacity=1024)
        p2 = packer_cls(bin_capacity=1024)
        assert p1.pack(lengths) == p2.pack(lengths)

    def test_no_overflow(self, packer_cls):
        # No bin's content may exceed the bin capacity.
        capacity = 100
        lengths = [60, 50, 40, 30, 20, 10, 80, 70, 90, 100, 5]
        packer = packer_cls(bin_capacity=capacity)
        bins = packer.pack(lengths)
        for bin_indices in bins:
            assert sum(lengths[i] for i in bin_indices) <= capacity

    def test_all_indices_present(self, packer_cls):
        # Every original index must appear in exactly one bin.
        lengths = [10, 20, 30, 40, 50, 60, 70]
        packer = packer_cls(bin_capacity=100)
        bins = packer.pack(lengths)
        flat = [i for b in bins for i in b]
        assert sorted(flat) == list(range(len(lengths)))

    def test_single_seq_per_bin_when_too_big(self, packer_cls):
        # If every sequence is half of capacity, the packer pairs them.
        # If every sequence is more than half, each must get its own bin.
        capacity = 100
        lengths = [60, 70, 80]
        packer = packer_cls(bin_capacity=capacity)
        bins = packer.pack(lengths)
        assert len(bins) == 3

    def test_overflow_raises(self, packer_cls):
        with pytest.raises(ValueError, match="exceeds bin capacity"):
            packer_cls(bin_capacity=100).pack([150])

    def test_min_bin_count(self, packer_cls):
        # min_bin_count forces extra empty (then redistributed) bins.
        capacity = 100
        lengths = [10, 10, 10]  # natural: 1 bin
        packer = packer_cls(bin_capacity=capacity, min_bin_count=3)
        bins = packer.pack(lengths)
        assert len(bins) == 3
        flat = [i for b in bins for i in b]
        assert sorted(flat) == [0, 1, 2]

    def test_bin_count_multiple(self, packer_cls):
        # bin_count_multiple rounds up to the next multiple. Need enough
        # sequences for empty-bin redistribution to succeed.
        capacity = 100
        lengths = [40, 30, 20, 10, 5]  # Natural packing uses 2 bins.
        packer = packer_cls(bin_capacity=capacity, bin_count_multiple=4)
        bins = packer.pack(lengths)
        # 2 bins -> rounds up to 4
        assert len(bins) == 4
        flat = [i for b in bins for i in b]
        assert sorted(flat) == [0, 1, 2, 3, 4]

    def test_combined_min_and_multiple(self, packer_cls):
        # When both knobs apply, take the larger one and round up to the multiple.
        capacity = 100
        lengths = [10, 20, 30, 40]
        packer = packer_cls(bin_capacity=capacity, min_bin_count=3, bin_count_multiple=4)
        bins = packer.pack(lengths)
        # natural packing on 4 seqs of total 100 -> 1 bin; min=3 -> 3; multiple=4 -> 4.
        assert len(bins) == 4

    def test_redistribute_preserves_capacity(self, packer_cls):
        # Empty-bin redistribution must not push any bin over capacity.
        capacity = 100
        lengths = [40, 30, 20, 10]  # Natural packing: 1 bin (total 100)
        packer = packer_cls(bin_capacity=capacity, min_bin_count=2)
        bins = packer.pack(lengths)
        for b in bins:
            assert sum(lengths[i] for i in b) <= capacity

    def test_redistribute_fails_when_too_few_seqs(self, packer_cls):
        # Cannot create more bins than sequences.
        with pytest.raises(ValueError, match="Cannot create"):
            packer_cls(bin_capacity=100, min_bin_count=5).pack([10, 20])


class TestModifiedFirstFitDecreasing:
    def test_matches_mffd_phases(self):
        packer = ModifiedFirstFitDecreasing(bin_capacity=100)

        assert packer.pack([60, 55, 45, 40, 30, 25, 20, 10, 5]) == [
            [0, 3],
            [1, 2],
            [4, 5, 6, 7, 8],
        ]

    def test_deterministic_and_preserves_all_indices(self):
        lengths = [55, 48, 34, 31, 29, 22, 16, 9, 7, 4]
        packer = ModifiedFirstFitDecreasing(bin_capacity=100)

        first = packer.pack(lengths)
        second = packer.pack(lengths)

        assert first == second
        assert sorted(index for bin_indices in first for index in bin_indices) == list(range(len(lengths)))
        assert all(sum(lengths[index] for index in bin_indices) <= 100 for bin_indices in first)

    def test_leftovers_use_first_fit_not_least_loaded(self):
        packer = ModifiedFirstFitDecreasing(bin_capacity=100)

        bins = packer.pack([14, 91, 25, 37, 33, 22, 34, 99, 30])

        assert bins == [[7], [1], [3, 6, 2], [4, 8, 5, 0]]

    def test_rejects_nonpositive_lengths(self):
        with pytest.raises(ValueError, match="sequence lengths must be positive"):
            ModifiedFirstFitDecreasing(bin_capacity=100).pack([0])


class TestMakeSeqPackerFactory:
    def test_enum_value(self):
        packer = make_seq_packer(PackingStrategy.FIRST_FIT_DECREASING, bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasing)

    def test_string_value(self):
        packer = make_seq_packer("first_fit_decreasing", bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasing)

    def test_string_case_insensitive(self):
        packer = make_seq_packer("FIRST_FIT_DECREASING", bin_capacity=100)
        assert isinstance(packer, FirstFitDecreasing)

    def test_modified_first_fit_decreasing(self):
        packer = make_seq_packer(PackingStrategy.MODIFIED_FIRST_FIT_DECREASING, bin_capacity=100)

        assert isinstance(packer, ModifiedFirstFitDecreasing)

    def test_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unknown packing algorithm"):
            make_seq_packer("nonexistent", bin_capacity=100)

    def test_factory_forwards_kwargs(self):
        packer = make_seq_packer(
            "first_fit_decreasing",
            bin_capacity=100,
            min_bin_count=4,
            bin_count_multiple=2,
        )
        assert packer.bin_capacity == 100
        assert packer.min_bin_count == 4
        assert packer.bin_count_multiple == 2
