"""
Unit tests for token-based micro-batching utilities (CPU only, no Ray/GPU needed).

Tests verify the behavior of balanced_binpacking and TokenBasedBatchIterator.

Run with:
uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/test_token_based_batching_utils.py
"""

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import (
    TokenBasedBatchIterator,
    balanced_binpacking,
    get_microbatch_iterator,
)


class TestBalancedBinpacking:
    def test_basic_packing(self):
        result = balanced_binpacking([10, 10, 5, 5], 15)
        assert len(result) == 2
        # Each microbatch should have total <= 15
        for mb in result:
            total = sum([10, 10, 5, 5][i] for i in mb)
            assert total <= 15

    def test_single_large_item(self):
        result = balanced_binpacking([10, 1, 1, 1, 1, 1], 10)
        assert len(result) == 2
        # The large item should be alone
        for mb in result:
            total = sum([10, 1, 1, 1, 1, 1][i] for i in mb)
            assert total <= 10

    def test_all_items_equal(self):
        result = balanced_binpacking([5, 5, 5, 5], 10)
        assert len(result) == 2
        for mb in result:
            total = sum(5 for _ in mb)
            assert total <= 10

    def test_single_item(self):
        result = balanced_binpacking([10], 15)
        assert len(result) == 1
        assert result[0] == [0]

    def test_all_indices_covered(self):
        token_counts = [8, 3, 5, 6, 2, 7]
        result = balanced_binpacking(token_counts, 11)
        all_indices = sorted(idx for mb in result for idx in mb)
        assert all_indices == list(range(len(token_counts)))

    def test_no_overflow(self):
        token_counts = [8, 3, 5, 6, 2, 7]
        max_tokens = 11
        result = balanced_binpacking(token_counts, max_tokens)
        for mb in result:
            total = sum(token_counts[i] for i in mb)
            assert total <= max_tokens


class TestTokenBasedBatchIterator:
    def _make_batch(self, seq_lens, num_actions=4):
        """Create a dummy TrainingInputBatch with variable sequence lengths."""
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens)

        sequences = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
        for i, seq_len in enumerate(seq_lens):
            sequences[i, :seq_len] = torch.randint(0, 100, (seq_len,), dtype=int, device="cpu")
            attention_mask[i, :seq_len] = 1

        data = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
                "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
                "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
                "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
                "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
                "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
                "response_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            }
        )
        data.metadata = {"response_length": num_actions}
        return data

    def test_iterator_yields_all_samples(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        all_indices = []
        for mb_indices in iterator._microbatches:
            all_indices.extend(mb_indices)
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_iterator_respects_token_limit(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        for microbatch in iterator:
            token_count = microbatch["attention_mask"].sum().item()
            # Allow some slack for padding microbatches
            if microbatch["loss_mask"].sum() > 0:  # not a padding batch
                assert token_count <= 15

    def test_len_matches_iteration(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)
        count = sum(1 for _ in iterator)
        assert count == len(iterator)

    def test_reorder_and_combine(self):
        """Verify that reorder_and_combine_batches restores original order."""
        batch = self._make_batch([10, 3, 8, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=12)

        # Simulate forward outputs (just use the microbatch itself as output)
        outputs = []
        for microbatch in iterator:
            outputs.append(microbatch)

        reordered = iterator.reorder_and_combine_batches(outputs)
        # Check that the sequences match the original order
        for i in range(batch.batch_size):
            assert torch.equal(reordered["sequences"][i], batch["sequences"][i])

    def test_get_microbatch_iterator_factory(self):
        batch = self._make_batch([10, 10, 5, 5])

        # Token-based
        it = get_microbatch_iterator(batch, micro_batch_size=2, max_tokens_per_microbatch=15)
        assert isinstance(it, TokenBasedBatchIterator)

        # Sample-based (disabled)
        from skyrl.backends.skyrl_train.workers.worker_utils import (
            SampleBasedBatchIterator,
        )

        it = get_microbatch_iterator(batch, micro_batch_size=2, max_tokens_per_microbatch=-1)
        assert isinstance(it, SampleBasedBatchIterator)
