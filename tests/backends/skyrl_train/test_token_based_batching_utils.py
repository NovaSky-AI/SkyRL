"""
Unit tests for token-based micro-batching utilities (CPU only, no Ray/GPU needed).

Tests verify the behavior of balanced_binpacking and TokenBasedBatchIterator.

Run with:
uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/test_token_based_batching_utils.py
"""

from types import SimpleNamespace
from typing import List

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import (
    TokenBasedBatchIterator,
    get_microbatch_iterator,
)
from skyrl.train.dataset.bin_packing import make_seq_packer


def balanced_binpacking(token_counts: List[int], max_tokens_per_microbatch: int) -> List[List[int]]:
    """Pack via the shared Balanced SeqPacker (soft-cap semantics, as the iterator uses)."""
    return make_seq_packer("balanced", bin_capacity=max_tokens_per_microbatch).pack(token_counts)


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

    def test_oversized_sequence_gets_own_microbatch(self):
        """A single sequence longer than max_tokens is never split: it lands alone in its own
        microbatch that exceeds the (soft) cap, while the other sequences still pack normally."""
        token_counts = [100, 10, 10]
        max_tokens = 50
        result = balanced_binpacking(token_counts, max_tokens)

        # Every sequence is placed exactly once.
        assert sorted(idx for mb in result for idx in mb) == [0, 1, 2]
        # The oversized sequence (index 0) is alone in its own microbatch, exceeding the cap.
        oversized_mb = next(mb for mb in result if 0 in mb)
        assert oversized_mb == [0]
        assert sum(token_counts[i] for i in oversized_mb) > max_tokens
        # The remaining (fitting) sequences still respect the cap.
        for mb in result:
            if mb == oversized_mb:
                continue
            assert sum(token_counts[i] for i in mb) <= max_tokens

    def test_single_oversized_sequence(self):
        """A lone sequence longer than max_tokens still yields one microbatch (no error/split)."""
        result = balanced_binpacking([100], 50)
        assert result == [[0]]


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

    def test_reorder_and_combine_items_drops_padding(self):
        batch = self._make_batch([10, 3, 8, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=12)
        output_batches = [
            [{"sample": index} for index in indices] + [{"sample": "padding"}] for indices in iterator._microbatches
        ]
        iterator._num_padding_microbatches = 1
        output_batches.append([{"sample": "padding-microbatch"}])

        reordered = iterator.reorder_and_combine_items(output_batches)

        assert [output["sample"] for output in reordered] == list(range(batch.batch_size))

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

    def test_num_padding_microbatches_property(self):
        """num_padding_microbatches is exposed for metrics; without distributed init no
        padding microbatches are added, so len() equals the real microbatch count."""
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)
        assert iterator.num_padding_microbatches == 0
        assert len(iterator) == len(iterator._microbatches) + iterator.num_padding_microbatches

    def test_padding_microbatch_matches_seq_len(self):
        """Padding microbatches must share seq_len with real data (not a hardcoded short length),
        so Megatron sees a uniform seq_length and FSDP/Megatron can extract num_actions log-probs."""
        batch = self._make_batch([10, 10, 5, 5], num_actions=4)
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        padding = iterator._create_padding_microbatch()
        assert padding["sequences"].shape[1] == batch["sequences"].shape[1]
        assert padding["attention_mask"].shape[1] == batch["attention_mask"].shape[1]
        # Only a single token is marked valid (full seq_len for shape uniformity, but
        # cheap to compute in the packed path).
        assert padding["attention_mask"].sum().item() == padding["attention_mask"].shape[0]
        assert padding["attention_mask"][:, 0].sum().item() == padding["attention_mask"].shape[0]
        # Padding rows must not contribute to the loss.
        assert padding["loss_mask"].sum().item() == 0

    def test_padding_microbatch_uses_unique_dummy_routes(self):
        batch = self._make_batch([4, 4], num_actions=2)
        batch["rollout_expert_indices"] = torch.full((2, 4, 2, 3), 7, dtype=torch.int16)
        batch["router_padding_mask"] = torch.zeros((2, 4), dtype=torch.bool)
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=8)

        padding = iterator._create_padding_microbatch()

        expected = torch.tensor([0, 1, 2], dtype=torch.int16).expand_as(padding["rollout_expert_indices"])
        assert torch.equal(padding["rollout_expert_indices"], expected)
        assert torch.all(padding["router_padding_mask"])

    def test_worker_forward_backward_restores_input_order(self, monkeypatch):
        """Regression for the base (FSDP) Worker.forward_backward: with token-based
        batching, per-sample ``loss_fn_outputs`` must come back in input order with
        padding-microbatch entries dropped. Before the fix they were returned in
        packed microbatch order, attributing one sample's logprobs/elementwise loss
        to another (the megatron worker had the same bug, fixed in #2043)."""
        marker_base = 1000
        seq_lens = [8, 2, 6, 3, 5]
        batch = self._make_batch(seq_lens)
        # Tag each sample with a unique first-token marker so outputs are traceable.
        for i in range(len(seq_lens)):
            batch["sequences"][i, : seq_lens[i]] = marker_base + i

        max_tokens = 8
        # The test is only meaningful if packing actually permutes the samples.
        reference = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=max_tokens)
        packed_order = [i for mb in reference._microbatches for i in mb]
        assert packed_order != list(range(len(seq_lens))), "packing no longer permutes; pick new seq_lens"

        # Force one padding microbatch (normally added only under torch.distributed
        # to equalize microbatch counts across DP ranks).
        monkeypatch.setattr(
            TokenBasedBatchIterator,
            "_sync_num_microbatches",
            lambda self: len(self._microbatches) + 1,
        )
        # Metric all-reduce needs a process group; it is not under test here.
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.workers.worker.all_reduce_metrics",
            lambda metrics, strategy, group=None, sum_loss_metrics=False: metrics,
        )

        class _StubWorker(PolicyWorkerBase):
            def __init__(self, cfg):
                self.cfg = cfg
                self.strategy = None
                self.device_mesh = SimpleNamespace(get_group=lambda name: None)

            def _forward_backward_micro(self, experience, microbatch_weight, **kwargs):
                # One output per sample, identified by its first-token marker.
                markers = experience.sequences[:, 0].tolist()
                return {"loss": 1.0, "loss_fn_outputs": [{"logprobs": [float(m)]} for m in markers]}

        worker = _StubWorker(SimpleNamespace(micro_train_batch_size_per_gpu=2, max_tokens_per_microbatch=max_tokens))
        output = worker.forward_backward(batch, loss_fn="cross_entropy")

        got = [o["logprobs"][0] for o in output.loss_fn_outputs]
        assert got == [float(marker_base + i) for i in range(len(seq_lens))]

        # Sample-based batching (max_tokens_per_microbatch <= 0) already preserves
        # order; the flatten path must keep doing so.
        worker = _StubWorker(SimpleNamespace(micro_train_batch_size_per_gpu=2, max_tokens_per_microbatch=-1))
        output = worker.forward_backward(batch, loss_fn="cross_entropy")
        got = [o["logprobs"][0] for o in output.loss_fn_outputs]
        assert got == [float(marker_base + i) for i in range(len(seq_lens))]

    def test_worker_forward_backward_no_per_token_outputs(self, monkeypatch):
        """Callers that skip per-token outputs (metrics-only) still get an empty list."""
        batch = self._make_batch([8, 2, 6])
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.workers.worker.all_reduce_metrics",
            lambda metrics, strategy, group=None, sum_loss_metrics=False: metrics,
        )

        class _StubWorker(PolicyWorkerBase):
            def __init__(self, cfg):
                self.cfg = cfg
                self.strategy = None
                self.device_mesh = SimpleNamespace(get_group=lambda name: None)

            def _forward_backward_micro(self, experience, microbatch_weight, **kwargs):
                return {"loss": 1.0}

        worker = _StubWorker(SimpleNamespace(micro_train_batch_size_per_gpu=2, max_tokens_per_microbatch=8))
        output = worker.forward_backward(batch, loss_fn="cross_entropy")
        assert output.loss_fn_outputs == []

    def test_multimodal_tensorlist_microbatching(self):
        """Token-based microbatching must gather TensorList fields (multi-modal pixel_values /
        image_grid_thw) via the same index gather used for regular tensors."""
        seq_lens = [10, 10, 5, 5]
        batch = self._make_batch(seq_lens, num_actions=4)
        batch_size = len(seq_lens)
        # Variable per-sample shapes, like real vision inputs.
        batch["pixel_values"] = TensorList([torch.randn(3 + i, 8) for i in range(batch_size)])
        batch["image_grid_thw"] = TensorList([torch.tensor([[1, 2, 2]]) for _ in range(batch_size)])

        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        total_pv = 0
        for microbatch in iterator:
            if microbatch["loss_mask"].sum() == 0:
                continue  # skip padding microbatches (no multi-modal fields)
            pv = microbatch["pixel_values"]
            assert isinstance(pv, TensorList)
            assert len(pv) == microbatch["sequences"].shape[0]
            total_pv += len(pv)
        assert total_pv == batch_size  # every sample's pixel_values is accounted for
