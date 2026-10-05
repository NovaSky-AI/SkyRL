"""
Unit tests for token-based micro-batching utilities (CPU only, no Ray/GPU needed).

Tests verify the behavior of balanced_binpacking and TokenBasedBatchIterator.

Run with:
uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/test_token_based_batching_utils.py
"""

from types import SimpleNamespace
from typing import List

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import (
    TokenBasedBatchIterator,
    get_microbatch_iterator,
)
from skyrl.train.dataset.bin_packing import make_seq_packer


def balanced_binpacking(token_counts: List[int], max_tokens_per_microbatch: int) -> List[List[int]]:
    """Pack via the shared Balanced SeqPacker (soft-cap semantics, as the iterator uses)."""
    return make_seq_packer("balanced", bin_capacity=max_tokens_per_microbatch).pack(token_counts)


# Lengths that packing permutes at LOSS_FORWARD_MAX_TOKENS; sample 0 exceeds the budget on its own.
LOSS_FORWARD_SEQ_LENS = [12, 2, 6, 3, 5]
LOSS_FORWARD_MAX_TOKENS = 8
MARKER_BASE = 1000


def _import_megatron_worker():
    try:
        from skyrl.backends.skyrl_train.workers.megatron import megatron_worker
    except ModuleNotFoundError as e:
        if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
            raise
        pytest.skip(f"megatron unavailable: {e}")
    return megatron_worker


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

    def _add_packed_side_channels(self, batch: TrainingInputBatch) -> None:
        """Attach both packed side channels: routes over real tokens, support over responses."""
        batch["rollout_expert_indices"] = PackedTensor(
            torch.full((8, 2, 3), 7, dtype=torch.int16),
            cu_seqlens_from_lengths([4, 4]),
        )
        batch["router_padding_mask"] = torch.zeros((2, 4), dtype=torch.bool)
        batch[SAMPLE_SUPPORT_FIELD] = PackedTensor(
            torch.full((4, 5), 11, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
            cu_seqlens_from_lengths([2, 2]),
        )

    def test_padding_microbatch_uses_unique_dummy_routes(self):
        batch = self._make_batch([4, 4], num_actions=2)
        self._add_packed_side_channels(batch)
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=8)

        padding = iterator._create_padding_microbatch()

        padded_routes = padding["rollout_expert_indices"]
        assert padded_routes.sequence_lengths.tolist() == [1]
        expected = torch.tensor([0, 1, 2], dtype=torch.int16).expand_as(padded_routes.values)
        assert torch.equal(padded_routes.values, expected)
        assert torch.all(padding["router_padding_mask"])

    def test_padding_microbatch_sample_support_holds_no_response_rows(self):
        """A dummy row attends one token but generates no response."""
        batch = self._make_batch([4, 4], num_actions=2)
        self._add_packed_side_channels(batch)
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=8)

        padding = iterator._create_padding_microbatch()

        padded_support = padding[SAMPLE_SUPPORT_FIELD]
        assert len(padded_support) == 1
        assert padded_support.sequence_lengths.tolist() == [0]
        assert padded_support.values.shape == (0, 5)
        assert padded_support.dtype == SAMPLE_SUPPORT_TORCH_DTYPE
        assert padding["rollout_expert_indices"].sequence_lengths.tolist() == [1]

    def test_microbatch_selection_gathers_packed_sample_support_segments(self):
        batch = self._make_batch([4, 2], num_actions=2)
        batch[SAMPLE_SUPPORT_FIELD] = PackedTensor.from_segments(
            [
                torch.full((2, 5), 1, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
                torch.full((1, 5), SAMPLE_SUPPORT_PADDING, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
            ]
        )

        microbatch = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=8)._create_microbatch_from_indices([1])

        support = microbatch[SAMPLE_SUPPORT_FIELD]
        assert support.sequence_lengths.tolist() == [1]
        assert torch.all(support.segment(0) == SAMPLE_SUPPORT_PADDING)

    def test_microbatch_selection_gathers_packed_route_segments(self):
        batch = self._make_batch([4, 2], num_actions=2)
        batch["rollout_expert_indices"] = PackedTensor.from_segments(
            [torch.full((4, 2, 3), 1, dtype=torch.int16), torch.full((2, 2, 3), 2, dtype=torch.int16)]
        )

        microbatch = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=8)._create_microbatch_from_indices([1])

        routes = microbatch["rollout_expert_indices"]
        assert routes.sequence_lengths.tolist() == [2]
        assert torch.equal(routes.segment(0), torch.full((2, 2, 3), 2, dtype=torch.int16))

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

    def _make_marked_loss_forward_batch(self):
        """Batch whose first token identifies each sample, plus a check that packing permutes it."""
        seq_lens = LOSS_FORWARD_SEQ_LENS
        batch = self._make_batch(seq_lens)
        for i, seq_len in enumerate(seq_lens):
            batch["sequences"][i, :seq_len] = MARKER_BASE + i
        reference = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=LOSS_FORWARD_MAX_TOKENS)
        packed_order = [i for mb in reference._microbatches for i in mb]
        assert packed_order != list(range(len(seq_lens))), "packing no longer permutes; pick new seq_lens"
        return batch

    @staticmethod
    def _expected_marker_outputs():
        return [{"logprobs": [float(MARKER_BASE + i)]} for i in range(len(LOSS_FORWARD_SEQ_LENS))]

    @staticmethod
    def _assert_within_token_budget(attention_mask: torch.Tensor):
        """A microbatch stays within the budget unless it holds one oversized sample."""
        tokens = int(attention_mask.sum())
        assert tokens <= LOSS_FORWARD_MAX_TOKENS or attention_mask.shape[0] == 1, (
            f"{attention_mask.shape[0]} samples with {tokens} tokens exceed "
            f"max_tokens_per_microbatch={LOSS_FORWARD_MAX_TOKENS}"
        )

    def test_worker_loss_forward_honors_token_budget(self, monkeypatch):
        """Base (FSDP) ``forward(loss_fn=...)`` packs by token budget, drops the padding
        microbatch's outputs, and returns per-sample outputs and metrics that match
        sample-based chunking."""
        batch = self._make_marked_loss_forward_batch()
        # Force one padding microbatch, as a DP peer with more microbatches would.
        monkeypatch.setattr(
            TokenBasedBatchIterator,
            "_sync_num_microbatches",
            lambda self: len(self._microbatches) + 1,
        )
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.workers.worker.all_reduce_metrics",
            lambda metrics, strategy, group=None, sum_loss_metrics=False: metrics,
        )

        class _StubWorker(PolicyWorkerBase):
            def __init__(self, cfg):
                self.cfg = cfg
                self.strategy = None
                self.device_mesh = SimpleNamespace(get_group=lambda name: None)
                self.microbatches = []

            def _forward_micro_with_loss(self, experience, loss_fn, loss_fn_config=None, return_per_token_outputs=True):
                self.microbatches.append(experience)
                markers = experience.sequences[:, 0].tolist()
                return {
                    "loss": float(experience.loss_mask.sum()),
                    "response_length": experience.num_actions,
                    "loss_fn_outputs": [{"logprobs": [float(m)]} for m in markers],
                }

        token_worker = _StubWorker(
            SimpleNamespace(micro_forward_batch_size_per_gpu=2, max_tokens_per_microbatch=LOSS_FORWARD_MAX_TOKENS)
        )
        token_output = token_worker.forward(batch, loss_fn="cross_entropy")

        assert token_output.loss_fn_outputs == self._expected_marker_outputs()
        padding = [e for e in token_worker.microbatches if e.metadata.get("is_padding_batch")]
        real = [e for e in token_worker.microbatches if not e.metadata.get("is_padding_batch")]
        assert len(padding) == 1
        for experience in real:
            self._assert_within_token_budget(experience.attention_mask)

        # Unset budget keeps fixed sample-count chunking in input order.
        sample_worker = _StubWorker(SimpleNamespace(micro_forward_batch_size_per_gpu=2, max_tokens_per_microbatch=-1))
        sample_output = sample_worker.forward(batch, loss_fn="cross_entropy")

        assert [e.sequences.shape[0] for e in sample_worker.microbatches] == [2, 2, 1]
        assert sample_output.loss_fn_outputs == self._expected_marker_outputs()
        assert token_output.metrics == sample_output.metrics

    @pytest.mark.megatron
    def test_megatron_worker_loss_forward_honors_token_budget(self, monkeypatch):
        """Megatron ``forward(loss_fn=...)`` packs by token budget, pads microbatches to a
        uniform size, and excludes padding rows and padding microbatches from outputs
        and metrics."""
        megatron_worker = _import_megatron_worker()
        batch = self._make_marked_loss_forward_batch()
        monkeypatch.setattr(
            TokenBasedBatchIterator,
            "_sync_num_microbatches",
            lambda self: len(self._microbatches) + 1,
        )
        monkeypatch.setattr(
            megatron_worker,
            "all_reduce_metrics",
            lambda metrics, strategy, group=None, sum_loss_metrics=False: metrics,
        )
        monkeypatch.setattr(
            megatron_worker,
            "mpu",
            SimpleNamespace(
                get_pipeline_model_parallel_rank=lambda: 0,
                get_data_parallel_group=lambda with_context_parallel=False: None,
            ),
        )

        class _StubModel:
            def __init__(self):
                self.calls = []

            def eval(self):
                pass

            def forward_backward_mini_batch(self, micro_batches, seq_len, micro_batch_size, forward_only, **kwargs):
                assert forward_only
                self.calls.append((micro_batches, micro_batch_size))
                return [
                    {
                        "loss": float(mb["loss_mask"].sum()),
                        # Mean-reduced; a padding microbatch would pull it off 1.0.
                        "policy_entropy": 100.0 if mb.get("is_padding_batch") else 1.0,
                        "loss_fn_outputs": [{"logprobs": [float(m)]} for m in mb["sequences"][:, 0].tolist()],
                    }
                    for mb in micro_batches
                ]

        def _run(max_tokens_per_microbatch):
            worker = megatron_worker.MegatronPolicyWorkerBase.__new__(megatron_worker.MegatronPolicyWorkerBase)
            worker.cfg = SimpleNamespace(
                micro_forward_batch_size_per_gpu=2,
                max_tokens_per_microbatch=max_tokens_per_microbatch,
                algorithm=SimpleNamespace(temperature=1.0),
            )
            worker.model = _StubModel()
            worker.strategy = None
            worker.enable_router_replay = False
            worker.enable_sample_support_replay = False
            worker.empty_cuda_cache = False
            output = worker.forward(batch, loss_fn="cross_entropy")
            assert len(worker.model.calls) == 1
            return output, *worker.model.calls[0]

        token_output, micro_batches, micro_batch_size = _run(LOSS_FORWARD_MAX_TOKENS)

        assert token_output.loss_fn_outputs == self._expected_marker_outputs()
        assert all(mb["sequences"].shape[0] == micro_batch_size for mb in micro_batches)
        assert [mb["is_padding_batch"] for mb in micro_batches].count(True) == 1
        assert all(mb["num_microbatches"] == len(micro_batches) for mb in micro_batches)
        assert all(mb["num_real_microbatches"] == len(micro_batches) - 1 for mb in micro_batches)
        for mb in micro_batches:
            if not mb["is_padding_batch"]:
                real_rows = mb["loss_mask"].sum(dim=1) > 0
                self._assert_within_token_budget(mb["attention_mask"][real_rows])
        assert token_output.metrics["policy_entropy"] == 1.0

        # Unset budget keeps fixed sample-count chunking with no padding.
        sample_output, micro_batches, micro_batch_size = _run(-1)

        assert micro_batch_size == 2
        assert [mb["sequences"].shape[0] for mb in micro_batches] == [2, 2, 1]
        assert not any(mb["is_padding_batch"] for mb in micro_batches)
        assert all(mb["num_real_microbatches"] == mb["num_microbatches"] for mb in micro_batches)
        assert sample_output.loss_fn_outputs == self._expected_marker_outputs()
        assert token_output.metrics == sample_output.metrics

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
