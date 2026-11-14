"""
CPU-only tests that PolicyWorkerBase / CriticWorkerBase.forward_backward exclude
fully-padding microbatches from metric aggregation, mirroring the Megatron-side
behavior (megatron_worker.py).

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/test_forward_backward_padding_metrics.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker import CriticWorkerBase, PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import TokenBasedBatchIterator

MAX_TOKENS_PER_MICROBATCH = 15


def _make_batch(seq_lens, num_actions=4):
    """Dummy TrainingInputBatch with variable sequence lengths."""
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


@pytest.fixture
def force_one_padding_microbatch(monkeypatch):
    """Force TokenBasedBatchIterator to add exactly one padding microbatch, as if another
    DP rank had packed one more microbatch (dist is not initialized in CPU tests, so
    _sync_num_microbatches would otherwise just return the local count)."""
    original = TokenBasedBatchIterator._sync_num_microbatches

    def one_extra(self):
        return original(self) + 1

    monkeypatch.setattr(TokenBasedBatchIterator, "_sync_num_microbatches", one_extra)


def _identity_all_reduce(d, op=None, group=None):
    return dict(d)


def test_policy_padding_microbatch_excluded_from_metrics(force_one_padding_microbatch):
    """Mean-reduced metrics must ignore padding microbatches; summed metrics are unchanged.

    [10, 5], [10, 5] pack into 2 real microbatches at 15 tokens; the fixture forces one
    extra padding microbatch. Without the skip, policy_entropy = (1.0 + 1.0 + 0.0) / 3.
    """
    worker = PolicyWorkerBase.__new__(PolicyWorkerBase)
    worker.cfg = SimpleNamespace(
        micro_train_batch_size_per_gpu=1,
        max_tokens_per_microbatch=MAX_TOKENS_PER_MICROBATCH,
        algorithm=SimpleNamespace(policy_loss_type="regular"),
    )
    worker.strategy = MagicMock()
    worker.strategy.all_reduce.side_effect = _identity_all_reduce
    worker.device_mesh = MagicMock()

    padding_flags = []

    def fake_forward_backward_micro(experience, microbatch_weight, loss_fn=None, loss_fn_config=None):
        is_padding = bool(experience.metadata and experience.metadata.get("is_padding_batch", False))
        padding_flags.append(is_padding)
        if is_padding:
            # A fully-padding microbatch has an all-zero loss mask, so its masked-mean
            # metrics come out as exactly 0, and its loss_fn_outputs are dummy entries
            # (see TokenBasedBatchIterator._create_padding_microbatch).
            return {"policy_entropy": 0.0, "policy_loss": 0.0, "loss_fn_outputs": [{"logprobs": [0.0]}]}
        return {
            "policy_entropy": 1.0,
            "policy_loss": 0.5,
            "loss_fn_outputs": [{"logprobs": [1.0]}, {"logprobs": [1.0]}],
        }

    worker._forward_backward_micro = fake_forward_backward_micro

    out = worker.forward_backward(_make_batch([10, 10, 5, 5]))

    # The padding microbatch still ran forward/backward (collective parity across DP ranks)...
    assert len(padding_flags) == 3 and padding_flags.count(True) == 1
    # ...but is excluded from mean-reduced metrics (2/3 without the skip).
    assert out.metrics["policy_entropy"] == pytest.approx(1.0)
    # Summed metrics are unaffected either way: padding contributes 0 to a sum.
    assert out.metrics["policy_loss"] == pytest.approx(1.0)
    # Diagnostics still count the padding microbatch.
    assert out.metrics["num_microbatches"] == 3.0
    assert out.metrics["num_padding_microbatches"] == 1.0
    # loss_fn_outputs from the padding microbatch are excluded too: 2 real microbatches
    # x 2 samples each remain, with no dummy [0.0] entry.
    assert len(out.loss_fn_outputs) == 4
    assert all(entry == {"logprobs": [1.0]} for entry in out.loss_fn_outputs)


def test_critic_padding_microbatch_excluded_from_metrics(force_one_padding_microbatch):
    """critic_loss is mean-reduced on the critic path (reduce_metrics without
    sum_loss_metrics), so a padding microbatch's 0.0 would directly bias it:
    (0.5 + 0.5 + 0.0) / 3 without the skip."""
    worker = CriticWorkerBase.__new__(CriticWorkerBase)
    worker.cfg = SimpleNamespace(
        micro_train_batch_size_per_gpu=1,
        max_tokens_per_microbatch=MAX_TOKENS_PER_MICROBATCH,
    )
    worker.strategy = MagicMock()
    worker.strategy.all_reduce.side_effect = _identity_all_reduce

    def fake_forward_backward_micro(experience, microbatch_weight=None):
        if experience.metadata and experience.metadata.get("is_padding_batch", False):
            return {"critic_loss": 0.0}
        return {"critic_loss": 0.5}

    worker._forward_backward_micro = fake_forward_backward_micro

    out = worker.forward_backward(_make_batch([10, 10, 5, 5]))

    assert out.metrics["critic_loss"] == pytest.approx(0.5)
