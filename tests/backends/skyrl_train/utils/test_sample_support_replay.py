import sys
import types

import pytest
import torch

from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    sample_support_logprobs,
    synthetic_eos_logprobs,
)
from skyrl.utils.token_metadata import TokenMetadataLayout


def _reference(logits, sampled_ids, support_ids):
    outputs = []
    for row_logits, sampled_id, support in zip(
        logits.reshape(-1, logits.shape[-1]),
        sampled_ids.reshape(-1),
        support_ids.reshape(-1, support_ids.shape[-1]),
        strict=True,
    ):
        members = support[support >= 0].long()
        outputs.append(
            row_logits.new_zeros(())
            if members.numel() == 0
            else row_logits[sampled_id] - torch.logsumexp(row_logits[members], dim=0)
        )
    return torch.stack(outputs).reshape(sampled_ids.shape)


def test_support_logprobs_match_reference_values_and_gradients():
    logits = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 5, 1], [8, 3, 7]])
    support_ids = torch.tensor(
        [
            [[2, 4, 6, -1], [5, -1, -1, -1], [-1, -1, -1, -1]],
            [[8, 0, 9, 4], [3, 2, -1, -1], [7, 1, 5, -1]],
        ],
        dtype=torch.int32,
    )

    actual, valid = sample_support_logprobs(
        logits,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=logits.shape[-1],
        tp_group=None,
    )
    expected = _reference(logits, sampled_ids, support_ids)

    assert valid.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    actual_grad = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_(True)
    _reference(reference_logits, sampled_ids, support_ids).sum().backward()
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_fused_selected_projection_matches_explicit_logits_with_pair_chunking():
    temperature = 0.7
    hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
    support_ids = torch.tensor(
        [
            [[1, 0, 3], [4, 6, -1], [7, -1, -1]],
            [[2, 1, 8], [5, 4, -1], [8, 0, 6]],
        ],
        dtype=torch.int32,
    )

    fused, _ = sample_support_logprobs(
        hidden,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
        lm_head_weight=weight,
        temperature=temperature,
        chunk_size=4,
    )
    fused.sum().backward()
    fused_hidden_grad = hidden.grad.clone()
    fused_weight_grad = weight.grad.clone()

    explicit_hidden = hidden.detach().clone().requires_grad_(True)
    explicit_weight = weight.detach().clone().requires_grad_(True)
    explicit_logits = (explicit_hidden @ explicit_weight.T) / temperature
    explicit, _ = sample_support_logprobs(
        explicit_logits,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
    )
    explicit.sum().backward()

    torch.testing.assert_close(fused, explicit, check_dtype=False)
    torch.testing.assert_close(fused_hidden_grad, explicit_hidden.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused_weight_grad, explicit_weight.grad, rtol=1e-5, atol=1e-6)


def test_support_ids_must_be_int32():
    with pytest.raises(ValueError, match="int32"):
        sample_support_logprobs(
            torch.randn(1, 5),
            torch.tensor([1]),
            torch.tensor([[1, 2]], dtype=torch.int64),
            vocab_start_index=0,
            vocab_end_index=5,
            tp_group=None,
        )


def _install_fake_distributed_logprob(monkeypatch, calls):
    model_utils = types.ModuleType("skyrl.backends.skyrl_train.distributed.megatron.model_utils")

    class DistributedLogprob:
        @staticmethod
        def apply(source, targets, *args):
            calls.append(source.shape)
            return source.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    model_utils.DistributedLogprob = DistributedLogprob
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    return model_utils


def test_synthetic_eos_uses_one_fixed_slot_per_unpacked_trajectory(monkeypatch):
    calls = []
    _install_fake_distributed_logprob(monkeypatch, calls)
    logits = torch.arange(3 * 4 * 5, dtype=torch.float64).reshape(3, 4, 5).requires_grad_(True)
    sampled_ids = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 0]])
    synthetic_eos_mask = torch.tensor(
        [[False, False, True, False], [False, False, False, False], [False, True, False, False]]
    )

    actual = synthetic_eos_logprobs(
        logits,
        sampled_ids,
        synthetic_eos_mask,
        vocab_start_index=0,
        vocab_end_index=5,
        tp_group=object(),
        inference_only=False,
    )

    expected = torch.zeros_like(actual)
    expected[0, 2] = logits.detach()[0, 2, 2]
    expected[2, 1] = logits.detach()[2, 1, 3]
    torch.testing.assert_close(actual, expected)
    assert calls == [torch.Size([1, 3, 5])]

    actual.sum().backward()
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 2, 2] = 1
    expected_grad[2, 1, 3] = 1
    torch.testing.assert_close(logits.grad, expected_grad)


def test_synthetic_eos_uses_packed_cp_trajectory_segments(monkeypatch):
    calls = []
    _install_fake_distributed_logprob(monkeypatch, calls)
    logits = torch.arange(4 * 5, dtype=torch.float64).reshape(1, 4, 5).requires_grad_(True)
    sampled_ids = torch.tensor([[0, 1, 2, 3]])
    synthetic_eos_mask = torch.tensor([[False, True, False, True]])
    layout = TokenMetadataLayout(
        attention_mask=torch.ones((2, 3), dtype=torch.bool),
        sequence_lengths=[3, 3],
        aligned_sequence_length=8,
        padded_sequence_lengths=[4, 4],
        cu_seqlens_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        context_parallel_size=2,
        context_parallel_rank=0,
    )

    actual = synthetic_eos_logprobs(
        logits,
        sampled_ids,
        synthetic_eos_mask,
        vocab_start_index=0,
        vocab_end_index=5,
        tp_group=object(),
        inference_only=False,
        metadata_layout=layout,
    )

    expected = torch.zeros_like(actual)
    expected[0, 1] = logits.detach()[0, 1, 1]
    expected[0, 3] = logits.detach()[0, 3, 3]
    torch.testing.assert_close(actual, expected)
    assert calls == [torch.Size([1, 2, 5])]


def test_synthetic_eos_fused_projection_keeps_capacity_and_chunk_bound(monkeypatch):
    calls = []
    model_utils = _install_fake_distributed_logprob(monkeypatch, calls)

    def fused_apply(backend, hidden, weight, targets, start, end, chunk_size, group, inference_only):
        calls.append((hidden.shape, chunk_size))
        return hidden[..., 0]

    model_utils._fused_lm_head_logprob_apply = fused_apply
    hidden = torch.arange(3 * 4 * 2, dtype=torch.float64).reshape(3, 4, 2).requires_grad_(True)
    sampled_ids = torch.zeros((3, 4), dtype=torch.long)
    synthetic_eos_mask = torch.tensor(
        [[False, False, False, False], [False, True, False, False], [False, False, False, False]]
    )

    actual = synthetic_eos_logprobs(
        hidden,
        sampled_ids,
        synthetic_eos_mask,
        vocab_start_index=0,
        vocab_end_index=5,
        tp_group=object(),
        inference_only=False,
        lm_head_weight=torch.ones((5, 2), dtype=torch.float64),
        chunk_size=2,
    )

    expected = torch.zeros_like(actual)
    expected[1, 1] = hidden.detach()[1, 1, 0]
    torch.testing.assert_close(actual, expected)
    assert calls == [(torch.Size([1, 3, 2]), 2)]
    actual.sum().backward()
    expected_grad = torch.zeros_like(hidden)
    expected_grad[1, 1, 0] = 1
    torch.testing.assert_close(hidden.grad, expected_grad)
