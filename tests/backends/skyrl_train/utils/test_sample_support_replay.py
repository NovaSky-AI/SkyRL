"""Tests for support-conditioned scoring and synthetic-EOS fallback."""

import sys
import types
from typing import List, Tuple

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
)
from skyrl.backends.skyrl_train.utils import sample_support_replay
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    compute_sample_support_scores,
    reject_unsupported_sample_support_packing,
    sample_support_scores,
    synthetic_eos_logprobs,
)

VOCAB = 11
TOP_K = 4
DENSE_SCORER_KWARGS = dict(
    vocab_start_index=0,
    vocab_end_index=VOCAB,
    tp_group=None,
    inference_only=True,
    lm_head_weight=None,
    temperature=1.0,
    chunk_size=None,
    fused_backend="torch",
)


def _reference_support_logprobs(logits, sampled_ids, support_ids):
    """Renormalize over each row's recorded members with a dense full-vocabulary softmax."""
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


def test_support_scores_match_a_dense_reference_in_value_and_gradient():
    logits = torch.randn(2, 3, VOCAB, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 5, 1], [8, 3, 7]])
    support_ids = torch.tensor(
        [
            [[2, 4, 6, -1], [5, -1, -1, -1], [-1, -1, -1, -1]],
            [[8, 0, 9, 4], [3, 2, -1, -1], [7, 1, 5, -1]],
        ],
        dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
    )

    scores = sample_support_scores(
        logits,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
    )

    assert scores.entropy is None
    assert scores.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(scores.logprobs, _reference_support_logprobs(logits, sampled_ids, support_ids))

    scores.logprobs.sum().backward()
    reference_logits = logits.detach().clone().requires_grad_(True)
    _reference_support_logprobs(reference_logits, sampled_ids, support_ids).sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_support_scores_renormalize_over_the_support_not_the_vocabulary():
    """A row whose support omits most of the vocabulary must not match a full-vocab logprob."""
    logits = torch.randn(1, 1, VOCAB, dtype=torch.float64)

    scores = sample_support_scores(
        logits,
        torch.tensor([[3]]),
        torch.tensor([[[3, 7, -1, -1]]], dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
    )

    expected = logits[0, 0, 3] - torch.logsumexp(logits[0, 0, [3, 7]], dim=0)
    torch.testing.assert_close(scores.logprobs[0, 0], expected)
    assert scores.logprobs[0, 0] > logits[0, 0].log_softmax(dim=-1)[3]


def test_fused_projection_matches_explicit_logits_in_value_and_gradient():
    temperature = 0.7
    hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[1, 4, 7], [2, 5, 8]])
    support_ids = torch.tensor(
        [
            [[1, 0, 3], [4, 6, -1], [7, -1, -1]],
            [[2, 1, 8], [5, 4, -1], [8, 0, 6]],
        ],
        dtype=SAMPLE_SUPPORT_TORCH_DTYPE,
    )

    fused = sample_support_scores(
        hidden,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
        lm_head_weight=weight,
        temperature=temperature,
        chunk_size=4,
    ).logprobs
    fused.sum().backward()

    explicit_hidden = hidden.detach().clone().requires_grad_(True)
    explicit_weight = weight.detach().clone().requires_grad_(True)
    explicit = sample_support_scores(
        (explicit_hidden @ explicit_weight.T) / temperature,
        sampled_ids,
        support_ids,
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
    ).logprobs
    explicit.sum().backward()

    torch.testing.assert_close(fused, explicit, check_dtype=False)
    torch.testing.assert_close(hidden.grad, explicit_hidden.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.grad, explicit_weight.grad, rtol=1e-5, atol=1e-6)


def test_fused_projection_bounds_the_candidate_pair_temporary(monkeypatch):
    """No projection sees more than ``chunk_size`` candidate pairs at once."""
    chunk_size = 3
    widths: List[int] = []
    original = sample_support_replay._project_pair_chunk

    def spy(hidden, weight, row_ids, token_ids, temperature):
        widths.append(int(token_ids.numel()))
        return original(hidden, weight, row_ids, token_ids, temperature)

    monkeypatch.setattr(sample_support_replay, "_project_pair_chunk", spy)
    hidden = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)

    scores = sample_support_scores(
        hidden,
        torch.zeros((2, 4), dtype=torch.long),
        torch.randint(0, 9, (2, 4, TOP_K), dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        vocab_start_index=0,
        vocab_end_index=weight.shape[0],
        tp_group=None,
        lm_head_weight=weight,
        chunk_size=chunk_size,
    )
    scores.logprobs.sum().backward()

    assert max(widths) <= chunk_size
    assert len(widths) > 1
    # 2 * 4 * TOP_K support pairs plus 2 * 4 sampled pairs, all of them chunked.
    assert sum(widths) == 2 * 4 * TOP_K + 2 * 4
    assert hidden.grad is not None and weight.grad is not None


def test_support_ids_must_use_the_canonical_dtype():
    with pytest.raises(ValueError, match=str(SAMPLE_SUPPORT_TORCH_DTYPE)):
        sample_support_scores(
            torch.randn(1, VOCAB),
            torch.tensor([1]),
            torch.tensor([[1, 2]], dtype=torch.int64),
            vocab_start_index=0,
            vocab_end_index=VOCAB,
            tp_group=None,
        )


def _install_fake_distributed_logprob(monkeypatch, calls):
    """Stand in for the Megatron TP collectives, which cannot be imported on a CPU host."""
    model_utils = types.ModuleType("skyrl.backends.skyrl_train.distributed.megatron.model_utils")

    class DistributedLogprob:
        @staticmethod
        def apply(source, targets, *args):
            calls.append(source.shape)
            return source.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    model_utils.DistributedLogprob = DistributedLogprob
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)
    return model_utils


@pytest.fixture
def megatron_parallel_state(monkeypatch):
    """Let ``model_utils`` import for its pure-torch packed-index helpers."""
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)
    return mpu


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
    # One slot per trajectory whether or not it holds a fallback: the middle row holds none.
    assert calls == [torch.Size([1, 3, 5])]

    actual.sum().backward()
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 2, 2] = 1
    expected_grad[2, 1, 3] = 1
    torch.testing.assert_close(logits.grad, expected_grad)


def test_synthetic_eos_leaves_rows_without_a_fallback_at_zero(monkeypatch):
    calls = []
    _install_fake_distributed_logprob(monkeypatch, calls)

    actual = synthetic_eos_logprobs(
        torch.arange(2 * 3 * 5, dtype=torch.float64).reshape(2, 3, 5),
        torch.zeros((2, 3), dtype=torch.long),
        torch.zeros((2, 3), dtype=torch.bool),
        vocab_start_index=0,
        vocab_end_index=5,
        tp_group=object(),
        inference_only=True,
    )

    assert torch.all(actual == 0)
    # The collective still runs at full capacity, so TP ranks stay in step.
    assert calls == [torch.Size([1, 2, 5])]


def test_synthetic_eos_uses_packed_cp_trajectory_segments(monkeypatch):
    calls = []
    _install_fake_distributed_logprob(monkeypatch, calls)
    logits = torch.arange(4 * 5, dtype=torch.float64).reshape(1, 4, 5).requires_grad_(True)
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
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[False, True, False, True]]),
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
    # Two CP-local segments of two tokens each, so capacity is two rather than four.
    assert calls == [torch.Size([1, 2, 5])]


def test_synthetic_eos_rejects_a_second_unsupported_row_in_one_trajectory(monkeypatch):
    """``scatter_reduce(amin)`` keeps one slot, so a second row would silently score 0.0."""
    _install_fake_distributed_logprob(monkeypatch, [])

    with pytest.raises(ValueError, match="at most one loss-bearing token"):
        synthetic_eos_logprobs(
            torch.randn(2, 3, 5, dtype=torch.float64),
            torch.zeros((2, 3), dtype=torch.long),
            torch.tensor([[False, True, True], [False, False, False]]),
            vocab_start_index=0,
            vocab_end_index=5,
            tp_group=object(),
            inference_only=False,
        )


def test_synthetic_eos_fused_projection_keeps_capacity_and_chunk_bound(monkeypatch):
    calls = []
    model_utils = _install_fake_distributed_logprob(monkeypatch, calls)

    def fused_apply(backend, hidden, weight, targets, start, end, chunk_size, group, inference_only):
        calls.append((hidden.shape, chunk_size))
        return hidden[..., 0]

    model_utils._fused_lm_head_logprob_apply = fused_apply
    hidden = torch.arange(3 * 4 * 2, dtype=torch.float64).reshape(3, 4, 2).requires_grad_(True)

    actual = synthetic_eos_logprobs(
        hidden,
        torch.zeros((3, 4), dtype=torch.long),
        torch.tensor([[False] * 4, [False, True, False, False], [False] * 4]),
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


def test_multi_subsequence_rows_are_rejected():
    reject_unsupported_sample_support_packing(None)
    reject_unsupported_sample_support_packing([[7], [9]])

    with pytest.raises(ValueError, match="multi-subsequence"):
        reject_unsupported_sample_support_packing([[7], [4, 5]])


# (prompt_len, response_len) per trajectory, including unequal prompt lengths.
JOIN_LENGTHS: List[Tuple[int, int]] = [(1, 3), (2, 3)]


def _attention_mask(lengths: List[Tuple[int, int]]) -> torch.Tensor:
    """Left-padded, as ``convert_prompts_responses_to_batch_tensors`` builds it."""
    totals = [prompt + response for prompt, response in lengths]
    mask = torch.zeros((len(totals), max(totals)), dtype=torch.bool)
    for row, total in enumerate(totals):
        mask[row, max(totals) - total :] = True
    return mask


def _layout(lengths: List[Tuple[int, int]], *, packed: bool = False) -> TokenMetadataLayout:
    mask = _attention_mask(lengths)
    totals = [prompt + response for prompt, response in lengths]
    if not packed:
        return TokenMetadataLayout(
            attention_mask=mask,
            sequence_lengths=totals,
            aligned_sequence_length=max(totals),
        )
    return TokenMetadataLayout(
        attention_mask=mask,
        sequence_lengths=totals,
        aligned_sequence_length=sum(totals),
        padded_sequence_lengths=totals,
        cu_seqlens_padded=cu_seqlens_from_lengths(totals),
    )


def _support(lengths: List[Tuple[int, int]]) -> PackedTensor:
    """Disjoint support sets per row, so a misplaced join changes the renormalizer."""
    response_lens = [response for _, response in lengths]
    rows = [[(row_index * 3 + offset) % VOCAB for offset in range(3)] for row_index in range(sum(response_lens))]
    return PackedTensor(
        torch.tensor(rows, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths(response_lens),
    )


def _batch_tensors(lengths: List[Tuple[int, int]]):
    mask = _attention_mask(lengths)
    generator = torch.Generator().manual_seed(0)
    sequences = torch.randint(0, VOCAB, mask.shape, generator=generator)
    num_actions = max(response for _, response in lengths)
    loss_mask = torch.zeros((mask.shape[0], num_actions), dtype=torch.bool)
    for row, (_, response) in enumerate(lengths):
        loss_mask[row, num_actions - response :] = True
    logits = torch.randn((*mask.shape, VOCAB), dtype=torch.float64, generator=generator)
    return sequences, mask, loss_mask, num_actions, logits


def _reference_from_canonical_positions(logits, sequences, lengths, support: PackedTensor):
    """Score each response token at the canonical position whose logit predicts it."""
    expected = torch.zeros((logits.shape[0], logits.shape[1] - 1), dtype=logits.dtype)
    sequence_length = logits.shape[1]
    for row, (prompt, response) in enumerate(lengths):
        padding = sequence_length - prompt - response
        for offset in range(response):
            position = padding + prompt + offset - 1
            members = support.segment(row)[offset].long()
            expected[row, position] = logits[row, position, sequences[row, position + 1]] - torch.logsumexp(
                logits[row, position, members], dim=0
            )
    return expected


def _dense_scores(lengths, *, packed=False, support=None):
    sequences, mask, loss_mask, num_actions, logits = _batch_tensors(lengths)
    if packed:
        logits = torch.cat([logits[row, mask[row]] for row in range(logits.shape[0])], dim=0).unsqueeze(0)
    return compute_sample_support_scores(
        logits,
        sequences,
        loss_mask,
        _support(lengths) if support is None else support,
        num_actions,
        packed=packed,
        metadata_layout=_layout(lengths, packed=packed),
        **DENSE_SCORER_KWARGS,
    )


def test_row_id_join_scores_the_response_suffix_domain():
    sequences, _, _, _, logits = _batch_tensors(JOIN_LENGTHS)

    actual = _dense_scores(JOIN_LENGTHS).logprobs

    expected = _reference_from_canonical_positions(logits, sequences, JOIN_LENGTHS, _support(JOIN_LENGTHS))
    torch.testing.assert_close(actual, expected)


def test_packed_and_unpacked_joins_score_the_same_tokens(megatron_parallel_state):
    lengths = [(3, 3), (4, 2)]

    unpacked = _dense_scores(lengths).logprobs
    packed = _dense_scores(lengths, packed=True).logprobs

    torch.testing.assert_close(packed, unpacked)


def test_appended_eos_falls_back_to_the_full_vocabulary():
    """A loss-bearing token with an all-padding support row is scored densely, not as 0.0."""
    lengths = [(3, 3)]
    sequences, mask, _, _, logits = _batch_tensors(lengths)
    support = _support(lengths)
    support.values[-1] = SAMPLE_SUPPORT_PADDING

    scores = _dense_scores(lengths, support=support)

    eos_position = mask.shape[1] - 2
    expected = logits[0, eos_position].log_softmax(dim=-1)[sequences[0, eos_position + 1]]
    assert not scores.valid_mask[0, eos_position]
    torch.testing.assert_close(scores.logprobs[0, eos_position], expected)


def test_a_second_unsupported_response_token_is_rejected():
    lengths = [(3, 3)]
    support = _support(lengths)
    support.values[-2:] = SAMPLE_SUPPORT_PADDING

    with pytest.raises(ValueError, match="at most one loss-bearing token"):
        _dense_scores(lengths, support=support)


def test_replay_requires_support_a_loss_mask_and_a_layout():
    lengths = [(3, 3)]
    sequences, _, loss_mask, num_actions, logits = _batch_tensors(lengths)

    with pytest.raises(ValueError) as missing_support:
        compute_sample_support_scores(
            logits,
            sequences,
            loss_mask,
            None,
            num_actions,
            packed=False,
            metadata_layout=_layout(lengths),
            **DENSE_SCORER_KWARGS,
        )
    assert "generator.inference_engine.enable_return_sample_support_set" in str(missing_support.value)
    assert "SkyRLGymGenerator" in str(missing_support.value)
    with pytest.raises(ValueError, match="response loss mask"):
        compute_sample_support_scores(
            logits,
            sequences,
            None,
            _support(lengths),
            num_actions,
            packed=False,
            metadata_layout=_layout(lengths),
            **DENSE_SCORER_KWARGS,
        )
    with pytest.raises(ValueError, match="token metadata layout"):
        compute_sample_support_scores(
            logits,
            sequences,
            loss_mask,
            _support(lengths),
            num_actions,
            packed=False,
            metadata_layout=None,
            **DENSE_SCORER_KWARGS,
        )


def test_an_all_padding_microbatch_scores_nothing():
    """A synthetic batch row attends one token and generates nothing, so it holds no support row."""
    support = PackedTensor(
        torch.empty((0, TOP_K), dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths([0]),
    )
    mask = torch.zeros((1, 4), dtype=torch.bool)
    mask[0, 0] = True

    scores = compute_sample_support_scores(
        torch.randn((1, 4, VOCAB), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.long),
        torch.zeros((1, 2), dtype=torch.bool),
        support,
        2,
        packed=False,
        metadata_layout=TokenMetadataLayout(attention_mask=mask, sequence_lengths=[1], aligned_sequence_length=4),
        **DENSE_SCORER_KWARGS,
    )

    assert not scores.valid_mask.any()
    assert torch.all(scores.logprobs == 0)
