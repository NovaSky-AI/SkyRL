"""Head alignment and trainer-side head scoring for score centering on the sample-support channel.

Run with:
    uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/utils/test_score_centering_support.py
"""

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    canonical_token_metadata_layout,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_LOGPROBS_PADDING,
    SAMPLE_SUPPORT_LOGPROBS_TORCH_DTYPE,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.utils.score_centering_support import (
    compute_score_centering_head_logprobs,
    fused_label_and_head_logprobs,
    gather_packed_rows,
    sampled_in_head_fraction,
    sampler_head_for_actions,
    scatter_packed_rows_to_batch,
    score_head_members,
)

VOCAB = 40
K = 4


def _left_padded_batch(prompt_lens, response_lens, seed=0):
    """Build ``sequences``/``attention_mask`` in SkyRL's left-padded layout plus packed head rows.

    Response token ``j`` of trajectory ``i`` gets a head row whose first member is that token (so the
    sampled id is always in the head) followed by other ids, with ``K - 1 - (j % 2)`` valid members so
    some rows carry padding.
    """
    rng = np.random.default_rng(seed)
    batch = len(prompt_lens)
    max_prompt = max(prompt_lens)
    num_actions = max(response_lens)
    seq_len = max_prompt + num_actions
    sequences = torch.zeros((batch, seq_len), dtype=torch.long)
    attention_mask = torch.zeros((batch, seq_len), dtype=torch.long)
    support_rows, logprob_rows = [], []
    for i, (p, r) in enumerate(zip(prompt_lens, response_lens)):
        tokens = torch.from_numpy(rng.integers(1, VOCAB, size=p + r))
        # Left-padded block ending at the last position, so the response sits right-aligned in the
        # last ``num_actions`` positions, matching ``convert_prompts_responses_to_batch_tensors``.
        start = seq_len - p - r
        sequences[i, start : start + p + r] = tokens
        attention_mask[i, start : start + p + r] = 1
        ids = np.full((r, K), SAMPLE_SUPPORT_PADDING, dtype=np.int32)
        lps = np.full((r, K), SAMPLE_SUPPORT_LOGPROBS_PADDING, dtype=np.float32)
        for j in range(r):
            width = K - (j % 2)
            others = rng.choice([v for v in range(VOCAB) if v != int(tokens[p + j])], size=width - 1, replace=False)
            ids[j, 0] = int(tokens[p + j])
            ids[j, 1:width] = others
            lps[j, :width] = np.log(rng.dirichlet(np.ones(width)) * 0.9)
        support_rows.append(ids)
        logprob_rows.append(lps)
    lengths = np.array(response_lens)
    cu = cu_seqlens_from_lengths(lengths)
    support = PackedTensor(torch.from_numpy(np.concatenate(support_rows)).to(SAMPLE_SUPPORT_TORCH_DTYPE), cu)
    logprobs = PackedTensor(torch.from_numpy(np.concatenate(logprob_rows)).to(SAMPLE_SUPPORT_LOGPROBS_TORCH_DTYPE), cu)
    return sequences, attention_mask, num_actions, support, logprobs, support_rows, logprob_rows


def test_gather_packed_rows_pads_negative_row_ids():
    values = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    packed = PackedTensor(values, cu_seqlens_from_lengths(np.array([6])))
    row_ids = torch.tensor([[0, -1], [5, 2]])
    out = gather_packed_rows(packed, row_ids, -7.0)
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0, 0], values[0]) and torch.all(out[0, 1] == -7.0)
    assert torch.equal(out[1, 0], values[5]) and torch.equal(out[1, 1], values[2])


def test_sampler_head_for_actions_aligns_rows_to_response_positions():
    sequences, attention_mask, num_actions, support, logprobs, rows, lp_rows = _left_padded_batch([3, 5], [4, 2])
    head = sampler_head_for_actions(support, logprobs, attention_mask, num_actions)
    assert head.ids.shape == (2, num_actions, K) and head.logprobs.shape == (2, num_actions, K)
    responses = sequences[:, -num_actions:]
    for i, r in enumerate([4, 2]):
        # Right-aligned: the last ``r`` response positions hold this trajectory's rows in order.
        got_ids = head.ids[i, num_actions - r :]
        assert torch.equal(got_ids, torch.from_numpy(rows[i]).long())
        assert torch.allclose(head.logprobs[i, num_actions - r :], torch.from_numpy(lp_rows[i]))
        # The first member of every row is the sampled token at that position.
        assert torch.equal(got_ids[:, 0], responses[i, num_actions - r :])
        # Leading (padding) response positions carry padding rows.
        assert torch.all(head.ids[i, : num_actions - r] == SAMPLE_SUPPORT_PADDING)
        assert torch.all(torch.isneginf(head.logprobs[i, : num_actions - r]))
    loss_mask = (head.ids[..., 0] >= 0).long()
    assert sampled_in_head_fraction(head.ids, sequences, num_actions, loss_mask) == pytest.approx(1.0)
    # Shifting the head by one position breaks the alignment and the metric shows it.
    shifted = torch.roll(head.ids, shifts=1, dims=1)
    assert sampled_in_head_fraction(shifted, sequences, num_actions, loss_mask) < 0.5


def test_score_head_members_matches_full_softmax_and_backprops():
    torch.manual_seed(0)
    positions = 5
    logits = torch.randn(1, positions, VOCAB, requires_grad=True)
    sampled = torch.randint(0, VOCAB, (1, positions))
    head = torch.randint(0, VOCAB, (1, positions, K))
    head[0, 1, 2:] = SAMPLE_SUPPORT_PADDING
    log_softmax = torch.log_softmax(logits, dim=-1)
    sampled_logprobs = log_softmax.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

    member = score_head_members(
        logits,
        sampled,
        head,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        sampled_logprobs=sampled_logprobs,
    )
    valid = head >= 0
    expected = log_softmax.gather(-1, head.clamp(min=0))
    assert torch.allclose(member[valid], expected[valid], atol=1e-5)
    assert torch.all(torch.isneginf(member[~valid]))
    member[valid].sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # Renormalizing over the head gives a proper distribution on the recorded members.
    renorm = score_head_members(
        logits.detach(),
        sampled,
        head,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        sampled_logprobs=None,
        renormalize_over_head=True,
    )
    mass = torch.where(valid, renorm.exp(), torch.zeros_like(renorm)).sum(-1)
    assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)


def test_score_head_members_fused_lm_head_matches_logits_path():
    torch.manual_seed(1)
    hidden_size, positions = 8, 6
    hidden = torch.randn(1, positions, hidden_size)
    lm_head = torch.randn(VOCAB, hidden_size)
    temperature = 0.7
    logits = (hidden @ lm_head.T) / temperature
    sampled = torch.randint(0, VOCAB, (1, positions))
    head = torch.randint(0, VOCAB, (1, positions, K))
    log_softmax = torch.log_softmax(logits, dim=-1)
    sampled_logprobs = log_softmax.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    fused = score_head_members(
        hidden,
        sampled,
        head,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        sampled_logprobs=sampled_logprobs,
        lm_head_weight=lm_head,
        temperature=temperature,
        chunk_size=5,
    )
    assert torch.allclose(fused, log_softmax.gather(-1, head), atol=1e-5)


def test_compute_head_logprobs_unpacked_layout_matches_reference():
    torch.manual_seed(2)
    sequences, attention_mask, num_actions, support, logprobs, _, _ = _left_padded_batch([3, 5], [4, 2])
    batch, seq_len = sequences.shape
    logits = torch.randn(batch, seq_len, VOCAB)
    log_softmax = torch.log_softmax(logits, dim=-1)
    # Standard path: logprob of token t+1 at position t, canonical [batch, seq_len - 1].
    token_logprobs = log_softmax[:, :-1].gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
    layout = canonical_token_metadata_layout(attention_mask)
    label_logprobs, head_logprobs = compute_score_centering_head_logprobs(
        logits,
        sequences,
        support,
        token_logprobs,
        num_actions,
        packed=False,
        metadata_layout=layout,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        lm_head_weight=None,
        temperature=1.0,
        chunk_size=None,
    )
    assert label_logprobs is None  # materialized logits: the caller keeps its own label logprobs
    assert head_logprobs.shape == (batch, num_actions, K)
    # Reference: the head at response position j is scored by the logits one position earlier.
    head_ids = sampler_head_for_actions(support, logprobs, attention_mask, num_actions).ids
    predicting = log_softmax[:, seq_len - num_actions - 1 : seq_len - 1]
    expected = predicting.gather(-1, head_ids.clamp(min=0))
    valid = head_ids >= 0
    assert torch.allclose(head_logprobs[valid], expected[valid], atol=1e-5)
    assert torch.all(torch.isneginf(head_logprobs[~valid]))


def test_scatter_packed_rows_to_batch_restores_canonical_order():
    # Two trajectories of real lengths 3 and 2, padded to 4 tokens each in the THD layout.
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool)
    layout = TokenMetadataLayout(
        attention_mask=attention_mask,
        sequence_lengths=[3, 2],
        aligned_sequence_length=8,
        padded_sequence_lengths=[4, 4],
        cu_seqlens_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    values = torch.arange(8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    out = scatter_packed_rows_to_batch(values, layout, -1.0)
    assert out.shape == (2, 3, 2)
    # Trajectory 0 predicts 2 tokens from its first two packed positions; trajectory 1 predicts 1.
    assert torch.equal(out[0, 1], values[0, 0]) and torch.equal(out[0, 2], values[0, 1])
    assert torch.all(out[0, 0] == -1.0)
    assert torch.equal(out[1, 2], values[0, 4])
    assert torch.all(out[1, :2] == -1.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_label_and_head_logprobs_match_reference(dtype):
    """Label and head logprobs from the fused chunked pass match log_softmax of the same logits, with grads."""
    torch.manual_seed(3)
    hidden_size, positions, temperature = 16, 7, 0.9
    hidden = torch.randn(1, positions, hidden_size, dtype=dtype, requires_grad=True)
    weight = torch.randn(VOCAB, hidden_size, dtype=dtype, requires_grad=True)
    target = torch.randint(0, VOCAB, (1, positions))
    head = torch.randint(0, VOCAB, (1, positions, K))
    head[0, 2, 1:] = SAMPLE_SUPPORT_PADDING

    label, members = fused_label_and_head_logprobs(
        hidden,
        weight,
        target,
        head,
        vocab_start_index=0,
        vocab_end_index=VOCAB,
        tp_group=None,
        temperature=temperature,
        chunk_size=3,
    )
    valid = head >= 0
    (label.sum() + members[valid].sum()).backward()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    # Same numerics as the op: logits in the weight dtype (temperature folded into the weight), fp32 softmax.
    logits = torch.matmul(ref_hidden, (ref_weight / temperature).t()).float()
    log_softmax = torch.log_softmax(logits, dim=-1)
    ref_label = log_softmax.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    ref_members = log_softmax.gather(-1, head.clamp(min=0))
    (ref_label.sum() + ref_members[valid].sum()).backward()

    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(label, ref_label, atol=tol, rtol=tol)
    torch.testing.assert_close(members[valid], ref_members[valid], atol=tol, rtol=tol)
    assert torch.isneginf(members[~valid]).all()
    torch.testing.assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=tol * 5, rtol=tol * 5)
    torch.testing.assert_close(weight.grad.float(), ref_weight.grad.float(), atol=tol * 5, rtol=tol * 5)
