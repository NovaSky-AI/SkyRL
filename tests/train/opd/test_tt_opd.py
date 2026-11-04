"""CPU tests for TT-OPD's top-K term (skyrl/train/opd/utils.py): the weights, the advantages and the estimator.

Run with:
uv run --isolated --extra dev pytest tests/train/opd/test_tt_opd.py
"""

import math
from typing import Dict, Optional, Sequence

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import (
    PackedRaggedTensor,
    packed_ragged_padding_segments,
)
from skyrl.backends.skyrl_train.utils.sample_support import SampleSupportColumn
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.train.opd.config import OPDConfig
from skyrl.train.opd.utils import (
    STUDENT_LOGPROB_FLOOR,
    apply_opd_to_advantages,
    candidate_matrix,
    row_cells,
    topk_metrics,
    topk_weights,
    with_member_values,
)
from tests.backends.skyrl_train.sample_support_helpers import csr


def _scored_support(
    trajectories: Sequence[Dict[str, list]], log_p: np.ndarray, log_q: np.ndarray
) -> PackedRaggedTensor:
    """Scored token rows ``[sampled, candidates...]``, built the way ``OPDTrainer`` builds them.

    Each trajectory names its ``response`` ids and each token's captured ``candidates`` (``-1`` pads; a row of
    ``-1`` has none). The engine's logprobs come from ``log_p`` and the teacher's from ``log_q``, by token id.
    """
    rows = []
    for trajectory in trajectories:
        grid = np.asarray(trajectory["candidates"], dtype=np.int32)
        capture = csr(grid, logprobs=np.where(grid >= 0, log_p[np.maximum(grid, 0)], 0.0))
        candidates = candidate_matrix(trajectory["response"], capture)
        scores = np.where(candidates >= 0, log_q[np.maximum(candidates, 0)], -np.inf).astype(np.float32)
        rows.append(capture.with_teacher_scores(candidates, scores))
    return PackedRaggedTensor.from_rows(rows)


def _advantages(
    support: PackedRaggedTensor,
    sampled: torch.Tensor,
    log_p: np.ndarray,
    log_q: np.ndarray,
    opd: OPDConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rl_advantages: Optional[torch.Tensor] = None,
):
    """The trainer's two calls on ``[batch, response]`` sampled ids; returns the advantages and the weights."""
    loss_mask = torch.ones(sampled.shape) if loss_mask is None else loss_mask
    topk = topk_weights(support, loss_mask, opd)
    advantages, metrics = apply_opd_to_advantages(
        torch.zeros(sampled.shape) if rl_advantages is None else rl_advantages,
        torch.as_tensor(log_p, dtype=torch.float32)[sampled],
        torch.as_tensor(log_q, dtype=torch.float32)[sampled],
        loss_mask,
        opd,
        sampled_in_topk=topk.sampled_in_topk,
        topk_weight_sum=topk.token_weights,
    )
    return advantages, topk, metrics


# ---------------------------------------------------------------------------
# The worked example of plan/opd-blog/04-tt-opd.md §6
# ---------------------------------------------------------------------------

# Vocabulary {a, b, c, d} = ids 0..3.
EXAMPLE_P = np.log(np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32))
EXAMPLE_Q = np.log(np.array([0.6, 0.2, 0.1, 0.1], dtype=np.float32))
C_A = 0.5 * math.log(0.6 / 0.5)
C_B = 0.3 * math.log(0.2 / 0.3)


def test_worked_example_sampled_token_outside_the_top_k():
    support = _scored_support([{"response": [2], "candidates": [[0, 1]]}], EXAMPLE_P, EXAMPLE_Q)

    advantages, topk, _ = _advantages(support, torch.tensor([[2]]), EXAMPLE_P, EXAMPLE_Q, OPDConfig())

    np.testing.assert_allclose(topk.member_weights.numpy(), [-(C_A + C_B), C_A, C_B], rtol=1e-6)
    assert topk.sampled_in_topk.tolist() == [[False]]
    # A(c) + C = log(0.1 / 0.15) + C
    assert advantages.item() == pytest.approx(math.log(0.1 / 0.15) + C_A + C_B, rel=1e-6)
    assert advantages.item() == pytest.approx(-0.4359, abs=1e-4)


def test_worked_example_sampled_token_inside_the_top_k():
    support = _scored_support([{"response": [0], "candidates": [[0, 1]]}], EXAMPLE_P, EXAMPLE_Q)

    advantages, topk, _ = _advantages(support, torch.tensor([[0]]), EXAMPLE_P, EXAMPLE_Q, OPDConfig())

    # The row weights do not depend on which token was sampled; A(a) arrives through c_a, not the tail.
    np.testing.assert_allclose(topk.member_weights.numpy(), [-(C_A + C_B), C_A, C_B], rtol=1e-6)
    assert topk.sampled_in_topk.tolist() == [[True]]
    assert advantages.item() == pytest.approx(C_A + C_B, rel=1e-6)


# ---------------------------------------------------------------------------
# Unbiasedness, by exact enumeration of the sampled token
# ---------------------------------------------------------------------------


def _split_gradient(
    logits: torch.Tensor, y: int, advantage: float, support: Optional[PackedRaggedTensor] = None
) -> torch.Tensor:
    """The gradient the policy loss plus the logit term ascend at one position: A·log p(y) + sum_v w_v l_v."""
    logits = logits.detach().clone().requires_grad_(True)
    objective = advantage * torch.log_softmax(logits, dim=-1)[y]
    if support is not None:
        ids = support.columns[SampleSupportColumn.TOKEN_IDS].long()
        weights = support.columns[SampleSupportColumn.MEMBER_VALUES].to(logits.dtype)
        objective = objective + (weights * logits[ids]).sum()
    (gradient,) = torch.autograd.grad(objective, logits)
    return gradient


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("top_k", [0, 2, 4])
def test_the_split_estimator_is_unbiased_for_the_reverse_kl_gradient(seed, top_k):
    """Weighted by p(y) over every y, the split form's gradient is -grad KL(p || q) on the logits.

    Covers a sampled token inside and outside S; top_k=0 is sampled-token OPD.
    """
    generator = torch.Generator().manual_seed(seed)
    vocab = 7
    logits = torch.randn(vocab, dtype=torch.float64, generator=generator) * 1.5
    log_q64 = torch.log_softmax(torch.randn(vocab, dtype=torch.float64, generator=generator) * 1.5, dim=-1)
    log_p = torch.log_softmax(logits, dim=-1).float().numpy()
    log_q = log_q64.float().numpy()
    top = np.argsort(-log_p)[:top_k].tolist()

    expected_gradient = torch.zeros(vocab, dtype=torch.float64)
    for y in range(vocab):
        if top_k:
            support = _scored_support([{"response": [y], "candidates": [top]}], log_p, log_q)
            advantages, topk, _ = _advantages(support, torch.tensor([[y]]), log_p, log_q, OPDConfig())
            weighted = with_member_values(support, topk.member_weights)
        else:
            advantages, _ = apply_opd_to_advantages(
                torch.zeros(1, 1),
                torch.tensor([[log_p[y]]]),
                torch.tensor([[log_q[y]]]),
                torch.ones(1, 1),
                OPDConfig(),
            )
            weighted = None
        expected_gradient += math.exp(log_p[y]) * _split_gradient(logits, y, advantages.item(), weighted)

    reference = logits.detach().clone().requires_grad_(True)
    log_p_reference = torch.log_softmax(reference, dim=-1)
    reverse_kl = (log_p_reference.exp() * (log_p_reference - log_q64)).sum()
    (kl_gradient,) = torch.autograd.grad(reverse_kl, reference)
    torch.testing.assert_close(expected_gradient, -kl_gradient, atol=2e-6, rtol=0)


def test_the_split_form_equals_the_direct_gradient_of_eq_15():
    """For every sampled y, A·log p(y) + sum_v w_v l_v has the gradient of
    sum_{v in S} c_v log p(v) + 1{y not in S} A(y) log p(y)."""
    generator = torch.Generator().manual_seed(3)
    vocab, top_k = 6, 3
    logits = torch.randn(vocab, dtype=torch.float64, generator=generator)
    log_p = torch.log_softmax(logits, dim=-1).float().numpy()
    log_q = torch.log_softmax(torch.randn(vocab, generator=generator), dim=-1).numpy()
    top = np.argsort(-log_p)[:top_k].tolist()
    coefficients = {v: math.exp(log_p[v]) * (log_q[v] - log_p[v]) for v in top}

    for y in range(vocab):
        support = _scored_support([{"response": [y], "candidates": [top]}], log_p, log_q)
        advantages, topk, _ = _advantages(support, torch.tensor([[y]]), log_p, log_q, OPDConfig())
        split = _split_gradient(logits, y, advantages.item(), with_member_values(support, topk.member_weights))

        direct_logits = logits.detach().clone().requires_grad_(True)
        log_probs = torch.log_softmax(direct_logits, dim=-1)
        objective = sum(coefficients[v] * log_probs[v] for v in top)
        if y not in top:
            objective = objective + (log_q[y] - log_p[y]) * log_probs[y]
        (direct,) = torch.autograd.grad(objective, direct_logits)
        torch.testing.assert_close(split, direct, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# top_k = 0 is PR 2's sampled-token term, bit for bit
# ---------------------------------------------------------------------------


def _pr2_apply_opd_to_advantages(advantages, action_log_probs, teacher_logprobs, loss_mask, opd):
    """PR 2's inline regulation, kept verbatim as the reference for ``teacher_signal``'s extraction."""
    mask = loss_mask.to(dtype=torch.bool)
    action_log_probs = action_log_probs.clamp(min=STUDENT_LOGPROB_FLOOR)
    reverse_kl = torch.where(mask, action_log_probs - teacher_logprobs, torch.zeros_like(action_log_probs))
    signal = -reverse_kl
    divergence_clipped = torch.zeros_like(mask)
    if opd.divergence_clip is not None:
        divergence_clipped = mask & (action_log_probs.exp() * reverse_kl > opd.divergence_clip)
        signal = torch.where(divergence_clipped, torch.zeros_like(signal), signal)
    advantage_clipped = torch.zeros_like(mask)
    if opd.advantage_clip_min is not None or opd.advantage_clip_max is not None:
        clamped = signal.clamp(min=opd.advantage_clip_min, max=opd.advantage_clip_max)
        advantage_clipped = clamped != signal
        signal = clamped
    if opd.advantage_log_compress:
        signal = signal.sign() * signal.abs().log1p()
    opd_term = opd.kl_coef * signal
    mask_f = mask.to(dtype=advantages.dtype)
    metrics = {
        "opd/reverse_kl": masked_mean(reverse_kl, mask_f).item(),
        "opd/adv_opd_abs_mean": masked_mean(opd_term.abs(), mask_f).item(),
        "opd/divergence_clip_fraction": masked_mean(divergence_clipped.to(mask_f.dtype), mask_f).item(),
        "opd/advantage_clip_fraction": masked_mean(advantage_clipped.to(mask_f.dtype), mask_f).item(),
    }
    return advantages + opd_term, metrics


@pytest.mark.parametrize(
    "opd",
    [
        OPDConfig(),
        OPDConfig(kl_coef=0.5),
        OPDConfig(divergence_clip=0.05),
        OPDConfig(advantage_clip_min=-0.5, advantage_clip_max=0.3),
        OPDConfig(advantage_clip_min=0.0, advantage_clip_max=0.0),
        OPDConfig(advantage_log_compress=True),
        OPDConfig(divergence_clip=0.05, advantage_log_compress=True, kl_coef=2.0),
    ],
)
def test_top_k_zero_matches_pr2_for_every_regulation_setting(opd):
    generator = torch.Generator().manual_seed(0)
    student = torch.log_softmax(torch.randn(4, 6, 9, generator=generator), dim=-1)[..., 0]
    student[0, 1] = float("-inf")  # floored
    teacher = torch.log_softmax(torch.randn(4, 6, 9, generator=generator), dim=-1)[..., 0]
    loss_mask = (torch.rand(4, 6, generator=generator) > 0.3).float()
    rl_advantages = torch.randn(4, 6, generator=generator)

    actual, metrics = apply_opd_to_advantages(rl_advantages, student, teacher, loss_mask, opd)
    expected, expected_metrics = _pr2_apply_opd_to_advantages(rl_advantages, student, teacher, loss_mask, opd)

    assert torch.equal(actual, expected)
    for key, value in expected_metrics.items():
        assert metrics[key] == value, key


# ---------------------------------------------------------------------------
# One regulation in both terms
# ---------------------------------------------------------------------------


def test_a_divergence_clipped_candidate_drops_out_of_the_top_k_sum():
    # Candidate 1 holds p = 0.6 where the teacher has 0.01: its pointwise KL is about 2.4 nats.
    log_p = np.log(np.array([0.3, 0.6, 0.05, 0.05], dtype=np.float32))
    log_q = np.log(np.array([0.29, 0.01, 0.35, 0.35], dtype=np.float32))
    support = _scored_support([{"response": [2], "candidates": [[0, 1]]}], log_p, log_q)
    opd = OPDConfig(divergence_clip=1.0)

    _, topk, _ = _advantages(support, torch.tensor([[2]]), log_p, log_q, opd)

    c_0 = 0.3 * math.log(0.29 / 0.3)
    np.testing.assert_allclose(topk.member_weights.numpy(), [-c_0, c_0, 0.0], rtol=1e-5)


def test_a_divergence_clipped_sampled_token_gets_no_tail_term():
    log_p = np.log(np.array([0.1, 0.1, 0.8], dtype=np.float32))
    log_q = np.log(np.array([0.4, 0.4, 0.2], dtype=np.float32))
    support = _scored_support([{"response": [2], "candidates": [[0, 1]]}], log_p, log_q)

    advantages, topk, metrics = _advantages(support, torch.tensor([[2]]), log_p, log_q, OPDConfig(divergence_clip=0.5))

    # p(y) log(p/q) = 0.8 log 4 > 0.5: only C remains.
    assert advantages.item() == pytest.approx(topk.token_weights.item())
    assert metrics["opd/divergence_clip_fraction"] == 1.0


def test_the_advantage_bounds_act_on_both_terms_alike():
    # Candidate 0 and the sampled token 2 have the same log-ratio, log 3, above the bound.
    log_p = np.log(np.array([0.1, 0.4, 0.1, 0.4], dtype=np.float32))
    log_q = np.log(np.array([0.3, 0.3, 0.3, 0.1], dtype=np.float32))
    support = _scored_support([{"response": [2], "candidates": [[0, 1]]}], log_p, log_q)
    opd = OPDConfig(advantage_clip_min=-0.2, advantage_clip_max=0.5)

    advantages, topk, _ = _advantages(support, torch.tensor([[2]]), log_p, log_q, opd)

    c_0, c_1 = topk.member_weights[1].item(), topk.member_weights[2].item()
    assert c_0 == pytest.approx(0.1 * 0.5) and c_1 == pytest.approx(0.4 * -0.2)
    assert advantages.item() == pytest.approx(0.5 + c_0 + c_1)


# ---------------------------------------------------------------------------
# topk_weights on a batch
# ---------------------------------------------------------------------------

BATCH_P = np.log(np.linspace(0.5, 1.5, 16) / np.linspace(0.5, 1.5, 16).sum()).astype(np.float32)
BATCH_Q = np.log(np.linspace(1.5, 0.5, 16) / np.linspace(1.5, 0.5, 16).sum()).astype(np.float32)


def _batch_support() -> PackedRaggedTensor:
    """Two trajectories: three tokens, then two. The last token of each has no captured row."""
    return _scored_support(
        [
            {"response": [5, 7, 9], "candidates": [[5, 6], [1, -1], [-1, -1]]},
            {"response": [3, 4], "candidates": [[2, 8], [-1, -1]]},
        ],
        BATCH_P,
        BATCH_Q,
    )


def test_every_token_row_sums_to_zero():
    support = _batch_support()
    topk = topk_weights(support, torch.ones(2, 3), OPDConfig())

    rows = torch.arange(support.num_rows).repeat_interleave(support.row_lengths.long())
    row_sums = torch.zeros(support.num_rows).index_add(0, rows, topk.member_weights)
    torch.testing.assert_close(row_sums, torch.zeros(support.num_rows), atol=1e-7, rtol=0)


def test_rows_land_right_aligned_and_empty_rows_get_plain_opd():
    support = _batch_support()
    sampled = torch.tensor([[5, 7, 9], [0, 3, 4]])  # row 1 is left-padded
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])

    advantages, topk, _ = _advantages(support, sampled, BATCH_P, BATCH_Q, OPDConfig(), loss_mask=loss_mask)

    def c(v):
        return math.exp(BATCH_P[v]) * (BATCH_Q[v] - BATCH_P[v])

    def a(v):
        return BATCH_Q[v] - BATCH_P[v]

    expected_c = [[c(5) + c(6), c(1), 0.0], [0.0, c(2) + c(8), 0.0]]
    torch.testing.assert_close(topk.token_weights, torch.tensor(expected_c), atol=1e-6, rtol=0)
    # Token 5 is its own candidate; tokens without a row get plain sampled-token OPD.
    assert topk.sampled_in_topk.tolist() == [[True, False, False], [False, False, False]]
    expected = [[c(5) + c(6), a(7) + c(1), a(9)], [0.0, a(3) + c(2) + c(8), a(4)]]
    torch.testing.assert_close(advantages, torch.tensor(expected), atol=1e-6, rtol=0)
    # The rows cover their trajectory's last response columns.
    batch, column = row_cells(support, 3)
    assert list(zip(batch.tolist(), column.tolist())) == [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]


def test_padding_batch_rows_carry_no_weight():
    support = _batch_support()
    padded = PackedRaggedTensor.cat([support, packed_ragged_padding_segments(support, segment_lengths=[3])])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    topk = topk_weights(padded, loss_mask, OPDConfig())

    assert topk.token_weights[2].tolist() == [0.0, 0.0, 0.0]
    assert not topk.sampled_in_topk[2].any()
    torch.testing.assert_close(topk.member_weights, topk_weights(support, loss_mask[:2], OPDConfig()).member_weights)


def test_topk_metrics_cover_trainable_tokens():
    support = _batch_support()
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    metrics = topk_metrics(topk_weights(support, loss_mask, OPDConfig()), loss_mask)

    p, q = np.exp(BATCH_P), np.exp(BATCH_Q)
    assert metrics["opd/topk_mass_mean"] == pytest.approx((p[5] + p[6] + p[1] + p[2] + p[8]) / 3, rel=1e-5)
    assert metrics["opd/teacher_topk_mass_mean"] == pytest.approx((q[5] + q[6] + q[1] + q[2] + q[8]) / 3, rel=1e-5)
    assert metrics["opd/sampled_in_topk_fraction"] == pytest.approx(1 / 3)


def test_with_member_values_replaces_only_the_member_value_column():
    support = _batch_support()
    values = torch.arange(support.num_members, dtype=torch.float64)

    replaced = with_member_values(support, values)

    assert replaced.columns[SampleSupportColumn.MEMBER_VALUES].dtype == torch.float32
    assert replaced.columns[SampleSupportColumn.MEMBER_VALUES].tolist() == values.tolist()
    for column in (SampleSupportColumn.TOKEN_IDS, SampleSupportColumn.LOGPROBS):
        assert torch.equal(replaced.columns[column], support.columns[column])
    assert torch.equal(replaced.row_offsets, support.row_offsets)
    assert torch.equal(replaced.cu_seqlens, support.cu_seqlens)
