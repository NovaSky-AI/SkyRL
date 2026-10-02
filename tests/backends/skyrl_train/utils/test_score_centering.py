"""Tests for score centering (https://arxiv.org/abs/2609.20807)."""

import math

import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    compute_score_centering_loss,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.utils.torch_utils import (
    logprobs_and_topk_logprobs_from_logits,
)
from skyrl.train.config import AlgorithmConfig


def _reference_loss_single_token(
    train_logp, samp_logp, topk_ids, sampled_token, samp_token_logp, advantage, weight_fn, eps
):
    """Direct port of the paper's JAX `score_centering_loss` (Appendix A.3) for one token position."""
    train_head_logp = train_logp[topk_ids]
    head_weights = weight_fn(torch.exp(train_head_logp - samp_logp))
    train_tail_mass = torch.clamp(1 - torch.exp(train_head_logp).sum(), min=eps)
    samp_tail_mass = torch.clamp(1 - torch.exp(samp_logp).sum(), min=eps)
    tail_mass_ratio = samp_tail_mass / train_tail_mass
    tail_scale = tail_mass_ratio * weight_fn(1 / tail_mass_ratio)
    head_prob_residual = torch.exp(samp_logp) * head_weights - tail_scale * torch.exp(train_head_logp)
    logp_correction = (head_prob_residual.detach() * train_head_logp).sum()
    sampled_ratio = torch.exp(train_logp[sampled_token] - samp_token_logp)
    weighted_logp = weight_fn(sampled_ratio).detach() * train_logp[sampled_token]
    return -advantage * (weighted_logp - logp_correction)


def _identity_weight(r):
    return torch.ones_like(r)


def _mis_weight(r):
    return torch.where((r >= 0.5) & (r <= 5.0), r, torch.zeros_like(r))


@pytest.mark.parametrize("weight_fn", [_identity_weight, _mis_weight])
def test_matches_paper_reference_implementation(weight_fn):
    torch.manual_seed(0)
    vocab, k, batch, num_actions = 12, 4, 2, 3
    eps = 1e-6
    train_logits = torch.randn(batch, num_actions, vocab, requires_grad=True)
    samp_logits = train_logits.detach() + 0.3 * torch.randn(batch, num_actions, vocab)
    train_logp = torch.log_softmax(train_logits, dim=-1)
    samp_logp = torch.log_softmax(samp_logits, dim=-1)
    topk_logp, topk_ids = samp_logp.topk(k, dim=-1)
    sampled = torch.randint(0, vocab, (batch, num_actions))
    advantages = torch.randn(batch, num_actions)

    # Vectorized implementation under test: base weighted-PG term + centering term.
    sampled_train_logp = train_logp.gather(-1, sampled[..., None]).squeeze(-1)
    sampled_samp_logp = samp_logp.gather(-1, sampled[..., None]).squeeze(-1)
    w = weight_fn(torch.exp(sampled_train_logp - sampled_samp_logp)).detach()
    base_loss = -(w * advantages * sampled_train_logp)
    centering_loss, metrics = compute_score_centering_loss(
        advantages, train_logp.gather(-1, topk_ids), topk_logp, torch.ones(batch, num_actions), weight_fn, eps
    )
    loss = (base_loss + centering_loss).sum()

    ref = torch.stack(
        [
            _reference_loss_single_token(
                train_logp[b, t],
                topk_logp[b, t],
                topk_ids[b, t],
                sampled[b, t],
                sampled_samp_logp[b, t],
                advantages[b, t],
                weight_fn,
                eps,
            )
            for b in range(batch)
            for t in range(num_actions)
        ]
    ).sum()
    torch.testing.assert_close(loss, ref, atol=1e-6, rtol=1e-5)

    (grad,) = torch.autograd.grad(loss, train_logits, retain_graph=True)
    (ref_grad,) = torch.autograd.grad(ref, train_logits)
    torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=1e-5)
    assert set(metrics) == {
        "score_centering_sampler_head_mass",
        "score_centering_trainer_head_mass",
        "score_centering_tail_mass_ratio",
        "score_centering_residual_abs_sum",
        "score_centering_term_abs_mean",
        "score_centering_loss_abs_mean",
    }


@pytest.mark.parametrize("weight_fn", [_identity_weight, _mis_weight])
def test_expected_update_has_no_drift_under_constant_reward(weight_fn):
    """Eq. 9 / Eq. 10: with a constant reward the expected centered update is exactly zero under the
    reconstructed sampler distribution (top-k head from the sampler, tail from the rescaled trainer)."""
    torch.manual_seed(1)
    vocab, k = 10, 3
    train_logits = torch.randn(vocab, requires_grad=True)
    samp_logp = torch.log_softmax(train_logits.detach() + 0.5 * torch.randn(vocab), dim=-1)
    train_logp = torch.log_softmax(train_logits, dim=-1)
    topk_logp, topk_ids = samp_logp.topk(k)
    in_head = torch.zeros(vocab, dtype=torch.bool)
    in_head[topk_ids] = True
    rho = (1 - topk_logp.exp().sum()) / (1 - train_logp.detach()[topk_ids].exp().sum())
    # q_hat: sampler on the head, rescaled trainer on the tail.
    q_hat = torch.where(in_head, samp_logp.exp(), rho * train_logp.detach().exp())
    assert math.isclose(q_hat.sum().item(), 1.0, abs_tol=1e-5)

    expected_grad = torch.zeros(vocab)
    for y in range(vocab):
        sampled_train_logp = train_logp[y]
        sampled_samp_logp = torch.log(q_hat[y])
        w = weight_fn(torch.exp(sampled_train_logp - sampled_samp_logp)).detach()
        base_loss = -(w * 1.0 * sampled_train_logp)
        centering_loss, _ = compute_score_centering_loss(
            torch.ones(1, 1),
            train_logp[topk_ids].view(1, 1, k),
            topk_logp.view(1, 1, k),
            None,
            weight_fn,
            1e-9,
        )
        (grad,) = torch.autograd.grad(base_loss + centering_loss.sum(), train_logits, retain_graph=True)
        expected_grad += q_hat[y] * grad
    torch.testing.assert_close(expected_grad, torch.zeros(vocab), atol=1e-5, rtol=0)


def test_centering_is_noop_when_sampler_matches_trainer_on_full_vocab():
    torch.manual_seed(2)
    vocab = 8
    train_logits = torch.randn(1, 1, vocab, requires_grad=True)
    train_logp = torch.log_softmax(train_logits, dim=-1)
    centering_loss, _ = compute_score_centering_loss(
        torch.ones(1, 1), train_logp, train_logp.detach(), None, _identity_weight, 1e-6
    )
    (grad,) = torch.autograd.grad(centering_loss.sum(), train_logits)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-6, rtol=0)


def test_invalid_head_entries_carry_no_mass():
    # A `-inf` sampler logprob marks an absent head entry (padding / dummy rows) and must not
    # contribute to the residual, even though the trainer logprob at the padded id is finite.
    train_logp = torch.log_softmax(torch.randn(1, 1, 6), dim=-1).requires_grad_(True)
    ids = torch.tensor([[[0, 0, 0]]])
    samp = torch.tensor([[[0.0, -float("inf"), -float("inf")]]])  # dummy row: prob one on token 0
    centering_loss, metrics = compute_score_centering_loss(
        torch.ones(1, 1), train_logp.gather(-1, ids), samp, None, _identity_weight, 1e-6
    )
    # q_0 = 1, rho = eps / tail -> ~0, so the residual is q_0 * 1 - ~0 * p_0 = 1 on the dummy token only.
    assert metrics["score_centering_sampler_head_mass"] == pytest.approx(1.0)
    torch.testing.assert_close(centering_loss, train_logp[..., 0], atol=1e-5, rtol=1e-5)


def test_rollout_is_loss_adds_centering_term_when_enabled():
    torch.manual_seed(3)
    batch, num_actions, vocab, k = 2, 4, 9, 3
    train_logits = torch.randn(batch, num_actions, vocab, requires_grad=True)
    train_logp = torch.log_softmax(train_logits, dim=-1)
    samp_logp = torch.log_softmax(train_logits.detach() + 0.2 * torch.randn_like(train_logits), dim=-1)
    sampled = torch.randint(0, vocab, (batch, num_actions))
    log_probs = train_logp.gather(-1, sampled[..., None]).squeeze(-1)
    rollout_logprobs = samp_logp.gather(-1, sampled[..., None]).squeeze(-1)
    topk_logp, topk_ids = samp_logp.topk(k, dim=-1)
    topk_log_probs = train_logp.gather(-1, topk_ids)
    advantages = torch.randn(batch, num_actions)
    loss_mask = torch.ones(batch, num_actions)

    loss_fn = PolicyLossRegistry.get("rollout_is")
    cfg = AlgorithmConfig(policy_loss_type="rollout_is", eps_clip_low=0.5, eps_clip_high=4.0)
    base, base_metrics = loss_fn(log_probs, log_probs.detach(), advantages, cfg, loss_mask, rollout_logprobs)
    assert "score_centering_loss_abs_mean" not in base_metrics

    cfg.score_centering.enabled = True
    centered, metrics = loss_fn(
        log_probs,
        log_probs.detach(),
        advantages,
        cfg,
        loss_mask,
        rollout_logprobs,
        rollout_topk_logprobs=topk_logp,
        topk_log_probs=topk_log_probs,
    )
    assert "score_centering_loss_abs_mean" in metrics

    def calibrate(r):
        return torch.where((r > 0.5) & (r < 5.0), r, torch.zeros_like(r))

    expected_term, _ = compute_score_centering_loss(advantages, topk_log_probs, topk_logp, loss_mask, calibrate)
    torch.testing.assert_close(centered, base + expected_term.sum(), atol=1e-6, rtol=1e-5)

    with pytest.raises(AssertionError, match="score centering requires"):
        loss_fn(log_probs, log_probs.detach(), advantages, cfg, loss_mask, rollout_logprobs)


def test_reinforce_loss_is_plain_policy_gradient_plus_centering():
    torch.manual_seed(5)
    batch, num_actions, vocab, k = 2, 3, 7, 3
    train_logits = torch.randn(batch, num_actions, vocab, requires_grad=True)
    train_logp = torch.log_softmax(train_logits, dim=-1)
    samp_logp = torch.log_softmax(train_logits.detach() + 0.3 * torch.randn_like(train_logits), dim=-1)
    sampled = torch.randint(0, vocab, (batch, num_actions))
    log_probs = train_logp.gather(-1, sampled[..., None]).squeeze(-1)
    rollout_logprobs = samp_logp.gather(-1, sampled[..., None]).squeeze(-1)
    topk_logp, topk_ids = samp_logp.topk(k, dim=-1)
    topk_log_probs = train_logp.gather(-1, topk_ids)
    advantages = torch.randn(batch, num_actions)
    loss_mask = torch.ones(batch, num_actions)

    loss_fn = PolicyLossRegistry.get("reinforce")
    cfg = AlgorithmConfig(policy_loss_type="reinforce")
    base, _ = loss_fn(log_probs, log_probs.detach(), advantages, cfg, loss_mask, rollout_logprobs)
    torch.testing.assert_close(base, -(advantages * log_probs).sum())

    cfg.score_centering.enabled = True
    centered, metrics = loss_fn(
        log_probs,
        log_probs.detach(),
        advantages,
        cfg,
        loss_mask,
        rollout_logprobs,
        rollout_topk_logprobs=topk_logp,
        topk_log_probs=topk_log_probs,
    )
    expected_term, _ = compute_score_centering_loss(advantages, topk_log_probs, topk_logp, loss_mask, torch.ones_like)
    torch.testing.assert_close(centered, base + expected_term.sum(), atol=1e-6, rtol=1e-5)
    assert metrics["score_centering_term_abs_mean"] > 0


def test_rollout_is_centering_vanishes_inside_the_calibration_band():
    """With f(r) = r on the band, q_v f(p_v/q_v) = p_v and alpha = 1, so the centering coefficient is
    exactly zero for in-band head tokens: composed with `rollout_is`, score centering only acts on
    out-of-band tokens."""
    torch.manual_seed(6)
    vocab, k = 20, 5
    train_logp = torch.log_softmax(torch.randn(1, 4, vocab), dim=-1)
    samp_logp = torch.log_softmax(train_logp + 0.05 * torch.randn(1, 4, vocab), dim=-1)  # all ratios in band
    topk_logp, topk_ids = samp_logp.topk(k, dim=-1)
    calibrate = lambda r: torch.where((r > 0.5) & (r < 5.0), r, torch.zeros_like(r))  # noqa: E731
    loss, metrics = compute_score_centering_loss(
        torch.ones(1, 4), train_logp.gather(-1, topk_ids).requires_grad_(True), topk_logp, None, calibrate
    )
    assert metrics["score_centering_residual_abs_sum"] < 1e-5
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-5, rtol=0)


def test_padding_members_with_neg_inf_on_both_heads_stay_finite():
    """The support channel marks absent members with -inf on the sampler and the trainer side."""
    torch.manual_seed(0)
    batch, actions, k = 2, 3, 4
    advantages = torch.randn(batch, actions)
    sampler = torch.log_softmax(torch.randn(batch, actions, k), dim=-1)
    trainer = torch.log_softmax(torch.randn(batch, actions, k), dim=-1).requires_grad_(True)
    sampler = sampler.clone()
    sampler[:, :, -1] = float("-inf")
    trainer_padded = torch.where(torch.isfinite(sampler), trainer, torch.full_like(trainer, float("-inf")))
    loss_mask = torch.ones(batch, actions)

    centering_loss, metrics = compute_score_centering_loss(
        advantages, trainer_padded, sampler, loss_mask, weight_fn=torch.ones_like
    )

    assert torch.isfinite(centering_loss).all()
    centering_loss.sum().backward()
    assert torch.isfinite(trainer.grad).all()
    assert all(math.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("inplace_backward", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logprobs_and_topk_logprobs_from_logits_matches_log_softmax(inplace_backward, dtype):
    """Label and head logprobs from the chunked pass match a plain log_softmax, in value and gradient."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 40, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 40, (2, 5))
    topk_ids = torch.randint(0, 40, (2, 5, 4))

    # The in-place backward overwrites the logits buffer, so take the reference copy first.
    reference = logits.detach().clone().float().requires_grad_(True)
    label_logp, topk_logp = logprobs_and_topk_logprobs_from_logits(
        logits, labels, topk_ids, chunk_size=2, inplace_backward=inplace_backward
    )
    (label_logp.sum() + topk_logp.sum()).backward()

    log_softmax = torch.log_softmax(reference, dim=-1)
    expected_label = log_softmax.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_topk = log_softmax.gather(-1, topk_ids)
    (expected_label.sum() + expected_topk.sum()).backward()

    tol = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(label_logp.float(), expected_label, atol=tol, rtol=tol)
    torch.testing.assert_close(topk_logp.float(), expected_topk, atol=tol, rtol=tol)
    torch.testing.assert_close(logits.grad.float(), reference.grad, atol=tol, rtol=tol)
