"""Tests for the DFT, ASFT, and KL-regularized SFT policy losses.

uv run --isolated --extra dev --extra skyrl-train -- pytest tests/train/algorithms/test_asft_loss.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.sft_loss_utils import (
    asft_loss,
    dft_loss,
    kl_reg_sft_loss,
)
from skyrl.train.config import AlgorithmConfig


def _cfg(coef=0.03, estimator="k3"):
    return AlgorithmConfig(asft_kl_coef=coef, asft_kl_estimator=estimator)


def test_dft_loss_matches_manual():
    """DFT loss = sum over masked tokens of -p(y).detach() * log p(y)."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, metrics = dft_loss(log_probs, None, None, config=_cfg(), loss_mask=loss_mask)

    w = torch.exp(log_probs)
    expected = (-(w * log_probs) * loss_mask).sum()
    assert torch.allclose(loss, expected)
    assert metrics["clip_ratio"] == 0.0


def test_asft_requires_a_reference():
    """A missing reference is a config error, not something to silently train through.

    Asking for an anchored loss and getting no anchor would train a different objective than
    requested; ``dft`` is the explicit way to opt out.
    """
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="require reference log-probs"):
        asft_loss(log_probs, None, None, config=_cfg(coef=0.1), loss_mask=loss_mask, base_log_probs=None)


def test_anchor_disabled_by_zero_coef_still_runs():
    """coef=0 is a legitimate 'anchor off' setting and must not need a reference forward's values."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = torch.tensor([[-0.7, -0.8, -2.5]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    asft, m = asft_loss(log_probs, None, None, config=_cfg(coef=0.0), loss_mask=loss_mask, base_log_probs=base)
    dft, _ = dft_loss(log_probs, None, None, config=_cfg(), loss_mask=loss_mask)
    assert torch.allclose(asft, dft)
    assert m["asft_kl"] == 0.0


def test_asft_zero_kl_when_policy_equals_reference():
    """When policy == reference, the KL anchor is ~0, so ASFT == DFT (the step-0 sanity check)."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = log_probs.clone()
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    for estimator in ("k1", "k3"):
        asft, m = asft_loss(
            log_probs, None, None, config=_cfg(coef=0.5, estimator=estimator), loss_mask=loss_mask, base_log_probs=base
        )
        dft, _ = dft_loss(log_probs, None, None, config=_cfg(), loss_mask=loss_mask)
        assert torch.allclose(asft, dft, atol=1e-6), estimator
        assert abs(m["asft_kl"]) < 1e-6, estimator


def test_asft_k3_matches_manual():
    """ASFT (k3) = DFT + coef * sum_masked(expm1(r) - r), r = base - policy."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = torch.tensor([[-0.7, -0.8, -2.5]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    coef = 0.2
    loss, _ = asft_loss(
        log_probs, None, None, config=_cfg(coef=coef, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )

    w = torch.exp(log_probs)
    dft_elem = -(w * log_probs)
    r = base - log_probs
    kl = torch.expm1(r) - r
    expected = ((dft_elem + coef * kl) * loss_mask).sum()
    assert torch.allclose(loss, expected)


def test_asft_k1_matches_manual():
    """ASFT (k1) uses kl = -(base - policy) = policy - base."""
    log_probs = torch.tensor([[-0.5, -1.0]])
    base = torch.tensor([[-0.7, -0.8]])
    loss_mask = torch.tensor([[1.0, 1.0]])
    coef = 0.3
    loss, _ = asft_loss(
        log_probs, None, None, config=_cfg(coef=coef, estimator="k1"), loss_mask=loss_mask, base_log_probs=base
    )
    w = torch.exp(log_probs)
    dft_elem = -(w * log_probs)
    kl = log_probs - base
    expected = ((dft_elem + coef * kl) * loss_mask).sum()
    assert torch.allclose(loss, expected)


def test_asft_gradient_only_through_policy():
    """DFT weight is detached and the reference is detached: grad flows only via log_probs."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]], requires_grad=True)
    base = torch.tensor([[-0.7, -0.8, -2.5]], requires_grad=True)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    loss, _ = asft_loss(
        log_probs, None, None, config=_cfg(coef=0.2, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    loss.backward()
    assert log_probs.grad is not None
    # Reference contributes no gradient (it is a frozen anchor).
    assert base.grad is None or torch.allclose(base.grad, torch.zeros_like(base))


def test_asft_k3_kl_is_nonnegative():
    """k3 estimator is non-negative per token, so the anchor never reduces the loss below DFT."""
    torch.manual_seed(0)
    log_probs = -torch.rand(4, 8).abs()
    base = -torch.rand(4, 8).abs()
    loss_mask = torch.ones(4, 8)
    asft, _ = asft_loss(
        log_probs, None, None, config=_cfg(coef=1.0, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    dft, _ = dft_loss(log_probs, None, None, config=_cfg(), loss_mask=loss_mask)
    assert asft.item() >= dft.item() - 1e-6


# ---- kl_reg_sft: cross-entropy + KL anchor (ASFT without the DFT p(y) weight) ----


def test_kl_reg_sft_requires_a_reference():
    """Same contract as asft: no reference is an error; use cross_entropy to opt out."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="require reference log-probs"):
        kl_reg_sft_loss(log_probs, None, None, config=_cfg(coef=0.1), loss_mask=loss_mask, base_log_probs=None)


def test_kl_reg_sft_base_term_is_uniform_cross_entropy():
    """With the anchor off, the base term is plain CE (weight 1), not p(y)-weighted."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = torch.tensor([[-0.7, -0.8, -2.5]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, m = kl_reg_sft_loss(log_probs, None, None, config=_cfg(coef=0.0), loss_mask=loss_mask, base_log_probs=base)
    expected = (-log_probs * loss_mask).sum()  # uniform weights, NOT p(y)-weighted
    assert torch.allclose(loss, expected)
    assert m["asft_kl"] == 0.0
    assert m["dft_weight_mean"] == 1.0


def test_kl_reg_sft_differs_from_asft_by_dft_weight_only():
    """kl_reg_sft and asft share the identical anchor term; only the NLL weight differs."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = torch.tensor([[-0.7, -0.8, -2.5]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    coef = 0.2
    anch, _ = kl_reg_sft_loss(
        log_probs, None, None, config=_cfg(coef=coef, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    asft, _ = asft_loss(
        log_probs, None, None, config=_cfg(coef=coef, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    w = torch.exp(log_probs)
    # asft base term is p(y)-weighted; kl_reg_sft base term is uniform. The difference between
    # the two losses must be exactly the difference in the (unanchored) base terms (CE - DFT),
    # since the anchor term is identical.
    base_term_diff = (-log_probs) - (-(w * log_probs))
    assert torch.allclose(anch - asft, (base_term_diff * loss_mask).sum())


def test_kl_reg_sft_k3_matches_manual():
    """kl_reg_sft (k3) = CE + coef * sum_masked(expm1(r) - r), r = base - policy."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = torch.tensor([[-0.7, -0.8, -2.5]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]])
    coef = 0.2
    loss, _ = kl_reg_sft_loss(
        log_probs, None, None, config=_cfg(coef=coef, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    ce_elem = -log_probs
    r = base - log_probs
    kl = torch.expm1(r) - r
    expected = ((ce_elem + coef * kl) * loss_mask).sum()
    assert torch.allclose(loss, expected)


def test_kl_reg_sft_zero_kl_when_policy_equals_reference():
    """policy == reference -> anchor ~0, so kl_reg_sft == plain CE."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]])
    base = log_probs.clone()
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    for estimator in ("k1", "k3"):
        loss, m = kl_reg_sft_loss(
            log_probs, None, None, config=_cfg(coef=0.5, estimator=estimator), loss_mask=loss_mask, base_log_probs=base
        )
        expected_ce = (-log_probs * loss_mask).sum()
        assert torch.allclose(loss, expected_ce, atol=1e-6), estimator
        assert abs(m["asft_kl"]) < 1e-6, estimator


def test_kl_reg_sft_gradient_only_through_policy():
    """Reference is detached: grad flows only via log_probs (uniform-weighted CE term + anchor)."""
    log_probs = torch.tensor([[-0.5, -1.0, -2.0]], requires_grad=True)
    base = torch.tensor([[-0.7, -0.8, -2.5]], requires_grad=True)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]])
    loss, _ = kl_reg_sft_loss(
        log_probs, None, None, config=_cfg(coef=0.2, estimator="k3"), loss_mask=loss_mask, base_log_probs=base
    )
    loss.backward()
    assert log_probs.grad is not None
    assert base.grad is None or torch.allclose(base.grad, torch.zeros_like(base))
