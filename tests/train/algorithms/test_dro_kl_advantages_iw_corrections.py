"""
Tests for DRO loss, KL-in-advantages, importance-weighted corrections, and related utilities.

Covers:
- DRO policy loss (5 tests)
- .sum() loss reduction (4 tests)
- IW metrics helper (6 tests)
- KL-in-advantages formula (3 tests)
- Zero-variance filter (4 tests)
- Per-token masking (3 tests)
- Config fields (2 tests)

Run with:
    uv run --isolated --extra dev pytest tests/train/algorithms/test_dro_kl_advantages_iw_corrections.py -v
"""

import math
from dataclasses import dataclass, field

import pytest
import torch

from skyrl.train.config import AlgorithmConfig, DROConfig, CISPOConfig, OffPolicyCorrectionConfig
from skyrl.train.utils.trainer_utils import zero_variance_filter


# ---------------------------------------------------------------------------
# Helpers -- lightweight stubs so we can test without Ray
# ---------------------------------------------------------------------------


def _make_algorithm_config(**overrides) -> AlgorithmConfig:
    """Create an AlgorithmConfig with sensible defaults and any overrides."""
    defaults = dict(
        policy_loss_type="regular",
        loss_reduction="token_mean",
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        clip_ratio_c=3.0,
        max_seq_len=128,
    )
    defaults.update(overrides)
    return AlgorithmConfig(**defaults)


def _simple_reduce_loss(loss, mask, mode, max_seq_len=None):
    """Re-implement reduce_loss locally to avoid Ray dependency at import time."""
    from skyrl.backends.skyrl_train.utils.ppo_utils import reduce_loss

    return reduce_loss(loss, mask, mode, max_seq_len)


def _compute_iw_metrics(log_probs, old_log_probs, mask=None):
    from skyrl.backends.skyrl_train.utils.ppo_utils import compute_iw_metrics

    return compute_iw_metrics(log_probs, old_log_probs, mask)


def _compute_token_mask(old_lp, rollout_lp, mask, eps_low, eps_high):
    from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import compute_token_mask

    return compute_token_mask(old_lp, rollout_lp, mask, eps_low, eps_high)


# ===================================================================
# 1. DRO loss tests (5)
# ===================================================================


class TestDROPolicyLoss:
    """Tests for the DRO (Distributionally Robust Optimization) policy loss."""

    @staticmethod
    def _call_dro(log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None):
        from skyrl.backends.skyrl_train.utils.ppo_utils import dro_policy_loss

        return dro_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask, rollout_logprobs)

    def test_dro_returns_scalar_loss(self):
        """DRO loss should return a scalar tensor."""
        cfg = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=0.1))
        lp = torch.zeros(2, 4)
        olp = torch.zeros(2, 4)
        adv = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        loss, metrics = self._call_dro(lp, olp, adv, cfg, mask)
        assert loss.dim() == 0, "DRO loss must be a scalar"

    def test_dro_loss_increases_with_beta(self):
        """Higher beta should give a larger (or equal) loss because the exponential
        tilt concentrates on worse-case tokens."""
        lp = torch.randn(4, 8) * 0.1
        olp = torch.zeros(4, 8)
        adv = torch.randn(4, 8)
        mask = torch.ones(4, 8)

        cfg_low = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=0.01))
        cfg_high = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=1.0))
        loss_low, _ = self._call_dro(lp, olp, adv, cfg_low, mask)
        loss_high, _ = self._call_dro(lp, olp, adv, cfg_high, mask)
        # With large beta the DRO tilt amplifies the worst-case region
        assert loss_high >= loss_low - 1e-5

    def test_dro_returns_clip_ratio_metric(self):
        """DRO metrics dict should contain clip_ratio."""
        cfg = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=0.5))
        lp = torch.zeros(2, 4)
        olp = torch.zeros(2, 4)
        adv = torch.ones(2, 4)
        mask = torch.ones(2, 4)
        _, metrics = self._call_dro(lp, olp, adv, cfg, mask)
        assert "clip_ratio" in metrics

    def test_dro_with_zero_beta_equivalent_to_mean(self):
        """When beta -> 0 the DRO loss degenerates to the mean of the
        elementwise loss.  Use a very small beta to approximate."""
        cfg = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=1e-4))
        lp = torch.zeros(2, 6)
        olp = torch.zeros(2, 6)
        adv = torch.ones(2, 6) * 2.0
        mask = torch.ones(2, 6)
        loss, _ = self._call_dro(lp, olp, adv, cfg, mask)
        # With ratio = 1, elementwise = -min(adv, adv) = -adv = -2
        # mean of -2 = -2; at tiny beta, DRO ~ mean
        assert abs(loss.item() - (-2.0)) < 0.1

    def test_dro_respects_loss_mask(self):
        """Masked tokens should not contribute to the DRO loss."""
        cfg = _make_algorithm_config(policy_loss_type="dro", dro=DROConfig(beta=0.1))
        lp = torch.zeros(1, 4)
        olp = torch.zeros(1, 4)
        adv = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask_full = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        loss_full, _ = self._call_dro(lp, olp, adv, cfg, mask_full)
        loss_half, _ = self._call_dro(lp, olp, adv, cfg, mask_half)
        # They should differ because different tokens are included
        assert loss_full.item() != pytest.approx(loss_half.item(), abs=1e-6)


# ===================================================================
# 2. Sum reduction tests (4)
# ===================================================================


class TestSumReduction:
    """Tests for the 'sum' loss reduction mode."""

    def test_sum_basic(self):
        """Sum reduction should return the masked sum of all elements."""
        loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.ones(2, 3)
        result = _simple_reduce_loss(loss, mask, "sum")
        assert result.item() == pytest.approx(21.0)

    def test_sum_with_mask(self):
        """Only masked positions should contribute to the sum."""
        loss = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        result = _simple_reduce_loss(loss, mask, "sum")
        assert result.item() == pytest.approx(4.0)

    def test_sum_no_mask(self):
        """When mask is None, sum all elements."""
        loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _simple_reduce_loss(loss, None, "sum")
        assert result.item() == pytest.approx(10.0)

    def test_sum_zero_mask(self):
        """All-zero mask should yield zero loss."""
        loss = torch.tensor([[5.0, 5.0]])
        mask = torch.tensor([[0.0, 0.0]])
        result = _simple_reduce_loss(loss, mask, "sum")
        assert result.item() == pytest.approx(0.0)


# ===================================================================
# 3. IW metrics tests (6)
# ===================================================================


class TestIWMetrics:
    """Tests for compute_iw_metrics helper."""

    def test_keys_present(self):
        """All four expected metric keys should be present."""
        lp = torch.zeros(2, 4)
        olp = torch.zeros(2, 4)
        m = _compute_iw_metrics(lp, olp)
        for key in ["iw_ratio_mean", "iw_ratio_std", "iw_ratio_max", "iw_ratio_min"]:
            assert key in m

    def test_identity_ratio(self):
        """When log_probs == old_log_probs, ratio should be 1.0 everywhere."""
        lp = torch.zeros(3, 5)
        olp = torch.zeros(3, 5)
        mask = torch.ones(3, 5)
        m = _compute_iw_metrics(lp, olp, mask)
        assert m["iw_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
        assert m["iw_ratio_min"] == pytest.approx(1.0, abs=1e-5)
        assert m["iw_ratio_max"] == pytest.approx(1.0, abs=1e-5)

    def test_mask_filters_positions(self):
        """Only unmasked positions should be considered."""
        lp = torch.tensor([[0.0, 1.0]])
        olp = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])
        m = _compute_iw_metrics(lp, olp, mask)
        # Only position 0 is valid (ratio = exp(0) = 1)
        assert m["iw_ratio_mean"] == pytest.approx(1.0, abs=1e-5)

    def test_std_nonzero_for_varied_ratios(self):
        """Std should be > 0 when ratios differ."""
        lp = torch.tensor([[0.0, 0.5, 1.0]])
        olp = torch.zeros(1, 3)
        mask = torch.ones(1, 3)
        m = _compute_iw_metrics(lp, olp, mask)
        assert m["iw_ratio_std"] > 0

    def test_empty_mask(self):
        """All-zero mask should return zeros gracefully."""
        lp = torch.ones(2, 3)
        olp = torch.zeros(2, 3)
        mask = torch.zeros(2, 3)
        m = _compute_iw_metrics(lp, olp, mask)
        assert m["iw_ratio_mean"] == 0.0

    def test_no_mask(self):
        """When mask is None, all positions should be used."""
        lp = torch.zeros(2, 2)
        olp = torch.zeros(2, 2)
        m = _compute_iw_metrics(lp, olp, None)
        assert m["iw_ratio_mean"] == pytest.approx(1.0, abs=1e-5)


# ===================================================================
# 4. KL-in-advantages formula (3)
# ===================================================================


class TestKLInAdvantages:
    """Tests for KL-in-advantages penalty logic.

    These test the formula: advantages_new = advantages - coef * KL
    """

    def test_penalty_reduces_advantages(self):
        """With positive KL, the penalty should reduce the advantages."""
        from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl

        action_lp = torch.zeros(2, 4)
        base_lp = torch.zeros(2, 4) + 0.5  # base > action => KL > 0 for k1
        mask = torch.ones(2, 4)
        kl = compute_approx_kl(action_lp, base_lp, loss_mask=mask, kl_estimator_type="k3")
        advantages = torch.ones(2, 4)
        coef = 0.1
        new_adv = advantages - coef * kl
        # KL should be non-negative for k3
        assert (new_adv <= advantages + 1e-6).all()

    def test_zero_coef_no_change(self):
        """When coef == 0, advantages should be unchanged."""
        from skyrl.backends.skyrl_train.utils.ppo_utils import compute_approx_kl

        action_lp = torch.randn(2, 4)
        base_lp = torch.randn(2, 4)
        mask = torch.ones(2, 4)
        kl = compute_approx_kl(action_lp, base_lp, loss_mask=mask, kl_estimator_type="k3")
        advantages = torch.randn(2, 4)
        new_adv = advantages - 0.0 * kl
        assert torch.allclose(new_adv, advantages)

    def test_config_defaults(self):
        """Default config should have use_kl_in_advantages=False, coef=0.01."""
        cfg = AlgorithmConfig()
        assert cfg.use_kl_in_advantages is False
        assert cfg.kl_advantages_coef == 0.01
        assert cfg.kl_reference_source == "ref_model"


# ===================================================================
# 5. Zero-variance filter (4)
# ===================================================================


class TestZeroVarianceFilter:
    """Tests for the zero_variance_filter utility."""

    def test_filters_zero_variance_groups(self):
        """Groups with all-same rewards should be filtered out."""
        rewards = [1.0, 1.0, 2.0, 3.0]
        uids = ["a", "a", "b", "b"]
        kept = zero_variance_filter(rewards, uids)
        # group "a" has variance 0, group "b" has variance > 0
        assert 0 not in kept
        assert 1 not in kept
        assert 2 in kept
        assert 3 in kept

    def test_keeps_singletons(self):
        """A UID with only one sample should always be kept."""
        rewards = [5.0]
        uids = ["only"]
        kept = zero_variance_filter(rewards, uids)
        assert kept == [0]

    def test_all_same_filtered(self):
        """If all groups have zero variance, the result should be empty."""
        rewards = [1.0, 1.0, 2.0, 2.0]
        uids = ["a", "a", "b", "b"]
        kept = zero_variance_filter(rewards, uids)
        assert len(kept) == 0

    def test_all_different_kept(self):
        """If all groups have positive variance, all indices are kept."""
        rewards = [1.0, 2.0, 3.0, 4.0]
        uids = ["x", "x", "y", "y"]
        kept = zero_variance_filter(rewards, uids)
        assert sorted(kept) == [0, 1, 2, 3]


# ===================================================================
# 6. Per-token masking (3)
# ===================================================================


class TestPerTokenMasking:
    """Tests for compute_token_mask."""

    def test_no_thresholds_returns_unchanged(self):
        """When both eps_low and eps_high are None, mask is unchanged."""
        old_lp = torch.zeros(2, 4)
        rollout_lp = torch.zeros(2, 4)
        mask = torch.ones(2, 4)
        new_mask, metrics = _compute_token_mask(old_lp, rollout_lp, mask, None, None)
        assert torch.equal(new_mask, mask)
        assert len(metrics) == 0

    def test_high_threshold_masks_outliers(self):
        """Tokens with IS ratio above eps_high should be masked."""
        # ratio = exp(old - rollout) = exp(5) >> 2.0
        old_lp = torch.tensor([[5.0, 0.0, 0.0]])
        rollout_lp = torch.tensor([[0.0, 0.0, 0.0]])
        mask = torch.ones(1, 3)
        new_mask, metrics = _compute_token_mask(old_lp, rollout_lp, mask, None, 2.0)
        # First token should be masked (ratio ~ exp(5) >> 2)
        assert new_mask[0, 0].item() == 0.0
        # Other tokens should remain
        assert new_mask[0, 1].item() == 1.0
        assert new_mask[0, 2].item() == 1.0
        assert "token_mask_over_high_ratio" in metrics

    def test_low_threshold_masks_outliers(self):
        """Tokens with IS ratio below eps_low should be masked."""
        # ratio = exp(-5) ~ 0.0067 < 0.1
        old_lp = torch.tensor([[-5.0, 0.0]])
        rollout_lp = torch.tensor([[0.0, 0.0]])
        mask = torch.ones(1, 2)
        new_mask, metrics = _compute_token_mask(old_lp, rollout_lp, mask, 0.1, None)
        assert new_mask[0, 0].item() == 0.0
        assert new_mask[0, 1].item() == 1.0
        assert "token_mask_under_low_ratio" in metrics


# ===================================================================
# 7. Config fields (2)
# ===================================================================


class TestConfigFields:
    """Tests for new config dataclass fields."""

    def test_dro_config_defaults(self):
        """DROConfig should have beta=0.1 by default."""
        cfg = DROConfig()
        assert cfg.beta == 0.1

    def test_algorithm_config_new_fields(self):
        """AlgorithmConfig should have the new DRO, KL-advantage, and
        zero_variance_filter_mode fields."""
        cfg = AlgorithmConfig()
        assert hasattr(cfg, "dro")
        assert isinstance(cfg.dro, DROConfig)
        assert hasattr(cfg, "use_kl_in_advantages")
        assert hasattr(cfg, "kl_advantages_coef")
        assert hasattr(cfg, "kl_reference_source")
        assert hasattr(cfg, "zero_variance_filter_mode")
        assert cfg.zero_variance_filter_mode == "mask"

        # OffPolicyCorrectionConfig new fields
        opc = OffPolicyCorrectionConfig()
        assert hasattr(opc, "token_mask_eps_low")
        assert hasattr(opc, "token_mask_eps_high")
        assert opc.token_mask_eps_low is None
        assert opc.token_mask_eps_high is None
