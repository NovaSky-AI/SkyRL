"""
Tests for new algorithm features: DRO loss, .sum() reduction, IW SNR metrics,
KL-in-advantages, and efficient zero-variance filter.

uv run --isolated --extra dev -- pytest tests/cpu/algorithms/test_new_algorithm_features.py
"""

import pytest
import torch
import numpy as np
from skyrl_train.config import (
    AlgorithmConfig,
    DROConfig,
    CISPOConfig,
    OffPolicyCorrectionConfig,
)
from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    reduce_loss,
    compute_iw_metrics,
    compute_grpo_outcome_advantage,
)
from skyrl_train.utils.trainer_utils import zero_variance_filter


NULL_OFF_POLICY_CORR = OffPolicyCorrectionConfig(
    tis_ratio_type=None,
    sequence_mask_metric=None,
    outlier_token_is_threshold_low=None,
    outlier_token_is_threshold_high=None,
)


# ============================================================================
# D4: .sum() loss reduction
# ============================================================================


class TestSumReduction:
    def test_sum_reduction_matches_manual(self):
        loss = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        result = reduce_loss(loss, mask, "sum")
        expected = (loss * mask).sum()
        assert torch.allclose(result, expected)

    def test_sum_reduction_no_mask(self):
        loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = reduce_loss(loss, None, "sum")
        expected = loss.sum()
        assert torch.allclose(result, expected)

    def test_sum_scales_with_batch_size(self):
        """Sum reduction should scale linearly with batch size."""
        loss_small = torch.ones(2, 10)
        loss_big = torch.ones(4, 10)
        mask_small = torch.ones(2, 10)
        mask_big = torch.ones(4, 10)
        r_small = reduce_loss(loss_small, mask_small, "sum")
        r_big = reduce_loss(loss_big, mask_big, "sum")
        assert torch.allclose(r_big, 2.0 * r_small)

    def test_sum_vs_token_mean_differ(self):
        """Sum and token_mean should give different values when tokens vary."""
        loss = torch.ones(2, 10)
        mask = torch.ones(2, 10)
        r_sum = reduce_loss(loss, mask, "sum")
        r_mean = reduce_loss(loss, mask, "token_mean")
        assert r_sum.item() == 20.0
        assert r_mean.item() == 1.0


# ============================================================================
# D3: DRO loss function
# ============================================================================


class TestDROLoss:
    @pytest.fixture
    def dro_config(self):
        return AlgorithmConfig(
            policy_loss_type="dro",
            loss_reduction="token_mean",
            dro=DROConfig(beta=0.1),
            off_policy_correction=NULL_OFF_POLICY_CORR,
        )

    def test_dro_registered(self):
        loss_fn = PolicyLossRegistry.get("dro")
        assert loss_fn is not None

    def test_dro_returns_loss_and_metrics(self, dro_config):
        loss_fn = PolicyLossRegistry.get("dro")
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        advantages = torch.tensor([[1.0, 0.5, -0.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, dro_config, loss_mask=mask)
        assert isinstance(loss, torch.Tensor)
        assert "dro_quadratic_term" in metrics

    def test_dro_zero_divergence(self, dro_config):
        """When old and new policies match, quadratic term is zero."""
        loss_fn = PolicyLossRegistry.get("dro")
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        loss, metrics = loss_fn(log_probs, old_log_probs, advantages, dro_config, loss_mask=mask)
        assert metrics["dro_quadratic_term"] == pytest.approx(0.0, abs=1e-6)

    def test_dro_quadratic_penalty_grows(self, dro_config):
        """Quadratic penalty should increase with divergence."""
        loss_fn = PolicyLossRegistry.get("dro")
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        advantages = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        # Small divergence
        log_probs_small = torch.tensor([[-1.1, -2.1, -3.1]])
        _, m_small = loss_fn(log_probs_small, old_log_probs, advantages, dro_config, loss_mask=mask)

        # Large divergence
        log_probs_large = torch.tensor([[-2.0, -3.0, -4.0]])
        _, m_large = loss_fn(log_probs_large, old_log_probs, advantages, dro_config, loss_mask=mask)

        assert m_large["dro_quadratic_term"] > m_small["dro_quadratic_term"]

    def test_dro_positive_advantage_direction(self, dro_config):
        """With positive advantage and no divergence, DRO should encourage higher log_probs."""
        loss_fn = PolicyLossRegistry.get("dro")
        log_probs = torch.tensor([[-2.0]], requires_grad=True)
        old_log_probs = torch.tensor([[-2.0]])
        advantages = torch.tensor([[1.0]])
        mask = torch.tensor([[1.0]])

        loss, _ = loss_fn(log_probs, old_log_probs, advantages, dro_config, loss_mask=mask)
        loss.backward()
        # With positive advantage, gradient should push log_probs up (negative gradient on loss)
        assert log_probs.grad.item() < 0, "DRO should decrease loss when increasing log_prob for positive advantage"


# ============================================================================
# D5: IW SNR diagnostic metric
# ============================================================================


class TestIWMetrics:
    def test_all_ones_snr(self):
        """SNR should equal sqrt(N) when all weights are 1.0 (on-policy)."""
        n = 100
        ratio = torch.ones(n)
        metrics = compute_iw_metrics(ratio, None)
        expected_snr = n ** 0.5
        assert metrics["iw_snr"] == pytest.approx(expected_snr, rel=1e-3)

    def test_variable_weights_lower_snr(self):
        """Variable weights should produce lower SNR than uniform."""
        ratio_uniform = torch.ones(100)
        ratio_variable = torch.cat([torch.ones(50) * 0.1, torch.ones(50) * 10.0])
        snr_uniform = compute_iw_metrics(ratio_uniform, None)["iw_snr"]
        snr_variable = compute_iw_metrics(ratio_variable, None)["iw_snr"]
        assert snr_variable < snr_uniform

    def test_iw_metrics_with_mask(self):
        """Mask should exclude tokens from IW computation."""
        ratio = torch.tensor([1.0, 1.0, 100.0])  # outlier at position 2
        mask_all = torch.tensor([1.0, 1.0, 1.0])
        mask_no_outlier = torch.tensor([1.0, 1.0, 0.0])

        snr_all = compute_iw_metrics(ratio, mask_all)["iw_snr"]
        snr_masked = compute_iw_metrics(ratio, mask_no_outlier)["iw_snr"]
        assert snr_masked > snr_all  # masking the outlier should improve SNR

    def test_iw_percentiles(self):
        """Percentile metrics should be present and ordered."""
        ratio = torch.linspace(0.5, 2.0, 100)
        metrics = compute_iw_metrics(ratio, None)
        assert metrics["iw_p1"] < metrics["iw_p50"] < metrics["iw_p99"]

    def test_empty_mask(self):
        """Empty mask should return empty metrics."""
        ratio = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([0.0, 0.0, 0.0])
        metrics = compute_iw_metrics(ratio, mask)
        assert metrics == {}

    def test_cispo_includes_iw_metrics(self):
        """CISPO loss should include IW metrics in its output."""
        loss_fn = PolicyLossRegistry.get("cispo")
        config = AlgorithmConfig(
            policy_loss_type="cispo",
            loss_reduction="token_mean",
            cispo=CISPOConfig(cispo_eps_clip_low=0.0, cispo_eps_clip_high=5.0),
            off_policy_correction=NULL_OFF_POLICY_CORR,
        )
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        advantages = torch.tensor([[1.0, 0.5, -0.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        _, metrics = loss_fn(log_probs, old_log_probs, advantages, config, loss_mask=mask)
        assert "iw_snr" in metrics
        assert "iw_mean" in metrics
        assert "iw_p50" in metrics

    def test_is_loss_includes_iw_metrics(self):
        """Importance sampling loss should include IW metrics."""
        loss_fn = PolicyLossRegistry.get("importance_sampling")
        config = AlgorithmConfig(
            policy_loss_type="importance_sampling",
            loss_reduction="token_mean",
            off_policy_correction=NULL_OFF_POLICY_CORR,
        )
        log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        advantages = torch.tensor([[1.0, 0.5, -0.5]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])

        _, metrics = loss_fn(log_probs, old_log_probs, advantages, config, loss_mask=mask)
        assert "iw_snr" in metrics
        assert "iw_mean" in metrics


# ============================================================================
# D1: KL-in-advantages (unit test of the formula, not the trainer integration)
# ============================================================================


class TestKLInAdvantages:
    def test_kl_advantage_centering(self):
        """KL advantages should sum to approximately zero (batch-centered)."""
        # Simulate the formula: kl_adv = coef * (avg_kl - token_kl) * mask
        action_log_probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        base_log_probs = torch.tensor([[0.0, 0.1, 0.2, 0.3]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        coef = 0.01

        token_kl = (action_log_probs - base_log_probs) * mask
        avg_kl = token_kl.sum() / mask.sum()
        kl_advantage = coef * (avg_kl - token_kl) * mask

        # Sum should be approximately zero (batch-centered)
        assert kl_advantage.sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_high_drift_tokens_penalized(self):
        """Tokens drifting more than average should get negative KL advantage."""
        action_log_probs = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # token 3 drifts a lot
        base_log_probs = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        coef = 0.01

        token_kl = (action_log_probs - base_log_probs) * mask
        avg_kl = token_kl.sum() / mask.sum()
        kl_advantage = coef * (avg_kl - token_kl) * mask

        # Token 3 drifts the most, should have negative KL advantage
        assert kl_advantage[0, 3].item() < 0
        # Tokens 0-2 drift less than average, should have positive KL advantage
        assert kl_advantage[0, 0].item() > 0

    def test_no_drift_no_kl_advantage(self):
        """When policies match, KL advantages should be zero."""
        action_log_probs = torch.tensor([[0.1, 0.2, 0.3]])
        base_log_probs = torch.tensor([[0.1, 0.2, 0.3]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        coef = 0.01

        token_kl = (action_log_probs - base_log_probs) * mask
        avg_kl = token_kl.sum() / mask.sum()
        kl_advantage = coef * (avg_kl - token_kl) * mask

        assert torch.allclose(kl_advantage, torch.zeros_like(kl_advantage), atol=1e-6)


# ============================================================================
# D6: Zero-variance filter
# ============================================================================


class TestZeroVarianceFilter:
    def test_filter_removes_constant_groups(self):
        """Groups where all rewards are the same should be filtered."""
        rewards = [1.0, 1.0, 1.0, 0.5, 1.0, 0.0]
        uids = ["a", "a", "a", "b", "b", "b"]
        kept = zero_variance_filter(rewards, uids)
        # Group "a" has all 1.0 (zero variance) — filtered
        # Group "b" has mixed rewards — kept
        assert 0 not in kept  # uid "a" index 0
        assert 3 in kept or 4 in kept or 5 in kept  # uid "b" indices

    def test_filter_keeps_singleton(self):
        """Singleton groups should be kept regardless of variance."""
        rewards = [0.5]
        uids = ["x"]
        kept = zero_variance_filter(rewards, uids)
        assert kept == [0]

    def test_filter_keeps_mixed_groups(self):
        """Groups with different rewards should be kept."""
        rewards = [0.0, 1.0, 0.0, 1.0]
        uids = ["a", "a", "b", "b"]
        kept = zero_variance_filter(rewards, uids)
        assert set(kept) == {0, 1, 2, 3}

    def test_remove_mode_reduces_list_length(self):
        """Simulate the remove mode: list comprehension should reduce length."""
        rewards = [1.0, 1.0, 0.5, 1.0]
        uids = ["a", "a", "b", "b"]
        kept_indices = zero_variance_filter(rewards, uids)

        # Simulate remove mode
        filtered_rewards = [rewards[i] for i in kept_indices]
        filtered_uids = [uids[i] for i in kept_indices]

        assert len(filtered_rewards) <= len(rewards)
        # Group "a" has zero variance (all 1.0), so filtered
        # Group "b" has variance (0.5, 1.0), so kept
        assert len(filtered_rewards) == 2
        assert filtered_uids == ["b", "b"]


# ============================================================================
# D7: Per-token hard masking (IcePop-style)
# ============================================================================


class TestPerTokenMask:
    def test_tokens_in_bounds_kept(self):
        """Tokens within [1-eps, 1+eps] should be kept."""
        from skyrl_train.utils.off_policy_correction_utils import compute_token_mask
        from skyrl_train.config import OffPolicyCorrectionConfig

        # Identical logprobs → ratio = 1.0 → all in bounds
        old_lp = torch.tensor([[0.1, 0.2, 0.3]])
        rollout_lp = torch.tensor([[0.1, 0.2, 0.3]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        cfg = OffPolicyCorrectionConfig(token_mask_eps_low=0.2, token_mask_eps_high=0.2)

        token_mask, metrics = compute_token_mask(old_lp, rollout_lp, mask, cfg)
        assert token_mask.sum() == 3.0  # all kept
        assert metrics["token_mask_ratio"] == pytest.approx(0.0)

    def test_divergent_tokens_masked(self):
        """Tokens with ratio outside bounds should be zeroed."""
        from skyrl_train.utils.off_policy_correction_utils import compute_token_mask
        from skyrl_train.config import OffPolicyCorrectionConfig

        # Token 2: old_lp=1.0, rollout_lp=-1.0 → ratio = exp(2) ≈ 7.4 → way outside [0.8, 1.2]
        old_lp = torch.tensor([[0.0, 0.0, 1.0]])
        rollout_lp = torch.tensor([[0.0, 0.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0]])
        cfg = OffPolicyCorrectionConfig(token_mask_eps_low=0.2, token_mask_eps_high=0.2)

        token_mask, metrics = compute_token_mask(old_lp, rollout_lp, mask, cfg)
        assert token_mask[0, 0] == 1.0  # in bounds
        assert token_mask[0, 1] == 1.0  # in bounds
        assert token_mask[0, 2] == 0.0  # out of bounds → masked
        assert metrics["token_mask_ratio"] > 0

    def test_masked_tokens_ignored(self):
        """Tokens already masked (loss_mask=0) should not be counted."""
        from skyrl_train.utils.off_policy_correction_utils import compute_token_mask
        from skyrl_train.config import OffPolicyCorrectionConfig

        old_lp = torch.tensor([[0.0, 0.0, 1.0]])
        rollout_lp = torch.tensor([[0.0, 0.0, -1.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # token 2 already masked
        cfg = OffPolicyCorrectionConfig(token_mask_eps_low=0.2, token_mask_eps_high=0.2)

        token_mask, metrics = compute_token_mask(old_lp, rollout_lp, mask, cfg)
        # Token 2 is already masked, so it should not count in the ratio
        assert metrics["token_mask_ratio"] == pytest.approx(0.0)


# ============================================================================
# Config validation
# ============================================================================


class TestNewConfigFields:
    def test_default_values(self):
        cfg = AlgorithmConfig()
        assert cfg.use_kl_in_advantages is False
        assert cfg.kl_advantages_coef == 0.01
        assert cfg.kl_reference_source == "ref_model"
        assert cfg.dro.beta == 0.1
        assert cfg.zero_variance_filter_mode == "mask"

    def test_custom_values(self):
        cfg = AlgorithmConfig(
            use_kl_in_advantages=True,
            kl_advantages_coef=0.05,
            kl_reference_source="rollout",
            dro=DROConfig(beta=0.2),
            zero_variance_filter_mode="remove",
        )
        assert cfg.use_kl_in_advantages is True
        assert cfg.kl_advantages_coef == 0.05
        assert cfg.kl_reference_source == "rollout"
        assert cfg.dro.beta == 0.2
        assert cfg.zero_variance_filter_mode == "remove"
