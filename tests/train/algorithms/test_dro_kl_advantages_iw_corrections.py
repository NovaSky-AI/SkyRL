"""
Tests for DRO loss, sum reduction, zero-variance filter, and related config fields.

Covers:
- DRO policy loss (5 tests)
- .sum() loss reduction (4 tests)
- Zero-variance filter (4 tests)
- Config fields (1 test)

Run with:
    uv run --isolated --extra dev pytest tests/train/algorithms/test_dro_kl_advantages_iw_corrections.py -v
"""

import pytest
import torch

from skyrl.train.config import AlgorithmConfig, DROConfig, OffPolicyCorrectionConfig
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
# 3. Zero-variance filter (4)
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
# 4. Config fields (1)
# ===================================================================


class TestConfigFields:
    """Tests for new config dataclass fields."""

    def test_dro_config_defaults(self):
        """DROConfig should have beta=0.1 by default and be nested in AlgorithmConfig."""
        cfg = DROConfig()
        assert cfg.beta == 0.1

        algo_cfg = AlgorithmConfig()
        assert hasattr(algo_cfg, "dro")
        assert isinstance(algo_cfg.dro, DROConfig)

        # OffPolicyCorrectionConfig fields
        opc = OffPolicyCorrectionConfig()
        assert hasattr(opc, "token_mask_eps_low")
        assert hasattr(opc, "token_mask_eps_high")
        assert opc.token_mask_eps_low is None
        assert opc.token_mask_eps_high is None
