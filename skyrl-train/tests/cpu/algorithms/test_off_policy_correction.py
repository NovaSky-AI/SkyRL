"""
Tests for off policy correction utilities.

uv run --isolated --extra dev -- pytest tests/cpu/algorithms/test_off_policy_correction.py
"""

import pytest
import torch
from skyrl_train.config import OffPolicyCorrectionConfig, AlgorithmConfig

from skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl_train.utils.off_policy_correction_utils import (
    compute_tis_ratio,
    compute_sequence_mask,
    compute_outlier_token_mask,
    compute_off_policy_correction,
)


def test_compute_tis_ratio_token_level():
    """Tests token-level TIS ratio computation with capping."""
    device = "cpu"

    # old_log_probs - rollout_logprobs gives the log importance ratio
    # Token ratios: exp([0.5, -0.5, 1.0]) = [1.6487, 0.6065, 2.7183]
    old_log_probs = torch.tensor([[-1.0, -1.5, -0.5]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="token",
        token_tis_ratio_clip_high=2.0,
    )

    tis_ratio, metrics = compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "token", config)

    # Expected: [1.6487, 0.6065, 2.0] (third token capped at 2.0)
    expected = torch.tensor([[1.6487, 0.6065, 2.0]], device=device)
    torch.testing.assert_close(tis_ratio, expected, rtol=1e-3, atol=1e-4)
    # One token out of 3 was capped
    assert "tis_token_clip_high_ratio" in metrics
    assert abs(metrics["tis_token_clip_high_ratio"] - 1 / 3) < 0.01


def test_compute_tis_ratio_sequence_level():
    """Tests sequence-level TIS ratio computation with capping."""
    device = "cpu"

    # Token log ratios: [0.5, -0.5, 1.0]
    # Sequence log ratio (sum of masked): 0.5 + (-0.5) + 1.0 = 1.0
    # Sequence ratio: exp(1.0) = 2.7183
    old_log_probs = torch.tensor([[-1.0, -1.5, -0.5]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="sequence",
        sequence_tis_ratio_clip_high=5.0,
    )

    tis_ratio, metrics = compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "sequence", config)

    # Expected: exp(1.0) = 2.7183, shape [batch, 1] for sequence-level
    expected = torch.tensor([[2.7183]], device=device)
    torch.testing.assert_close(tis_ratio, expected, rtol=1e-3, atol=1e-4)
    # No sequence was capped (2.7183 < 5.0)
    assert "tis_seq_clip_high_ratio" in metrics
    assert metrics["tis_seq_clip_high_ratio"] == 0.0


def test_compute_tis_ratio_sequence_level_with_cap():
    """Tests sequence-level TIS ratio capping."""
    device = "cpu"

    # Token log ratios: [1.0, 1.0, 1.0]
    # Sequence log ratio: 3.0
    # Sequence ratio: exp(3.0) = 20.09, should be capped at 5.0
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="sequence",
        sequence_tis_ratio_clip_high=5.0,
    )

    tis_ratio, metrics = compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "sequence", config)

    # Expected: capped at 5.0, shape [batch, 1] for sequence-level
    expected = torch.tensor([[5.0]], device=device)
    torch.testing.assert_close(tis_ratio, expected, rtol=1e-3, atol=1e-4)
    # One sequence out of 1 was capped
    assert "tis_seq_clip_high_ratio" in metrics
    assert metrics["tis_seq_clip_high_ratio"] == 1.0


def test_compute_tis_ratio_with_mask():
    """Tests that loss_mask correctly excludes tokens from sequence-level computation."""
    device = "cpu"

    # Token log ratios: [0.5, -0.5, 1.0]
    # With mask [1, 0, 1], sequence log ratio = 0.5 + 1.0 = 1.5
    old_log_probs = torch.tensor([[-1.0, -1.5, -0.5]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 0.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="sequence",
        sequence_tis_ratio_clip_high=10.0,
    )

    tis_ratio, metrics = compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "sequence", config)

    # Expected: exp(1.5) = 4.4817, shape [batch, 1] for sequence-level
    expected_val = torch.exp(torch.tensor(1.5))
    expected = expected_val.reshape(1, 1)
    torch.testing.assert_close(tis_ratio, expected, rtol=1e-3, atol=1e-4)
    # No sequence was capped (4.4817 < 10.0)
    assert "tis_seq_clip_high_ratio" in metrics
    assert metrics["tis_seq_clip_high_ratio"] == 0.0


def test_compute_sequence_mask_geometric():
    """Tests geometric sequence mask computation."""
    device = "cpu"

    # Token log ratios: [0.1, -0.1, 0.0] -> sum = 0.0, geometric mean = exp(0/3) = 1.0
    old_log_probs = torch.tensor([[-1.0, -1.1, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.1, -1.0, -1.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        sequence_mask_metric="geometric",
        geo_mask_high=1.1,
        geo_mask_low=0.9,
    )

    sequence_mask, metrics = compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "geometric", config)

    # Geometric mean ≈ 1.0, which is within [0.9, 1.1], so mask should be 1.0
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(sequence_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked
    assert metrics["geo_sequence_mask_masked_ratio"] == 0.0
    assert metrics["geo_sequence_mask_over_high_ratio"] == 0.0
    assert metrics["geo_sequence_mask_under_low_ratio"] == 0.0


def test_compute_sequence_mask_geometric_rejects():
    """Tests geometric sequence mask correctly rejects sequences outside bounds."""
    device = "cpu"

    # Token log ratios: [0.5, 0.5, 0.5] -> sum = 1.5, geometric mean = exp(1.5/3) = exp(0.5) ≈ 1.6487
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.5, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        sequence_mask_metric="geometric",
        geo_mask_high=1.1,
        geo_mask_low=0.9,
    )

    sequence_mask, metrics = compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "geometric", config)

    # Geometric mean ≈ 1.6487, which is outside [0.9, 1.1], so mask should be 0.0
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[0.0]], device=device)
    torch.testing.assert_close(sequence_mask, expected, rtol=1e-3, atol=1e-4)
    # One sequence masked, over high cap
    assert metrics["geo_sequence_mask_masked_ratio"] == 1.0
    assert metrics["geo_sequence_mask_over_high_ratio"] == 1.0
    assert metrics["geo_sequence_mask_under_low_ratio"] == 0.0


def test_compute_sequence_mask_product():
    """Tests product sequence mask computation."""
    device = "cpu"

    # Token log ratios: [0.2, 0.1, 0.0] -> sum = 0.3, seq ratio = exp(0.3) ≈ 1.35
    old_log_probs = torch.tensor([[-1.0, -1.1, -1.2]], device=device)
    rollout_logprobs = torch.tensor([[-1.2, -1.2, -1.2]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        sequence_mask_metric="product",
        product_mask_high=2.0,
        product_mask_low=0.5,
    )

    sequence_mask, metrics = compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "product", config)

    # Sequence ratio ≈ 1.35, which is within [0.5, 2.0]
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(sequence_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked
    assert metrics["product_sequence_mask_masked_ratio"] == 0.0
    assert metrics["product_sequence_mask_over_high_ratio"] == 0.0
    assert metrics["product_sequence_mask_under_low_ratio"] == 0.0


def test_compute_sequence_mask_product_rejects_by_seq_ratio():
    """Tests product sequence mask rejects when product ratio is out of bounds."""
    device = "cpu"

    # Token log ratios: [1.0, 1.0, 1.0] -> sum = 3.0, seq ratio = exp(3.0) ≈ 20.09
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        sequence_mask_metric="product",
        product_mask_high=2.0,
        product_mask_low=0.5,
    )

    sequence_mask, metrics = compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "product", config)

    # Sequence ratio ≈ 20.09, which is outside [0.5, 2.0], so mask should be 0.0
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[0.0]], device=device)
    torch.testing.assert_close(sequence_mask, expected, rtol=1e-3, atol=1e-4)
    # One sequence masked, over high cap
    assert metrics["product_sequence_mask_masked_ratio"] == 1.0
    assert metrics["product_sequence_mask_over_high_ratio"] == 1.0
    assert metrics["product_sequence_mask_under_low_ratio"] == 0.0


def test_compute_outlier_token_mask_masks_by_token_bounds():
    """Tests outlier token mask rejects when a token ratio is out of bounds."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, 5.0] -> token ratios = [1.0, 1.0, 148.4]
    # Third token ratio 148.4 > 100.0, so should reject
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -6.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,  # This should cause masking
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Token ratio 148.4 > 100.0, so mask should be 0.0
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[0.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    # One sequence masked, has token over high threshold
    assert metrics["outlier_seq_masked_ratio"] == 1.0
    assert metrics["outlier_seq_over_high_ratio"] == 1.0
    assert metrics["outlier_seq_under_low_ratio"] == 0.0


def test_compute_outlier_token_mask_accepts_in_bounds():
    """Tests outlier token mask accepts when all token ratios are in bounds."""
    device = "cpu"

    # Token log ratios: [0.5, -0.5, 0.0] -> token ratios = [1.65, 0.61, 1.0]
    # All token ratios within [1e-4, 100.0], so should accept
    old_log_probs = torch.tensor([[-1.0, -1.5, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -1.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # All token ratios in bounds, so mask should be 1.0
    # Shape is [batch, 1] for sequence-level mask
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked
    assert metrics["outlier_seq_masked_ratio"] == 0.0
    assert metrics["outlier_seq_over_high_ratio"] == 0.0
    assert metrics["outlier_seq_under_low_ratio"] == 0.0


def test_compute_outlier_token_mask_respects_loss_mask():
    """Tests outlier token mask ignores out-of-bounds tokens that are masked."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, 5.0] -> token ratios = [1.0, 1.0, 148.4]
    # Third token ratio 148.4 > 100.0, but it's masked, so should accept
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -6.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0]], device=device)  # Third token masked

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Third token is masked, so even though ratio is out of bounds, sequence should be accepted
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked (the out-of-bounds token was in a masked position)
    assert metrics["outlier_seq_masked_ratio"] == 0.0


def test_compute_outlier_token_mask_null_thresholds():
    """Tests outlier token mask accepts all tokens when both thresholds are None."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, 5.0] -> token ratios = [1.0, 1.0, 148.4]
    # With both thresholds None, even extreme ratios should be accepted
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -6.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=None,
        outlier_token_is_threshold_high=None,
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Both thresholds None, so all sequences should be accepted regardless of token ratios
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked
    assert metrics["outlier_seq_masked_ratio"] == 0.0
    assert metrics["outlier_seq_over_high_ratio"] == 0.0
    assert metrics["outlier_seq_under_low_ratio"] == 0.0


def test_compute_outlier_token_mask_only_high_threshold():
    """Tests outlier token mask with only high threshold set (low is None)."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, 5.0] -> token ratios = [1.0, 1.0, 148.4]
    # Only high threshold set, so only high ratios should be masked
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -6.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=None,  # No low threshold
        outlier_token_is_threshold_high=100.0,
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Token ratio 148.4 > 100.0, so mask should be 0.0
    expected = torch.tensor([[0.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    assert metrics["outlier_seq_masked_ratio"] == 1.0
    assert metrics["outlier_seq_over_high_ratio"] == 1.0
    assert metrics["outlier_seq_under_low_ratio"] == 0.0  # No low threshold, so always 0


def test_compute_outlier_token_mask_only_low_threshold():
    """Tests outlier token mask with only low threshold set (high is None)."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, -10.0] -> token ratios = [1.0, 1.0, ~4.5e-5]
    # Only low threshold set, so only low ratios should be masked
    old_log_probs = torch.tensor([[-1.0, -1.0, -11.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        outlier_token_is_threshold_low=1e-4,  # Third token ratio ~4.5e-5 < 1e-4
        outlier_token_is_threshold_high=None,  # No high threshold
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Token ratio ~4.5e-5 < 1e-4, so mask should be 0.0
    expected = torch.tensor([[0.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    assert metrics["outlier_seq_masked_ratio"] == 1.0
    assert metrics["outlier_seq_over_high_ratio"] == 0.0  # No high threshold, so always 0
    assert metrics["outlier_seq_under_low_ratio"] == 1.0


def test_compute_off_policy_correction_null_configs():
    """Tests that compute_off_policy_correction returns None tis_ratio when both configs are null."""
    device = "cpu"

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type=None,
        sequence_mask_metric=None,
    )

    tis_ratio, metrics, new_loss_mask = compute_off_policy_correction(
        old_log_probs, rollout_logprobs, loss_mask, config
    )

    # Should return None tis_ratio (early return) and empty metrics
    assert tis_ratio is None
    assert metrics == {}


def test_compute_off_policy_correction_tis_only():
    """Tests compute_off_policy_correction with only TIS enabled."""
    device = "cpu"

    # Token log ratios: [0.5, 0.5, 0.5] -> token ratios = [1.6487, 1.6487, 1.6487]
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.5, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="token",
        token_tis_ratio_clip_high=2.0,
        sequence_mask_metric=None,
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    tis_ratio, metrics, new_loss_mask = compute_off_policy_correction(
        old_log_probs, rollout_logprobs, loss_mask, config
    )

    # Expected tis_ratio: 1.6487 (no capping needed)
    expected_tis_ratio = torch.exp(torch.tensor(0.5))
    torch.testing.assert_close(
        tis_ratio, torch.full_like(old_log_probs, expected_tis_ratio.item()), rtol=1e-3, atol=1e-4
    )
    # Check metrics are populated
    assert "is_ratio_mean" in metrics
    assert "tis_token_clip_high_ratio" in metrics


def test_compute_off_policy_correction_sequence_mask_only():
    """Tests compute_off_policy_correction with only geometric sequence mask enabled."""
    device = "cpu"

    # Token log ratios: [0.0, 0.0, 0.0] -> geometric mean = 1.0
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type=None,
        sequence_mask_metric="geometric",
        geo_mask_high=1.1,
        geo_mask_low=0.9,
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    tis_ratio, metrics, new_loss_mask = compute_off_policy_correction(
        old_log_probs, rollout_logprobs, loss_mask, config
    )

    # Geometric mean = 1.0, within bounds, so loss_mask unchanged
    # tis_ratio is None since tis_ratio_type is None
    assert tis_ratio is None
    torch.testing.assert_close(new_loss_mask, loss_mask, rtol=1e-3, atol=1e-4)
    # Check metrics are populated
    assert "is_ratio_mean" in metrics
    assert "geo_sequence_mask_masked_ratio" in metrics


def test_compute_off_policy_correction_both_enabled():
    """Tests compute_off_policy_correction with both TIS and geometric sequence mask enabled."""
    device = "cpu"

    # Token log ratios: [0.1, 0.1, 0.1] -> token ratios ≈ [1.105, 1.105, 1.105]
    # Geometric mean ≈ 1.105
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.1, -1.1, -1.1]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type="token",
        token_tis_ratio_clip_high=2.0,
        sequence_mask_metric="geometric",
        geo_mask_high=1.2,
        geo_mask_low=0.8,
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    tis_ratio, metrics, new_loss_mask = compute_off_policy_correction(
        old_log_probs, rollout_logprobs, loss_mask, config
    )

    # TIS ratio ≈ 1.105, geometric mean ≈ 1.105 (within bounds, mask=1)
    expected_tis_ratio = torch.exp(torch.tensor(0.1))
    torch.testing.assert_close(
        tis_ratio, torch.full_like(old_log_probs, expected_tis_ratio.item()), rtol=1e-3, atol=1e-4
    )
    # Check metrics from both TIS and sequence mask are populated
    assert "tis_token_clip_high_ratio" in metrics
    assert "geo_sequence_mask_masked_ratio" in metrics


def test_compute_off_policy_correction_sequence_mask_zeros_loss():
    """Tests that sequence mask can zero out the loss_mask entirely."""
    device = "cpu"

    # Token log ratios: [1.0, 1.0, 1.0] -> geometric mean = exp(1.0) ≈ 2.718
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(
        tis_ratio_type=None,
        sequence_mask_metric="geometric",
        geo_mask_high=1.1,
        geo_mask_low=0.9,
        outlier_token_is_threshold_low=1e-4,
        outlier_token_is_threshold_high=100.0,
    )

    tis_ratio, metrics, new_loss_mask = compute_off_policy_correction(
        old_log_probs, rollout_logprobs, loss_mask, config
    )

    # Geometric mean ≈ 2.718, outside [0.9, 1.1], so loss_mask should be zeroed
    expected_mask = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    torch.testing.assert_close(new_loss_mask, expected_mask, rtol=1e-3, atol=1e-4)
    # Check that the sequence mask metrics show sequence mask happened
    assert metrics["geo_sequence_mask_masked_ratio"] == 1.0


def test_ppo_policy_loss_with_off_policy_correction():
    """Integration test for PPO policy loss with rollout correction enabled."""
    device = "cpu"

    advantages = torch.tensor([[1.0, -1.0, 0.5]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    log_probs = torch.tensor([[-1.1, -0.9, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.05, -1.05, -1.05]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        policy_loss_type="regular",
        loss_reduction="token_mean",
        max_seq_len=4,
        off_policy_correction=OffPolicyCorrectionConfig(
            tis_ratio_type="token",
            token_tis_ratio_clip_high=2.0,
            sequence_mask_metric=None,
            outlier_token_is_threshold_low=1e-4,
            outlier_token_is_threshold_high=100.0,
        ),
    )

    loss_fn = PolicyLossRegistry.get("regular")

    # Loss with rollout correction
    loss_with_correction, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
    )

    # Loss without rollout correction
    config_no_correction = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        policy_loss_type="regular",
        loss_reduction="token_mean",
        max_seq_len=4,
        off_policy_correction=OffPolicyCorrectionConfig(
            tis_ratio_type=None,
            sequence_mask_metric=None,
        ),
    )

    loss_without_correction, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config_no_correction,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
    )

    # TIS correction should modify the loss
    assert not torch.allclose(loss_with_correction, loss_without_correction, rtol=1e-3), (
        f"Rollout correction should change the loss: "
        f"with={loss_with_correction:.6f} vs without={loss_without_correction:.6f}"
    )


def test_compute_tis_ratio_invalid_type():
    """Tests that compute_tis_ratio raises error for invalid tis_ratio_type."""
    device = "cpu"

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.5, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(tis_ratio_type="invalid")

    with pytest.raises(ValueError, match="Unknown tis_ratio_type"):
        compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "invalid", config)


def test_compute_sequence_mask_invalid_type():
    """Tests that compute_sequence_mask raises error for invalid sequence_mask_metric."""
    device = "cpu"

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.5, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = OffPolicyCorrectionConfig(sequence_mask_metric="invalid")

    with pytest.raises(ValueError, match="Unknown sequence_mask_metric"):
        compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "invalid", config)
