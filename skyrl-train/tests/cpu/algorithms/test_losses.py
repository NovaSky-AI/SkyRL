"""
Tests for policy loss functions.

uv run --isolated --extra dev -- pytest tests/cpu/algorithms/test_losses.py
"""

import pytest
import torch
from omegaconf import DictConfig

from skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    masked_mean,
    compute_tis_ratio,
    compute_sequence_mask,
    compute_outlier_token_mask,
    compute_off_policy_correction,
)

NULL_OFF_POLICY_CORR = {
    "tis_ratio_type": None,
    "sequence_mask_metric": None,
    "outlier_token_is_threshold_low": 1e-4,
    "outlier_token_is_threshold_high": 100.0,
}


# Adapted a good test from NeMO-RL
def test_policy_loss_dual_clip():
    """Tests dual clipping in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for dual clipping
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "dual_clip",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Create loss function with dual clipping
    loss_fn = PolicyLossRegistry.get("dual_clip")

    # Calculate expected values
    ratio = torch.exp(log_probs - old_log_probs)  # approx [0.5, 1.0, 10.0]
    assert torch.allclose(ratio, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3)

    # Standard PPO clipping
    loss1 = -ratio * advantages  # [0.5, -1.0, -40.0]
    loss2 = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages  # [0.8, -1.0, -4.8]
    max_loss = torch.maximum(loss1, loss2)  # [0.5, -1.0, -40.0]

    # Dual clipping
    loss3 = -advantages * 3.0  # [-3.0, 3.0, 12.0]
    min_loss = torch.min(loss3, max_loss)  # [-3.0, 1.0, 12.0]

    # For negative advantages, use dual clipped loss
    final_loss = torch.where(advantages < 0, min_loss, max_loss)  # [-0.5, 1.0, 12.0]
    assert torch.allclose(final_loss, torch.tensor([[-0.5, 1.0, 12.0]], device=device), rtol=1e-3)
    expected_loss = final_loss.mean()  # -(-12.5/3) = 4.1667

    # Calculate actual loss
    actual_loss, _ = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages, config=config)

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(4.1667, abs=1e-4)


def test_policy_loss_cispo():
    """Tests CISPO in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for cispo
    config = DictConfig(
        {
            "cispo": {"cispo_eps_clip_low": 0.2, "cispo_eps_clip_high": 0.2},
            "policy_loss_type": "cispo",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Create loss function with cispo
    loss_fn = PolicyLossRegistry.get("cispo")

    # Calculate expected values
    ratio = torch.exp(log_probs - old_log_probs)  # approx [0.5, 1.0, 10.0]
    assert torch.allclose(ratio, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3)

    # Hand-calculation for expected loss:
    # ratio = [0.5, 1.0, 10.0]
    # clamped_ratio = ratio.clamp(0.8, 1.2) = [0.8, 1.0, 1.2]
    # advantages = [1.0, -1.0, -4.0]
    # log_probs = [-1.69315, -1.0, -0.69741]
    # loss_per_token = -advantages * clamped_ratio * log_probs
    # loss_per_token[0] = -(1.0 * 0.8 * -1.69315) = 1.35452
    # loss_per_token[1] = -(-1.0 * 1.0 * -1.0) = -1.0
    # loss_per_token[2] = -(-4.0 * 1.2 * -0.69741) = -3.347568
    # mean(loss) = (1.35452 - 1.0 - 3.347568) / 3 = -0.99768266666
    loss = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages * log_probs
    expected_loss = loss.mean()

    # Calculate actual loss
    actual_loss, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
    )

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(-0.99768266666, abs=1e-4)


def test_policy_loss_reduction_modes():
    """Tests different loss_reduction modes in PolicyLoss function.

    Note: token_mean and sequence_mean give the same result when all sequences
    have the same length and no mask is applied, but differ when masking creates
    different effective sequence lengths.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    advantages = torch.tensor(
        [
            [2.0, 2.0, 2.0],  # sequence 1: consistently higher advantages
            [1.0, 1.0, 1.0],  # sequence 2: consistently lower advantages
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor(
        [[-1.5, -0.5, -1.2], [-0.8, -1.3, -0.9]],  # ratios ≈ [[0.61, 1.65, 0.83],[1.22, 0.74, 1.11]]
        device=device,
    )

    # Create masks to test sequences with different numbers of valid tokens
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    # Test token_mean without mask
    loss_token_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)

    # Test token_mean with mask
    loss_token_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)

    # Test sequence_mean without mask
    loss_seq_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # Test sequence_mean with mask
    loss_seq_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Manual calculations to verify (using default PolicyLoss parameters)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages  # clip_eps_low=0.2, clip_eps_high=0.2
    loss_per_token = -torch.min(surr1, surr2)

    # Expected token_mean without mask: mean of all tokens
    expected_token_no_mask = loss_per_token.mean()

    # Expected token_mean with mask: masked mean of all tokens
    expected_token_with_mask = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Expected sequence_mean without mask: mean of sequence means
    expected_seq_no_mask = loss_per_token.mean(dim=1).mean()

    # Expected sequence_mean with mask: mean of masked sequence means
    seq_means_masked = (loss_per_token * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1) + 1e-8)
    expected_seq_with_mask = seq_means_masked.mean()

    # Verify results
    torch.testing.assert_close(loss_token_no_mask, expected_token_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_token_with_mask, expected_token_with_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_no_mask, expected_seq_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_with_mask, expected_seq_with_mask, rtol=1e-5, atol=1e-8)

    # Verify that the two reduction modes give the same results when sequences have equal length and no mask
    assert torch.allclose(
        loss_token_no_mask, loss_seq_no_mask, rtol=1e-5
    ), "token_mean and sequence_mean should give same results when sequences have equal length and no mask"
    # But they should give different results when mask creates different effective sequence lengths
    assert not torch.allclose(
        loss_token_with_mask, loss_seq_with_mask, rtol=1e-3
    ), "token_mean and sequence_mean with mask should give different results"


def test_policy_loss_reduction_edge_cases():
    """Tests edge cases for loss_reduction modes."""

    device = "cpu"

    # Test with single sequence (should give same result for both modes)
    advantages = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    log_probs = torch.tensor([[-1.5, -0.5, -1.2]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    loss_token, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)
    loss_seq, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # With single sequence, both modes should give same result
    torch.testing.assert_close(loss_token, loss_seq, rtol=1e-6, atol=1e-8)

    # Test with completely masked sequence
    loss_mask = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    loss_token_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)
    loss_seq_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Should handle zero mask gracefully (due to +1e-8 in denominator)
    assert torch.isfinite(loss_token_masked)
    assert torch.isfinite(loss_seq_masked)


def test_gspo_importance_sampling_levels():
    """Tests GSPO policy loss function with sequence-level importance sampling.

    This test focuses on GSPO's key benefit: stabilizing clipping behavior through sequence-level
    importance sampling, which should lead to more consistent training dynamics compared to
    token-level importance sampling in standard PPO.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    # Create test data with varied sequence lengths and extreme ratios to test clipping stability
    # GSPO's benefit is most apparent with sequences of different lengths and high variance
    advantages = torch.tensor(
        [
            [1.5, 2.0, 1.0, 0.8, 0.5, 0.0, 0.0, 0.0],  # long sequence: 5 valid tokens
            [3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # short sequence: 2 valid tokens
            [0.5, 0.8, 1.2, 2.5, 0.0, 0.0, 0.0, 0.0],  # medium sequence: 4 valid tokens
        ],
        device=device,
    )

    old_log_probs = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        device=device,
    )

    # Create extreme log probability ratios to trigger significant clipping
    # This tests GSPO's stability benefits under conditions that would cause unstable clipping
    log_probs = torch.tensor(
        [
            [0.2, -2.5, -0.3, 0.1, -1.8, -1.0, -1.0, -1.0],  # high variance within sequence
            [0.8, -0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # extreme ratios (exp(1.8)≈6.0, exp(0.8)≈2.2)
            [-0.5, 0.3, -1.7, 0.4, -1.0, -1.0, -1.0, -1.0],  # mixed extreme values
        ],
        device=device,
    )

    # Create masks for different sequence lengths (key for testing length normalization)
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 5 tokens
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 tokens
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 4 tokens
        ],
        device=device,
    )

    # Test standard PPO (token-level importance sampling)
    ppo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )
    ppo_loss_fn = PolicyLossRegistry.get("regular")
    loss_token, _ = ppo_loss_fn(log_probs, old_log_probs, advantages, ppo_config, loss_mask)

    # Test GSPO (sequence-level importance sampling)
    gspo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "gspo",
            "loss_reduction": "sequence_mean",  # GSPO recommended reduction
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )
    gspo_loss_fn = PolicyLossRegistry.get("gspo")
    loss_sequence, _ = gspo_loss_fn(log_probs, old_log_probs, advantages, gspo_config, loss_mask)

    # Manual calculation for token-level (standard PPO)
    log_ratio = log_probs - old_log_probs
    ratio_token = log_ratio.exp()
    surr1_token = ratio_token * advantages
    surr2_token = ratio_token.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_token = -torch.min(surr1_token, surr2_token)
    expected_token = (loss_per_token_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Calculate token-level clipping ratio
    is_clipped_token = (-surr2_token > -surr1_token) & (loss_mask.bool())
    clip_ratio_token = is_clipped_token.float().sum() / loss_mask.sum()

    # Manual calculation for sequence-level (GSPO)
    # First compute sequence-level importance weights (key GSPO innovation)
    log_importance_weights_seq = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # GSPO uses stop gradients: s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    ratio_sequence = torch.exp(log_importance_weights_seq.detach() + log_probs - log_probs.detach())
    surr1_sequence = ratio_sequence * advantages
    surr2_sequence = ratio_sequence.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_sequence = -torch.min(surr1_sequence, surr2_sequence)
    # GSPO uses sequence_mean reduction
    expected_sequence = masked_mean(loss_per_token_sequence, loss_mask, dim=-1).mean()

    # Calculate sequence-level clipping ratio
    is_clipped_sequence = (-surr2_sequence > -surr1_sequence) & (loss_mask.bool())
    clip_ratio_sequence = is_clipped_sequence.float().sum() / loss_mask.sum()

    # Verify loss calculations
    torch.testing.assert_close(loss_token, expected_token, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_sequence, expected_sequence, rtol=1e-5, atol=1e-8)

    # Core GSPO benefit test: Different clipping behavior
    # GSPO should produce different clipping patterns due to sequence-level importance sampling
    assert not torch.allclose(
        clip_ratio_token, clip_ratio_sequence, rtol=1e-2
    ), f"Clipping ratios should differ: token={clip_ratio_token:.4f} vs sequence={clip_ratio_sequence:.4f}"

    # Test stability: sequence-level should smooth out extreme per-token variations
    # Check that sequence-level ratios have lower variance within each sequence
    token_ratio_variance = torch.var(ratio_token * loss_mask, dim=-1).mean()
    sequence_ratio_variance = torch.var(ratio_sequence * loss_mask, dim=-1).mean()

    # The key insight: GSPO should reduce within-sequence variance by using sequence-averaged ratios
    assert sequence_ratio_variance < token_ratio_variance, (
        f"GSPO should reduce ratio variance: sequence={sequence_ratio_variance:.4f} < "
        f"token={token_ratio_variance:.4f}"
    )

    # Token-level and sequence-level should give different results due to different importance weighting
    assert not torch.allclose(
        loss_token, loss_sequence, rtol=1e-3
    ), f"Loss values should differ: token={loss_token:.6f} vs sequence={loss_sequence:.6f}"

    # Test length normalization effect: sequences with different lengths should be handled more uniformly
    # This is a key stability benefit of GSPO mentioned in the paper
    seq_lengths = loss_mask.sum(dim=-1)  # [5, 2, 4]

    # In GSPO, the sequence-level importance weights should be the same across all tokens in a sequence
    # This should make the treatment more uniform across different sequence lengths
    for seq_idx in range(log_importance_weights_seq.shape[0]):
        seq_len = int(seq_lengths[seq_idx])
        if seq_len > 1:
            # All importance weights within a sequence should be identical (GSPO property)
            seq_weights = log_importance_weights_seq[seq_idx, :seq_len]
            assert torch.allclose(
                seq_weights, seq_weights[0], rtol=1e-6
            ), f"GSPO should have uniform importance weights within sequence {seq_idx}"


def test_clip_cov_policy_loss():
    """Tests Clip-Cov policy loss function with covariance-based correction."""

    device = "cpu"
    torch.manual_seed(42)  # For reproducible randomization in clip-cov

    # Create test data
    advantages = torch.tensor(
        [
            [2.0, -1.0, 1.5, 0.8],
            [1.0, 0.5, -2.0, 1.2],
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor([[-0.5, -1.5, -0.8, -1.2], [-1.3, -0.7, -1.8, -0.9]], device=device)

    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], device=device)  # Last token masked

    # Create Clip-Cov config
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "clip_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "clip_cov": {"clip_ratio": 0.5, "clip_cov_lb": -5.0, "clip_cov_ub": 5.0},  # Large ratio for testing
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Get loss function
    clip_cov_fn = PolicyLossRegistry.get("clip_cov")

    # Calculate loss
    loss, loss_metrics = clip_cov_fn(log_probs, old_log_probs, advantages, config, loss_mask)
    clip_ratio = loss_metrics["clip_ratio"]

    # Basic sanity checks
    assert torch.isfinite(loss), "Loss should be finite"
    assert 0 <= clip_ratio <= 1, f"Clip ratio should be between 0 and 1, got {clip_ratio}"

    # Compare with regular PPO (should be different due to covariance correction)
    regular_config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    regular_fn = PolicyLossRegistry.get("regular")
    regular_loss, regular_loss_metrics = regular_fn(log_probs, old_log_probs, advantages, regular_config, loss_mask)

    # Clip-Cov should give different results due to covariance-based correction
    assert not torch.allclose(
        loss, regular_loss, rtol=1e-3
    ), f"Clip-Cov and regular PPO should differ: clip_cov={loss:.6f} vs regular={regular_loss:.6f}"


def test_kl_cov_policy_loss():
    """Tests KL-Cov policy loss function with covariance-based token selection."""

    device = "cpu"
    torch.manual_seed(42)  # For reproducible token selection

    # Create test data
    advantages = torch.tensor(
        [
            [1.5, -0.5, 2.0, 0.8],
            [0.5, 1.0, -1.5, 1.2],
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor([[-0.8, -1.2, -0.6, -1.1], [-1.1, -0.9, -1.4, -0.7]], device=device)

    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], device=device)  # Last token masked

    # Create KL-Cov config
    config = DictConfig(
        {
            "policy_loss_type": "kl_cov",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "kl_cov": {"kl_cov_frac": 0.5, "ppo_kl_coef": 1.0},  # Apply KL to 50% of tokens
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    # Get loss function
    kl_cov_fn = PolicyLossRegistry.get("kl_cov")

    # Calculate loss
    loss, loss_metrics = kl_cov_fn(log_probs, old_log_probs, advantages, config, loss_mask)

    # Basic sanity checks
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss_metrics["clip_ratio"] == 0.0, "KL-Cov should return 0.0 for clip_ratio value"

    # Compare with regular PPO (should be different due to KL regularization)
    regular_config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    regular_fn = PolicyLossRegistry.get("regular")
    regular_loss, _ = regular_fn(log_probs, old_log_probs, advantages, regular_config, loss_mask)

    # KL-Cov should give different results due to KL regularization on selected tokens
    assert not torch.allclose(
        loss, regular_loss, rtol=1e-3
    ), f"KL-Cov and regular PPO should differ: kl_cov={loss:.6f} vs regular={regular_loss:.6f}"


def test_sapo_policy_loss_basic():
    """Tests SAPO policy loss against a hand-computed expectation."""

    device = "cpu"

    # Mix of positive and negative advantages so tau_pos / tau_neg both get used
    advantages = torch.tensor([[1.0, -1.0, 0.5]], device=device)

    # Simple log-prob configuration to produce non-trivial ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    # Ratios ≈ [exp(-0.5), exp(0.2), exp(-0.1)] ≈ [0.6065, 1.2214, 0.9048]
    log_probs = torch.tensor([[-1.5, -0.8, -1.1]], device=device)

    # SAPO config: uses sequence_mean reduction and distinct tau_pos / tau_neg
    config = DictConfig(
        {
            "policy_loss_type": "sapo",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "sapo": {"tau_pos": 1.0, "tau_neg": 2.0},
            "off_policy_correction": NULL_OFF_POLICY_CORR,
        }
    )

    loss_fn = PolicyLossRegistry.get("sapo")

    # Actual SAPO loss
    actual_loss, loss_metrics = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
    )

    # --- Hand-computed expectation, mirroring sapo_policy_loss implementation ---

    tau_pos = torch.as_tensor(config.sapo.tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(config.sapo.tau_neg, dtype=advantages.dtype, device=advantages.device)

    def gate_function(x, tau):
        return torch.sigmoid(tau * (x - 1.0)) * (4.0 / tau)

    log_ratio = log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    taus = torch.where(advantages > 0, tau_pos, tau_neg)
    gates = gate_function(ratio, taus)

    loss_per_token = -gates * advantages
    # sequence_mean reduction: per-sequence token mean, then batch mean
    expected_loss = loss_per_token.mean(dim=-1).mean()

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-8)

    # SAPO should always report clip_ratio = 0.0
    assert loss_metrics["clip_ratio"] == 0.0


# ============================================================================
# Rollout Correction Tests
# ============================================================================


def test_compute_tis_ratio_token_level():
    """Tests token-level TIS ratio computation with capping."""
    device = "cpu"

    # old_log_probs - rollout_logprobs gives the log importance ratio
    # Token ratios: exp([0.5, -0.5, 1.0]) = [1.6487, 0.6065, 2.7183]
    old_log_probs = torch.tensor([[-1.0, -1.5, -0.5]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = DictConfig(
        {
            "tis_ratio_type": "token",
            "token_tis_ratio_clip_high": 2.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": "sequence",
            "sequence_tis_ratio_clip_high": 5.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": "sequence",
            "sequence_tis_ratio_clip_high": 5.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": "sequence",
            "sequence_tis_ratio_clip_high": 10.0,
        }
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

    config = DictConfig(
        {
            "sequence_mask_metric": "geometric",
            "geo_mask_high": 1.1,
            "geo_mask_low": 0.9,
        }
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

    config = DictConfig(
        {
            "sequence_mask_metric": "geometric",
            "geo_mask_high": 1.1,
            "geo_mask_low": 0.9,
        }
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


def test_compute_sequence_mask_sequence():
    """Tests sequence sequence mask computation."""
    device = "cpu"

    # Token log ratios: [0.2, 0.1, 0.0] -> sum = 0.3, seq ratio = exp(0.3) ≈ 1.35
    old_log_probs = torch.tensor([[-1.0, -1.1, -1.2]], device=device)
    rollout_logprobs = torch.tensor([[-1.2, -1.2, -1.2]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = DictConfig(
        {
            "sequence_mask_metric": "product",
            "product_mask_high": 2.0,
            "product_mask_low": 0.5,
        }
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


def test_compute_sequence_mask_sequence_rejects_by_seq_ratio():
    """Tests product sequence mask rejects when sequence ratio is out of bounds."""
    device = "cpu"

    # Token log ratios: [1.0, 1.0, 1.0] -> sum = 3.0, seq ratio = exp(3.0) ≈ 20.09
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = DictConfig(
        {
            "sequence_mask_metric": "product",
            "product_mask_high": 2.0,
            "product_mask_low": 0.5,
        }
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

    config = DictConfig(
        {
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,  # This should cause masking
        }
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

    config = DictConfig(
        {
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
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

    config = DictConfig(
        {
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
    )

    outlier_mask, metrics = compute_outlier_token_mask(old_log_probs, rollout_logprobs, loss_mask, config)

    # Third token is masked, so even though ratio is out of bounds, sequence should be accepted
    expected = torch.tensor([[1.0]], device=device)
    torch.testing.assert_close(outlier_mask, expected, rtol=1e-3, atol=1e-4)
    # No sequence was masked (the out-of-bounds token was in a masked position)
    assert metrics["outlier_seq_masked_ratio"] == 0.0


def test_compute_off_policy_correction_null_configs():
    """Tests that compute_off_policy_correction returns None tis_ratio when both configs are null."""
    device = "cpu"

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-2.0, -2.0, -2.0]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = DictConfig(
        {
            "tis_ratio_type": None,
            "sequence_mask_metric": None,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": "token",
            "token_tis_ratio_clip_high": 2.0,
            "sequence_mask_metric": None,
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": None,
            "sequence_mask_metric": "geometric",
            "geo_mask_high": 1.1,
            "geo_mask_low": 0.9,
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": "token",
            "token_tis_ratio_clip_high": 2.0,
            "sequence_mask_metric": "geometric",
            "geo_mask_high": 1.2,
            "geo_mask_low": 0.8,
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
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

    config = DictConfig(
        {
            "tis_ratio_type": None,
            "sequence_mask_metric": "geometric",
            "geo_mask_high": 1.1,
            "geo_mask_low": 0.9,
            "outlier_token_is_threshold_low": 1e-4,
            "outlier_token_is_threshold_high": 100.0,
        }
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

    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": {
                "tis_ratio_type": "token",
                "token_tis_ratio_clip_high": 2.0,
                "sequence_mask_metric": None,
                "outlier_token_is_threshold_low": 1e-4,
                "outlier_token_is_threshold_high": 100.0,
            },
        }
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
    config_no_correction = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "off_policy_correction": {
                "tis_ratio_type": None,
                "sequence_mask_metric": None,
            },
        }
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

    config = DictConfig({"tis_ratio_type": "invalid"})

    with pytest.raises(ValueError, match="Unknown tis_ratio_type"):
        compute_tis_ratio(old_log_probs, rollout_logprobs, loss_mask, "invalid", config)


def test_compute_sequence_mask_invalid_type():
    """Tests that compute_sequence_mask raises error for invalid sequence_mask_metric."""
    device = "cpu"

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    rollout_logprobs = torch.tensor([[-1.5, -1.5, -1.5]], device=device)
    loss_mask = torch.tensor([[1.0, 1.0, 1.0]], device=device)

    config = DictConfig({"sequence_mask_metric": "invalid"})

    with pytest.raises(ValueError, match="Unknown sequence_mask_metric"):
        compute_sequence_mask(old_log_probs, rollout_logprobs, loss_mask, "invalid", config)
