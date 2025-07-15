import pytest
import torch
from skyrl_train.workers.worker import PolicyLoss


# Adapted a good test from NeMO-RL
def test_policy_loss_dual_clip():
    """Tests dual clipping in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create loss function with dual clipping
    loss_fn = PolicyLoss(clip_eps_low=0.2, clip_eps_high=0.2, clip_ratio_c=3.0, loss_type="dual_clip")

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
    actual_loss, _ = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages)

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(4.1667, abs=1e-4)


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

    # Test token_mean without mask
    loss_fn_token = PolicyLoss(
        loss_type="regular", loss_reduction="token_mean", clip_eps_low=clip_eps_low, clip_eps_high=clip_eps_high
    )
    loss_token_no_mask, _ = loss_fn_token(log_probs, old_log_probs, advantages)

    # Test token_mean with mask
    loss_token_with_mask, _ = loss_fn_token(log_probs, old_log_probs, advantages, loss_mask)

    # Test sequence_mean without mask
    loss_fn_seq = PolicyLoss(
        loss_type="regular", loss_reduction="sequence_mean", clip_eps_low=clip_eps_low, clip_eps_high=clip_eps_high
    )
    loss_seq_no_mask, _ = loss_fn_seq(log_probs, old_log_probs, advantages)

    # Test sequence_mean with mask
    loss_seq_with_mask, _ = loss_fn_seq(log_probs, old_log_probs, advantages, loss_mask)

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

    loss_fn_token = PolicyLoss(loss_type="regular", loss_reduction="token_mean")
    loss_fn_seq = PolicyLoss(loss_type="regular", loss_reduction="sequence_mean")

    loss_token, _ = loss_fn_token(log_probs, old_log_probs, advantages)
    loss_seq, _ = loss_fn_seq(log_probs, old_log_probs, advantages)

    # With single sequence, both modes should give same result
    torch.testing.assert_close(loss_token, loss_seq, rtol=1e-6, atol=1e-8)

    # Test with completely masked sequence
    loss_mask = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    loss_token_masked, _ = loss_fn_token(log_probs, old_log_probs, advantages, loss_mask)
    loss_seq_masked, _ = loss_fn_seq(log_probs, old_log_probs, advantages, loss_mask)

    # Should handle zero mask gracefully (due to +1e-8 in denominator)
    assert torch.isfinite(loss_token_masked)
    assert torch.isfinite(loss_seq_masked)
