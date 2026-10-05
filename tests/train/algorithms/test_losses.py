"""
Tests for policy loss functions.

uv run --isolated --extra dev -- pytest tests/train/algorithms/test_losses.py
"""

import threading

import pytest
import torch

from skyrl.backends.skyrl_train.utils import ppo_utils
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossRegistry,
    compute_trajectory_log_importance_weights,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.train.config import (
    AlgorithmConfig,
    CISPOConfig,
    ClipCovConfig,
    DPPOConfig,
    KLCovConfig,
    OffPolicyCorrectionConfig,
    SAPOConfig,
)
from tests.train.util import ThreadedAllReduce

NULL_OFF_POLICY_CORR = OffPolicyCorrectionConfig(
    tis_ratio_type=None,
    sequence_mask_metric=None,
    outlier_token_is_threshold_low=None,
    outlier_token_is_threshold_high=None,
)


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
    config = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        clip_ratio_c=3.0,
        policy_loss_type="dual_clip",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
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
    expected_loss = final_loss.sum()

    # Calculate actual loss
    actual_loss, _ = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages, config=config)

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(12.5, abs=1e-4)


def test_policy_loss_cispo():
    """Tests CISPO in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for cispo
    config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_eps_clip_low=0.2, cispo_eps_clip_high=0.2),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
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
    # sum(loss) = (1.35452 - 1.0 - 3.347568) = -2.9930
    loss = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages * log_probs
    expected_loss = loss.sum()

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
    assert actual_loss.item() == pytest.approx(-2.9930, abs=1e-4)


def test_policy_loss_cispo_rollout_anchor():
    """CISPO with cispo_anchor='rollout' anchors the IS ratio on the rollout log-probs."""

    device = "cpu"

    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)
    # Distinct rollout log-probs so the rollout-anchored ratio differs from the old-anchored one.
    rollout_logprobs = torch.tensor([[-1.30685, -1.5, -1.0]], device=device)

    config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_eps_clip_low=0.2, cispo_eps_clip_high=0.2, cispo_anchor="rollout"),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    loss_fn = PolicyLossRegistry.get("cispo")

    # ratio is anchored on rollout_logprobs (NOT old_log_probs)
    ratio = torch.exp(log_probs - rollout_logprobs)
    expected_loss = (-ratio.clamp(1 - 0.2, 1 + 0.2) * advantages * log_probs).sum()

    actual_loss, metrics = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
        rollout_logprobs=rollout_logprobs,
    )
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)

    # IS-ratio diagnostics are logged and reflect the rollout-anchored ratio
    for k in ("clip_ratio", "cispo/ratio_mean", "cispo/ratio_clamped_mean", "cispo/ratio_max", "cispo/ratio_min"):
        assert k in metrics, f"missing metric {k}"
    assert metrics["cispo/ratio_mean"] == pytest.approx(ratio.mean().item(), rel=1e-3)
    expected_clamped = ratio.clamp(1 - 0.2, 1 + 0.2)
    assert metrics["cispo/ratio_clamped_mean"] == pytest.approx(expected_clamped.mean().item(), rel=1e-3)
    assert metrics["cispo/ratio_max"] == pytest.approx(ratio.max().item(), rel=1e-3)

    # rollout anchor genuinely differs from the default old anchor on the same inputs
    old_config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_eps_clip_low=0.2, cispo_eps_clip_high=0.2, cispo_anchor="old"),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    old_loss, _ = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=old_config,
        rollout_logprobs=rollout_logprobs,
    )
    assert not torch.allclose(actual_loss, old_loss)


def test_policy_loss_cispo_ratio_min_max_ignore_masked_tokens():
    """cispo/ratio_{min,max} must be computed over active tokens only.

    Regression test: `ratio` is strictly positive (safe_exp_delta), so computing the min over
    `ratio * loss_mask` would zero the masked positions and report cispo/ratio_min == 0.0 whenever
    any token is masked, instead of the true minimum over the unmasked ratios.
    """
    device = "cpu"

    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)
    # Mask out the middle token, whose ratio (== 1.0) is neither the min nor the max of the row.
    loss_mask = torch.tensor([[1.0, 0.0, 1.0]], device=device)

    config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_eps_clip_low=0.2, cispo_eps_clip_high=0.2, cispo_anchor="old"),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    loss_fn = PolicyLossRegistry.get("cispo")

    ratio = torch.exp(log_probs - old_log_probs)
    active_ratio = ratio[loss_mask.bool()]

    _, metrics = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
        loss_mask=loss_mask,
    )

    # min/max reflect only the unmasked ratios -- not 0.0 from the zeroed masked positions.
    assert metrics["cispo/ratio_min"] > 0.0
    assert metrics["cispo/ratio_min"] == pytest.approx(active_ratio.min().item(), rel=1e-3)
    assert metrics["cispo/ratio_max"] == pytest.approx(active_ratio.max().item(), rel=1e-3)


def test_policy_loss_cispo_rollout_anchor_requires_rollout_logprobs():
    """cispo_anchor='rollout' must be given rollout_logprobs."""
    config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_anchor="rollout"),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    loss_fn = PolicyLossRegistry.get("cispo")
    with pytest.raises(AssertionError):
        loss_fn(
            log_probs=torch.tensor([[-1.0]]),
            old_log_probs=torch.tensor([[-1.0]]),
            advantages=torch.tensor([[1.0]]),
            config=config,
            rollout_logprobs=None,
        )


def test_policy_loss_cispo_rollout_anchor_rejects_tis():
    """cispo_anchor='rollout' refuses to stack TIS (would double-count the off-policy gap)."""
    config = AlgorithmConfig(
        cispo=CISPOConfig(cispo_anchor="rollout"),
        policy_loss_type="cispo",
        max_seq_len=4,
        off_policy_correction=OffPolicyCorrectionConfig(tis_ratio_type="token"),
    )
    loss_fn = PolicyLossRegistry.get("cispo")
    with pytest.raises(ValueError, match="double-count"):
        loss_fn(
            log_probs=torch.tensor([[-1.0]]),
            old_log_probs=torch.tensor([[-1.0]]),
            advantages=torch.tensor([[1.0]]),
            config=config,
            rollout_logprobs=torch.tensor([[-1.2]]),
        )


def test_cispo_anchor_validation():
    """CISPOConfig rejects an invalid anchor."""
    with pytest.raises(ValueError, match="cispo_anchor"):
        CISPOConfig(cispo_anchor="bogus")


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
    ppo_config = AlgorithmConfig(
        eps_clip_low=clip_eps_low,
        eps_clip_high=clip_eps_high,
        clip_ratio_c=3.0,
        policy_loss_type="regular",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    ppo_loss_fn = PolicyLossRegistry.get("regular")
    loss_token, _ = ppo_loss_fn(log_probs, old_log_probs, advantages, ppo_config, loss_mask)

    # Test GSPO (sequence-level importance sampling)
    gspo_config = AlgorithmConfig(
        eps_clip_low=clip_eps_low,
        eps_clip_high=clip_eps_high,
        clip_ratio_c=3.0,
        policy_loss_type="gspo",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )
    gspo_loss_fn = PolicyLossRegistry.get("gspo")
    loss_sequence, _ = gspo_loss_fn(log_probs, old_log_probs, advantages, gspo_config, loss_mask)

    # Manual calculation for token-level (standard PPO)
    log_ratio = log_probs - old_log_probs
    ratio_token = log_ratio.exp()
    surr1_token = ratio_token * advantages
    surr2_token = ratio_token.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_token = -torch.min(surr1_token, surr2_token)
    expected_token = (loss_per_token_token * loss_mask).sum()

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
    # GSPO uses sum reduction
    expected_sequence = loss_per_token_sequence.sum()

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


# Sequence ratios 0.93, 0.61, 1.69, 1.49 and 0.68 with advantages +, -, +, -, +: rows 1 and 2 clip,
# rows 0, 3 and 4 do not.
GSPO_GOLDEN_LOG_PROBS = [
    [-0.9, -1.4, -0.2, -2.1, -1.0],
    [-1.1, -1.5, -1.9, -0.9, -1.5],
    [-1.6, 0.0, -0.4, -0.6, -0.2],
    [-0.7, -0.3, -1.0, -0.45, -0.65],
    [-0.9, -1.0, -1.1, -1.15, -0.8],
]
GSPO_GOLDEN_OLD_LOG_PROBS = [
    [-1.0, -1.0, -0.5, -1.8, -1.0],
    [-0.6, -0.9, -1.5, -0.4, -1.0],
    [-1.2, -0.5, -0.9, -1.0, -0.9],
    [-1.1, -0.8, -1.3, -0.9, -1.0],
    [-0.5, -0.7, -0.6, -0.8, -1.0],
]
GSPO_GOLDEN_ROLLOUT_LOGPROBS = [
    [-1.1, -0.9, -0.6, -1.7, -1.0],
    [-0.6, -1.0, -1.4, -0.4, -1.1],
    [-1.3, -0.6, -0.8, -1.2, -0.8],
    [-1.0, -0.9, -1.2, -0.8, -1.1],
    [-0.6, -0.6, -0.7, -0.9, -1.0],
]
GSPO_GOLDEN_ADVANTAGES = [
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [-0.5, -0.5, -0.5, -0.5, -0.5],
    [0.0, 2.0, 2.0, 2.0, 2.0],
    [-1.0, -1.0, -1.0, -1.0, -1.0],
    [0.8, 0.8, 0.8, 0.8, 0.0],
]
GSPO_GOLDEN_LOSS_MASK = [
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
]


@pytest.mark.parametrize(
    "off_policy_correction, loss_reduction, expected_loss, expected_metrics, expected_grad",
    [
        (
            OffPolicyCorrectionConfig(),
            "sequence_mean",
            -6.663854598999023,
            {"clip_ratio": 0.40909090638160706},
            [
                [-0.9277434945106506] * 4 + [0.0],
                [0.0] * 5,
                [0.0] * 5,
                [1.491824746131897] * 5,
                [-0.5430013537406921] * 4 + [0.0],
            ],
        ),
        (
            OffPolicyCorrectionConfig(tis_ratio_type="token", token_tis_ratio_clip_high=2.0),
            "token_mean",
            -7.2169060707092285,
            {
                "clip_ratio": 0.40909090638160706,
                "is_ratio_mean": 1.0189385414123535,
                "is_ratio_std": 0.35202884674072266,
                "is_ratio_max": 1.2214027643203735,
                "is_ratio_min": 0.0,
                "tis_token_clip_high_ratio": 0.0,
            },
            [
                [-1.0253151655197144, -0.8394569754600525, -1.0253151655197144, -0.839457094669342, 0.0],
                [0.0] * 5,
                [0.0] * 5,
                [1.3498587608337402, 1.6487212181091309, 1.3498589992523193, 1.3498588800430298, 1.6487213373184204],
                [-0.6001092791557312, -0.4913279414176941, -0.6001092195510864, -0.6001092195510864, 0.0],
            ],
        ),
    ],
)
def test_gspo_sequence_level_golden(
    off_policy_correction, loss_reduction, expected_loss, expected_metrics, expected_grad
):
    """Pins the default (sequence-level) GSPO loss, metrics and gradients for fixed inputs."""
    config = AlgorithmConfig(
        policy_loss_type="gspo",
        loss_reduction=loss_reduction,
        eps_clip_low=0.2,
        eps_clip_high=0.28,
        off_policy_correction=off_policy_correction,
    )
    log_probs = torch.tensor(GSPO_GOLDEN_LOG_PROBS, requires_grad=True)

    loss, metrics = PolicyLossRegistry.get("gspo")(
        log_probs,
        torch.tensor(GSPO_GOLDEN_OLD_LOG_PROBS),
        torch.tensor(GSPO_GOLDEN_ADVANTAGES),
        config,
        loss_mask=torch.tensor(GSPO_GOLDEN_LOSS_MASK),
        rollout_logprobs=torch.tensor(GSPO_GOLDEN_ROLLOUT_LOGPROBS),
    )
    loss.backward()

    # A few float32 ULPs of slack, for CPU vector math that differs across machines.
    assert loss.item() == pytest.approx(expected_loss, rel=1e-6)
    assert metrics == pytest.approx(expected_metrics, rel=1e-6, abs=1e-7)
    torch.testing.assert_close(log_probs.grad, torch.tensor(expected_grad), rtol=1e-6, atol=1e-7)


# One row per step: (trajectory index, old log-probs, log-ratios). Per step, trajectory 0 clips at
# step 0 and trajectory 1 at step 0, but neither clips at the trajectory level; trajectory 2 clips
# at both levels and trajectory 3 has a single step.
STEP_WISE_ROWS = [
    (0, [-1.0, -1.2, -0.8], [0.4, 0.6, 0.5]),
    (0, [-0.9, -1.1], [-0.5, -0.3]),
    (0, [-1.3, -0.7, -1.0, -1.05], [0.1, -0.1, 0.05, -0.05]),
    (1, [-0.6, -1.4], [-0.2, -0.4]),
    (1, [-1.0, -0.9, -1.1], [0.0, 0.2, 0.1]),
    (2, [-0.7, -1.2], [0.3, 0.4]),
    (2, [-0.8, -1.0, -0.9], [0.35, 0.25, 0.45]),
    (3, [-0.5, -0.8], [0.05, -0.15]),
]
TRAJECTORY_ADVANTAGES = [1.5, -0.7, 0.9, 0.3]


def _gspo_config() -> AlgorithmConfig:
    return AlgorithmConfig(
        policy_loss_type="gspo",
        loss_reduction="sequence_mean",
        eps_clip_low=0.2,
        eps_clip_high=0.28,
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )


def _right_aligned(rows, width):
    """Returns ``(values, mask)`` with each row's values right-aligned in a zero-padded tensor."""
    values = torch.zeros(len(rows), width)
    mask = torch.zeros(len(rows), width)
    for i, row in enumerate(rows):
        values[i, width - len(row) :] = torch.tensor(row)
        mask[i, width - len(row) :] = 1.0
    return values, mask


def _step_wise_batch():
    """Returns ``(log_probs, old_log_probs, advantages, loss_mask, trajectory_index)``, one row per step."""
    old_log_probs, loss_mask = _right_aligned([old for _, old, _ in STEP_WISE_ROWS], width=4)
    log_ratio, _ = _right_aligned([ratio for _, _, ratio in STEP_WISE_ROWS], width=4)
    trajectory_index = torch.tensor([trajectory for trajectory, _, _ in STEP_WISE_ROWS])
    advantages = torch.tensor(TRAJECTORY_ADVANTAGES)[trajectory_index].unsqueeze(-1) * loss_mask
    return old_log_probs + log_ratio, old_log_probs, advantages, loss_mask, trajectory_index


def _merged_batch():
    """Same tokens as ``_step_wise_batch``, with each trajectory's steps concatenated into one row."""
    olds = [[] for _ in TRAJECTORY_ADVANTAGES]
    ratios = [[] for _ in TRAJECTORY_ADVANTAGES]
    for trajectory, old, ratio in STEP_WISE_ROWS:
        olds[trajectory] += old
        ratios[trajectory] += ratio
    width = max(len(old) for old in olds)
    old_log_probs, loss_mask = _right_aligned(olds, width)
    log_ratio, _ = _right_aligned(ratios, width)
    advantages = torch.tensor(TRAJECTORY_ADVANTAGES).unsqueeze(-1) * loss_mask
    return old_log_probs + log_ratio, old_log_probs, advantages, loss_mask


def test_gspo_trajectory_level_with_single_step_trajectories_matches_sequence_level():
    """With one row per trajectory, trajectory-level weights reproduce sequence-level GSPO exactly."""
    config = _gspo_config()
    loss_fn = PolicyLossRegistry.get("gspo")
    old_log_probs = torch.tensor(GSPO_GOLDEN_OLD_LOG_PROBS)
    advantages = torch.tensor(GSPO_GOLDEN_ADVANTAGES)
    loss_mask = torch.tensor(GSPO_GOLDEN_LOSS_MASK)
    sequence_log_probs = torch.tensor(GSPO_GOLDEN_LOG_PROBS, requires_grad=True)
    trajectory_log_probs = torch.tensor(GSPO_GOLDEN_LOG_PROBS, requires_grad=True)

    weights = compute_trajectory_log_importance_weights(
        trajectory_log_probs.detach(), old_log_probs, loss_mask, trajectory_index=torch.arange(len(loss_mask))
    )
    sequence_loss, sequence_metrics = loss_fn(sequence_log_probs, old_log_probs, advantages, config, loss_mask)
    trajectory_loss, trajectory_metrics = loss_fn(
        trajectory_log_probs,
        old_log_probs,
        advantages,
        config,
        loss_mask,
        trajectory_log_importance_weights=weights,
    )
    sequence_loss.backward()
    trajectory_loss.backward()

    assert torch.equal(trajectory_loss, sequence_loss)
    assert torch.equal(trajectory_log_probs.grad, sequence_log_probs.grad)
    assert trajectory_metrics.pop("gspo_traj_step_log_weight_abs_diff") == 0.0
    assert trajectory_metrics == sequence_metrics


def test_gspo_trajectory_level_matches_sequence_level_on_merged_trajectories(monkeypatch):
    """Trajectory-level GSPO on per-step rows equals sequence-level GSPO on the concatenated steps.

    Compares per-token ratios (via the log weights and the gradients, which are ``-advantage * ratio``
    on unclipped tokens), clip masks, and unreduced per-token losses. The tolerance covers float32
    sums of the same terms taken in a different order.
    """
    monkeypatch.setattr(ppo_utils, "reduce_loss", lambda loss, loss_mask: loss * loss_mask)
    config = _gspo_config()
    loss_fn = PolicyLossRegistry.get("gspo")
    tolerance = dict(rtol=1e-6, atol=1e-6)

    log_probs, old_log_probs, advantages, loss_mask, trajectory_index = _step_wise_batch()
    log_probs.requires_grad_(True)
    weights = compute_trajectory_log_importance_weights(log_probs.detach(), old_log_probs, loss_mask, trajectory_index)
    step_losses, step_metrics = loss_fn(
        log_probs, old_log_probs, advantages, config, loss_mask, trajectory_log_importance_weights=weights
    )
    step_losses.sum().backward()

    merged_log_probs, merged_old_log_probs, merged_advantages, merged_mask = _merged_batch()
    merged_log_probs.requires_grad_(True)
    merged_losses, merged_metrics = loss_fn(
        merged_log_probs, merged_old_log_probs, merged_advantages, config, merged_mask
    )
    merged_losses.sum().backward()

    merged_weights = masked_mean(merged_log_probs.detach() - merged_old_log_probs, merged_mask, dim=-1)
    torch.testing.assert_close(weights, merged_weights[trajectory_index], **tolerance)

    tokens, merged_tokens = loss_mask.bool(), merged_mask.bool()
    torch.testing.assert_close(step_losses.detach()[tokens], merged_losses.detach()[merged_tokens], **tolerance)
    step_grads, merged_grads = log_probs.grad[tokens], merged_log_probs.grad[merged_tokens]
    torch.testing.assert_close(step_grads, merged_grads, **tolerance)
    clipped = step_grads == 0
    assert torch.equal(clipped, merged_grads == 0)
    assert clipped.any() and not clipped.all()
    assert step_metrics["clip_ratio"] == pytest.approx(merged_metrics["clip_ratio"])

    step_weights = masked_mean(log_probs.detach() - old_log_probs, loss_mask, dim=-1)
    assert step_metrics["gspo_traj_step_log_weight_abs_diff"] == pytest.approx(
        (weights - step_weights).abs().mean().item()
    )

    # Sequence-level GSPO on the same per-step rows clips differently.
    sequence_losses, sequence_metrics = loss_fn(log_probs.detach(), old_log_probs, advantages, config, loss_mask)
    assert not torch.allclose(sequence_losses[tokens], merged_losses.detach()[merged_tokens], **tolerance)
    assert sequence_metrics["clip_ratio"] > merged_metrics["clip_ratio"]


def test_trajectory_gspo_is_invariant_to_row_order_and_microbatch_split():
    """Permuting rows or regrouping them into microbatches leaves weights, loss and gradients unchanged."""
    config = _gspo_config()
    loss_fn = PolicyLossRegistry.get("gspo")
    log_probs, old_log_probs, advantages, loss_mask, trajectory_index = _step_wise_batch()

    def run(order, microbatch_sizes):
        order = torch.tensor(order)
        rows_log_probs = log_probs[order].clone().requires_grad_(True)
        weights = compute_trajectory_log_importance_weights(
            rows_log_probs.detach(), old_log_probs[order], loss_mask[order], trajectory_index[order]
        )
        total_loss = sum(
            loss_fn(
                rows_log_probs[rows],
                old_log_probs[order][rows],
                advantages[order][rows],
                config,
                loss_mask[order][rows],
                trajectory_log_importance_weights=weights[rows],
            )[0]
            for rows in torch.arange(len(order)).split(microbatch_sizes)
        )
        total_loss.backward()
        inverse = torch.argsort(order)
        return weights[inverse], total_loss.detach(), rows_log_probs.grad[inverse]

    reference = run(list(range(8)), [8])
    for order, microbatch_sizes in [
        (list(range(8)), [3, 5]),
        ([5, 0, 7, 2, 4, 1, 6, 3], [2, 2, 4]),
        ([7, 6, 5, 4, 3, 2, 1, 0], [1, 6, 1]),
    ]:
        for actual, expected in zip(run(order, microbatch_sizes), reference):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_trajectory_log_importance_weights_with_two_dp_ranks_match_one_rank(monkeypatch):
    """Splitting rows over two DP ranks (trajectory 0 spans both) gives the single-rank weights."""
    log_probs, old_log_probs, _, loss_mask, trajectory_index = _step_wise_batch()
    expected = compute_trajectory_log_importance_weights(log_probs, old_log_probs, loss_mask, trajectory_index)

    monkeypatch.setattr(torch.distributed, "all_reduce", ThreadedAllReduce(world_size=2))
    shards = [slice(0, 2), slice(2, 8)]
    results = [None, None]

    def run_rank(rank):
        rows = shards[rank]
        results[rank] = compute_trajectory_log_importance_weights(
            log_probs[rows], old_log_probs[rows], loss_mask[rows], trajectory_index[rows], dp_group=object()
        )

    threads = [threading.Thread(target=run_rank, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    torch.testing.assert_close(torch.cat(results), expected, rtol=1e-6, atol=1e-7)
    # Without the reduction each rank sees only part of trajectory 0.
    rank0_only = compute_trajectory_log_importance_weights(
        log_probs[:2], old_log_probs[:2], loss_mask[:2], trajectory_index[:2]
    )
    assert not torch.allclose(rank0_only, expected[:2])


def test_trajectory_log_importance_weights_ignore_padding_rows():
    """Fully masked padding rows with their own trajectory indices change neither real weights nor the loss."""
    config = _gspo_config()
    loss_fn = PolicyLossRegistry.get("gspo")
    log_probs, old_log_probs, advantages, loss_mask, trajectory_index = _step_wise_batch()
    num_rows, num_pad = len(trajectory_index), 2

    def pad(tensor):
        # Like `pad_training_input_batch`, padding rows copy row 0.
        return torch.cat([tensor, tensor[[0] * num_pad]])

    padded_mask = torch.cat([loss_mask, torch.zeros(num_pad, loss_mask.shape[1])])
    padded_index = torch.cat([trajectory_index, trajectory_index.max() + 1 + torch.arange(num_pad)])

    weights = compute_trajectory_log_importance_weights(log_probs, old_log_probs, loss_mask, trajectory_index)
    padded_weights = compute_trajectory_log_importance_weights(
        pad(log_probs), pad(old_log_probs), padded_mask, padded_index
    )
    assert torch.equal(padded_weights[:num_rows], weights)
    assert torch.equal(padded_weights[num_rows:], torch.zeros(num_pad))

    loss, metrics = loss_fn(
        log_probs, old_log_probs, advantages, config, loss_mask, trajectory_log_importance_weights=weights
    )
    padded_loss, padded_metrics = loss_fn(
        pad(log_probs),
        pad(old_log_probs),
        pad(advantages),
        config,
        padded_mask,
        trajectory_log_importance_weights=padded_weights,
    )
    torch.testing.assert_close(padded_loss, loss)
    assert padded_metrics == pytest.approx(metrics)


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
    config = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        policy_loss_type="clip_cov",
        max_seq_len=4,
        clip_cov=ClipCovConfig(clip_ratio=0.5, clip_cov_lb=-5.0, clip_cov_ub=5.0),  # Large ratio for testing
        off_policy_correction=NULL_OFF_POLICY_CORR,
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
    regular_config = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        policy_loss_type="regular",
        max_seq_len=4,
        off_policy_correction=NULL_OFF_POLICY_CORR,
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
    config = AlgorithmConfig(
        policy_loss_type="kl_cov",
        max_seq_len=4,
        kl_cov=KLCovConfig(kl_cov_frac=0.5, ppo_kl_coef=1.0),  # Apply KL to 50% of tokens
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )

    # Get loss function
    kl_cov_fn = PolicyLossRegistry.get("kl_cov")

    # Calculate loss
    loss, loss_metrics = kl_cov_fn(log_probs, old_log_probs, advantages, config, loss_mask)

    # Basic sanity checks
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss_metrics["clip_ratio"] == 0.0, "KL-Cov should return 0.0 for clip_ratio value"

    # Compare with regular PPO (should be different due to KL regularization)
    regular_config = AlgorithmConfig(
        eps_clip_low=0.2,
        eps_clip_high=0.2,
        policy_loss_type="regular",
        max_seq_len=4,
        use_tis=False,
        off_policy_correction=NULL_OFF_POLICY_CORR,
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

    # SAPO config with distinct tau_pos / tau_neg
    config = AlgorithmConfig(
        policy_loss_type="sapo",
        max_seq_len=4,
        sapo=SAPOConfig(tau_pos=1.0, tau_neg=2.0),
        off_policy_correction=NULL_OFF_POLICY_CORR,
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
    # sum reduction
    expected_loss = loss_per_token.sum()

    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-8)

    # SAPO should always report clip_ratio = 0.0
    assert loss_metrics["clip_ratio"] == 0.0


@pytest.mark.parametrize(
    "name, dppo_type, delta_low, delta_high, old_lp, new_lp, advs, rollout_lp, expect_mask, expect_clip_gt_zero",
    [
        (
            "tv_pos_adv_masked",
            "binary_tv",
            0.2,
            0.2,
            [[-1.0, -1.0, -1.0]],
            [[-0.3, -2.0, -0.9]],
            [[1.0, -1.0, 1.0]],
            None,
            [[0.0, 0.0, 1.0]],
            True,
        ),
        (
            "tv_wrong_direction_not_masked",
            "binary_tv",
            0.2,
            0.2,
            [[-1.0, -1.0]],
            [[-2.0, -0.3]],
            [[1.0, -1.0]],
            None,
            [[1.0, 1.0]],
            False,
        ),
        (
            "tv_uses_rollout_logprobs",
            "binary_tv",
            0.2,
            0.2,
            [[-1.0, -1.0]],
            [[-0.3, -0.95]],
            [[1.0, 1.0]],
            [[-0.35, -1.0]],
            None,
            None,
        ),
        (
            "kl_masked",
            "binary_kl",
            0.05,
            0.05,
            [[-1.0, -1.0]],
            [[-0.1, -0.95]],
            [[1.0, 1.0]],
            None,
            None,
            True,
        ),
        (
            "kl_no_masking_within_delta",
            "binary_kl",
            0.5,
            0.5,
            [[-1.0, -1.0]],
            [[-1.01, -0.99]],
            [[1.0, -1.0]],
            None,
            [[1.0, 1.0]],
            False,
        ),
    ],
)
def test_dppo_policy_loss(
    name,
    dppo_type,
    delta_low,
    delta_high,
    old_lp,
    new_lp,
    advs,
    rollout_lp,
    expect_mask,
    expect_clip_gt_zero,
):
    device = "cpu"
    old_log_probs = torch.tensor(old_lp, device=device)
    log_probs = torch.tensor(new_lp, device=device)
    advantages = torch.tensor(advs, device=device)
    rollout_logprobs = torch.tensor(rollout_lp, device=device) if rollout_lp is not None else None

    config = AlgorithmConfig(
        policy_loss_type="dppo",
        loss_reduction="token_mean",
        max_seq_len=4,
        dppo=DPPOConfig(dppo_type=dppo_type, delta_low=delta_low, delta_high=delta_high),
        off_policy_correction=NULL_OFF_POLICY_CORR,
    )

    loss_fn = PolicyLossRegistry.get("dppo")

    if name == "tv_uses_rollout_logprobs":
        loss_with, m1 = loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            config=config,
            rollout_logprobs=rollout_logprobs,
        )
        loss_without, m2 = loss_fn(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            config=config,
            rollout_logprobs=None,
        )
        assert not torch.allclose(
            loss_with, loss_without
        ), "rollout_logprobs should change the mask and therefore the loss"
        return

    loss, metrics = loss_fn(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        config=config,
        rollout_logprobs=rollout_logprobs,
    )

    if expect_mask is not None:
        expected_mask = torch.tensor(expect_mask, device=device)
        ratio = torch.exp(log_probs - old_log_probs)
        # reduce_loss sums over (loss * loss_mask); loss_mask is None here so it is a plain sum.
        expected_loss = -(ratio * advantages * expected_mask).sum()
        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)

    if expect_clip_gt_zero is True:
        assert metrics["clip_ratio"] > 0.0, f"{name}: expected some masking"
    elif expect_clip_gt_zero is False:
        assert metrics["clip_ratio"] == pytest.approx(0.0, abs=1e-6), f"{name}: expected no masking"
