"""Tests for the score-centering loss terms and the member weights they share with each loss."""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    DPPOTrustRegion,
    DPPOType,
    RatioBand,
    RatioClamp,
    UnitWeight,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossType,
    compute_policy_loss_cispo,
    dppo_policy_loss,
    reinforce_policy_loss,
    rollout_is_policy_loss,
    score_centering_weight,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    SampleSupportScores,
    ScoreCenteringMetric,
    score_centering_loss_inputs,
    score_centering_metrics,
)
from skyrl.train.config import AlgorithmConfig, CISPOConfig, DPPOConfig

CONFIG = AlgorithmConfig(policy_loss_type="rollout_is", eps_clip_low=0.2, eps_clip_high=0.25)
CISPO_ROLLOUT = AlgorithmConfig(
    policy_loss_type="cispo", cispo=CISPOConfig(cispo_eps_clip_low=1.0, cispo_eps_clip_high=0.1, cispo_anchor="rollout")
)
DPPO = AlgorithmConfig(policy_loss_type="dppo", dppo=DPPOConfig(dppo_type="binary_tv", delta_low=0.05, delta_high=0.1))


def _loss_inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    log_probs = (-torch.rand(2, 4, generator=generator, dtype=torch.float64)).requires_grad_(True)
    rollout_logprobs = log_probs.detach() + 0.3 * torch.randn(2, 4, generator=generator, dtype=torch.float64)
    advantages = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    loss_mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=torch.float64)
    return log_probs, rollout_logprobs, advantages, loss_mask


def test_the_band_keeps_ratios_strictly_inside_it():
    band = RatioBand.from_clip(0.2, 0.25)
    ratios = torch.tensor([0.79, 0.8, 0.81, 1.0, 1.24, 1.25, 1.3], dtype=torch.float64)

    assert band.contains(ratios).tolist() == [False, False, True, True, True, False, False]
    log_ratios = torch.tensor([-1.0, -0.1, 0.0, 0.1, 0.5], dtype=torch.float64)
    expected = torch.where(band.contains(log_ratios.exp()), log_ratios.exp(), 0.0)
    torch.testing.assert_close(band.weights(log_ratios, torch.zeros_like(log_ratios)), expected)


def test_the_clamp_and_unit_weights():
    log_ratios = torch.tensor([-1.0, 0.0, 0.5], dtype=torch.float64)
    zeros = torch.zeros_like(log_ratios)

    torch.testing.assert_close(
        RatioClamp.from_clip(1.0, 0.1).weights(log_ratios, zeros), log_ratios.exp().clamp(0.0, 1.1)
    )
    torch.testing.assert_close(UnitWeight().weights(log_ratios, zeros), torch.ones_like(log_ratios))


def test_the_dppo_trust_region_drops_by_the_advantage_sign():
    region = DPPOTrustRegion(DPPOType.BINARY_TV, delta_low=0.05, delta_high=0.1)
    sampler = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64).log()
    trainer = torch.tensor([0.65, 0.5, 0.4], dtype=torch.float64).log()

    # A positive advantage drops the token pushed 0.15 above the sampler; a negative one, the one 0.1 below.
    assert region.keep(trainer, sampler, negative_advantage=False).tolist() == [False, True, True]
    assert region.keep(trainer, sampler, negative_advantage=True).tolist() == [True, True, False]
    ratio = (trainer - sampler).exp()
    torch.testing.assert_close(region.weights(trainer, sampler), ratio * torch.tensor([0.0, 1.0, 1.0]))
    assert DPPOTrustRegion.depends_on_advantage_sign and not RatioBand.depends_on_advantage_sign


@pytest.mark.parametrize(("dppo_type", "delta"), [("binary_tv", 0.02), ("binary_kl", 0.002)])
def test_dppo_matches_its_reference_mask(dppo_type, delta):
    """The shared trust region reproduces DPPO's per-sign masking (Stable-RL's reference)."""
    log_probs, rollout_logprobs, advantages, loss_mask = _loss_inputs(seed=2)
    config = AlgorithmConfig(
        policy_loss_type="dppo", dppo=DPPOConfig(dppo_type=dppo_type, delta_low=delta, delta_high=delta)
    )

    loss, metrics = dppo_policy_loss(
        log_probs, None, advantages, config, loss_mask=loss_mask, rollout_logprobs=rollout_logprobs
    )

    with torch.no_grad():
        current, mu = log_probs.exp(), rollout_logprobs.exp()
        diff = current - mu
        mask = torch.ones_like(advantages)
        if dppo_type == "binary_tv":
            mask[(advantages > 0) & (diff > delta)] = 0.0
            mask[(advantages < 0) & (-diff > delta)] = 0.0
        else:
            kl = mu * (rollout_logprobs - log_probs) + (1 - mu) * torch.log((1 - mu + 1e-8) / (1 - current + 1e-8))
            mask[(advantages > 0) & (diff > 0) & (kl > delta)] = 0.0
            mask[(advantages < 0) & (diff < 0) & (kl > delta)] = 0.0
    ratio = (log_probs - rollout_logprobs).exp()
    torch.testing.assert_close(loss, (-(ratio * advantages * mask) * loss_mask).sum())
    assert 0 < metrics["clip_ratio"] < 1


def test_rollout_is_without_centering_matches_its_calibrated_policy_gradient():
    log_probs, rollout_logprobs, advantages, loss_mask = _loss_inputs()

    loss, _ = rollout_is_policy_loss(
        log_probs, None, advantages, CONFIG, loss_mask=loss_mask, rollout_logprobs=rollout_logprobs
    )

    ratio = (log_probs.detach() - rollout_logprobs).exp()
    calibrated = torch.where((ratio > 0.8) & (ratio < 1.25), ratio, 0.0)
    torch.testing.assert_close(loss, (-(calibrated * advantages * log_probs) * loss_mask).sum())


@pytest.mark.parametrize(
    "loss_fn, config",
    [
        (rollout_is_policy_loss, CONFIG),
        (compute_policy_loss_cispo, CISPO_ROLLOUT),
        (dppo_policy_loss, DPPO),
        (reinforce_policy_loss, AlgorithmConfig(policy_loss_type="reinforce")),
    ],
)
def test_centering_adds_the_advantage_weighted_term(loss_fn, config):
    log_probs, rollout_logprobs, advantages, loss_mask = _loss_inputs(seed=1)
    term = torch.randn(2, 4, dtype=torch.float64)
    kwargs = dict(loss_mask=loss_mask, rollout_logprobs=rollout_logprobs)

    centered, _ = loss_fn(log_probs, None, advantages, config, score_centering_term=term, **kwargs)
    plain, _ = loss_fn(log_probs, None, advantages, config, **kwargs)

    torch.testing.assert_close(centered - plain, (advantages * term * loss_mask).sum())


def test_reinforce_is_the_plain_policy_gradient():
    log_probs, rollout_logprobs, advantages, loss_mask = _loss_inputs()

    loss, metrics = reinforce_policy_loss(
        log_probs, None, advantages, AlgorithmConfig(), loss_mask=loss_mask, rollout_logprobs=rollout_logprobs
    )

    torch.testing.assert_close(loss, (-(advantages * log_probs) * loss_mask).sum())
    assert metrics["clip_ratio"] == 0.0


def test_old_anchored_cispo_refuses_a_centering_term():
    log_probs, rollout_logprobs, advantages, loss_mask = _loss_inputs()
    with pytest.raises(ValueError, match="cispo_anchor='rollout'"):
        compute_policy_loss_cispo(
            log_probs,
            log_probs.detach(),
            advantages,
            AlgorithmConfig(policy_loss_type="cispo"),
            loss_mask=loss_mask,
            rollout_logprobs=rollout_logprobs,
            score_centering_term=torch.zeros_like(log_probs),
        )


def test_each_sampler_anchored_loss_has_its_own_weight():
    assert score_centering_weight(PolicyLossType.REINFORCE, CONFIG) == UnitWeight()
    assert score_centering_weight(PolicyLossType.ROLLOUT_IS, CONFIG) == RatioBand.from_clip(0.2, 0.25)
    assert score_centering_weight(PolicyLossType.CISPO, CISPO_ROLLOUT) == RatioClamp(low=0.0, high=1.1)
    assert score_centering_weight(PolicyLossType.DPPO, DPPO) == DPPOTrustRegion(
        DPPOType.BINARY_TV, delta_low=0.05, delta_high=0.1
    )


@pytest.mark.parametrize(
    "loss_name, config",
    [
        (PolicyLossType.REGULAR, CONFIG),
        (PolicyLossType.GSPO, CONFIG),
        (PolicyLossType.CISPO, AlgorithmConfig(policy_loss_type="cispo")),
        (PolicyLossType.CROSS_ENTROPY, CONFIG),
    ],
)
def test_losses_weighted_against_the_old_policy_are_refused(loss_name, config):
    with pytest.raises(ValueError, match="weighted relative to the sampler"):
        score_centering_weight(loss_name, config)


def _scores(negative_term: bool = False) -> SampleSupportScores:
    return SampleSupportScores(
        logprobs=torch.zeros(1, 4),
        entropy=None,
        valid_mask=torch.tensor([[True, True, False, True]]),
        centering_term=torch.tensor([[0.5, -1.0, 0.0, 2.0]]),
        centering_term_negative=torch.tensor([[5.0, -10.0, 0.0, 20.0]]) if negative_term else None,
        sampler_logprobs=torch.tensor([[-0.1, -0.2, 0.0, -0.3]]),
        support_tv=torch.tensor([[0.1, 0.3, 0.0, 0.5]]),
        sampler_mass=torch.tensor([[1.0, 0.9, 0.0, 0.95]]),
    )


def test_positions_without_a_row_keep_the_rollout_logprob():
    advantages = torch.ones(1, 3)

    inputs = score_centering_loss_inputs(_scores(), slice(1, None), torch.full((1, 3), -7.0), advantages)

    torch.testing.assert_close(inputs.sampler_logprobs, torch.tensor([[-0.2, -7.0, -0.3]]))
    torch.testing.assert_close(inputs.term, torch.tensor([[-1.0, 0.0, 2.0]]))
    without_rollout = score_centering_loss_inputs(_scores(), slice(1, None), None, advantages)
    torch.testing.assert_close(without_rollout.sampler_logprobs, torch.tensor([[-0.2, 0.0, -0.3]]))


def test_a_sign_dependent_term_follows_each_token_advantage():
    advantages = torch.tensor([[1.0, -1.0, 0.0, -2.0]])

    inputs = score_centering_loss_inputs(_scores(negative_term=True), slice(None), None, advantages)

    torch.testing.assert_close(inputs.term, torch.tensor([[0.5, -10.0, 0.0, 20.0]]))
    assert inputs.loss_kwargs().keys() == {"score_centering_term"}


def test_metrics_average_scored_loss_positions_and_ignore_padding():
    loss_mask = torch.tensor([[1, 1, 1, 0]])
    scores = _scores()
    inputs = score_centering_loss_inputs(scores, slice(None), None, torch.ones(1, 4))

    metrics = score_centering_metrics(scores, inputs, slice(None), loss_mask)

    assert metrics[ScoreCenteringMetric.TERM_ABS_MEAN.value] == pytest.approx(0.75)
    assert metrics[ScoreCenteringMetric.SUPPORT_TV_MEAN.value] == pytest.approx(0.2)
    assert metrics[ScoreCenteringMetric.SAMPLER_ROW_MASS_MIN.value] == pytest.approx(0.9)
    padding = score_centering_metrics(scores, inputs, slice(None), torch.zeros(1, 4))
    assert padding[ScoreCenteringMetric.SAMPLER_ROW_MASS_MIN.value] == 1.0
