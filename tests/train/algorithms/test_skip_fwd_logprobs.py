"""
Tests for skipping the policy forward pass when the loss optimizes against rollout logprobs,
and for the worker-side per-mini-batch train/rollout logprob diff metric.

uv run --isolated --extra dev pytest tests/train/algorithms/test_skip_fwd_logprobs.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    off_policy_correction_enabled,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    LOSSES_WITHOUT_OLD_LOGPROBS,
    dppo_policy_loss,
    rollout_is_policy_loss,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.backends.skyrl_train.workers.worker_utils import (
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
    MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY,
    compute_minibatch_rollout_logprob_diff_metrics,
    reduce_metrics,
)
from skyrl.train.config import AlgorithmConfig, OffPolicyCorrectionConfig
from skyrl.train.utils.trainer_utils import finalize_minibatch_rollout_logprob_diff_std


def test_losses_without_old_logprobs():
    """rollout_is and dppo optimize against rollout logprobs and do not need old logprobs."""
    assert "rollout_is" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "dppo" in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "regular" not in LOSSES_WITHOUT_OLD_LOGPROBS
    assert "gspo" not in LOSSES_WITHOUT_OLD_LOGPROBS


@pytest.mark.parametrize(
    "loss_name, loss_fn",
    [("rollout_is", rollout_is_policy_loss), ("dppo", dppo_policy_loss)],
)
def test_skip_path_losses_run_with_old_log_probs_none(loss_name, loss_fn):
    """Skip-path invariant: these losses must produce a finite loss with ``old_log_probs=None``
    when off-policy correction is disabled (rollout logprobs stand in for the old logprobs)."""
    assert loss_name in LOSSES_WITHOUT_OLD_LOGPROBS
    config = AlgorithmConfig(policy_loss_type=loss_name, off_policy_correction=OffPolicyCorrectionConfig())
    assert not off_policy_correction_enabled(config.off_policy_correction)

    log_probs = torch.tensor([[-1.0, -1.5, -0.8], [-0.7, -1.2, -2.0]])
    rollout_logprobs = torch.tensor([[-1.2, -1.0, -0.9], [-0.6, -1.5, -1.7]])
    advantages = torch.tensor([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    loss, _ = loss_fn(
        log_probs,
        None,  # old_log_probs: skipped forward pass
        advantages,
        config=config,
        loss_mask=loss_mask,
        rollout_logprobs=rollout_logprobs,
    )
    assert torch.isfinite(loss)


def test_off_policy_correction_enabled():
    """The helper detects every path in compute_off_policy_correction that reads old logprobs."""
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig())
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(tis_ratio_type="token"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(sequence_mask_metric="product"))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(OffPolicyCorrectionConfig(outlier_token_is_threshold_high=100.0))
    # token mask requires both bounds to be set
    assert not off_policy_correction_enabled(OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4))
    assert off_policy_correction_enabled(
        OffPolicyCorrectionConfig(token_mask_is_threshold_low=1e-4, token_mask_is_threshold_high=100.0)
    )


def test_minibatch_metric_values_match_manual():
    """The worker metric reports the masked abs diff moments/extremes via the same masked_mean /
    `* loss_mask` pattern as the off-policy `is_ratio_*` metrics."""
    action_log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    rollout_logprobs = torch.tensor([[-1.5, -1.0, -3.0], [-0.5, -2.0, -1.0]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    metrics = compute_minibatch_rollout_logprob_diff_metrics(action_log_probs, rollout_logprobs, loss_mask)

    abs_diff = (action_log_probs - rollout_logprobs).abs()
    masked = abs_diff * loss_mask
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY] == masked_mean(abs_diff, loss_mask).item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY] == masked_mean(abs_diff.square(), loss_mask).item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY] == masked.max().item()
    assert metrics[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY] == masked.min().item()


def test_minibatch_metric_emitted_unconditionally_when_rollout_present():
    """Metric is omitted only when rollout logprobs are absent (a DP-uniform condition). With
    rollout logprobs present but every token masked, it still emits the keys as 0.0 so the key set
    stays consistent across DP ranks."""
    alp = torch.tensor([[-1.0, -2.0]])
    rlp = torch.tensor([[-1.5, -1.0]])
    assert compute_minibatch_rollout_logprob_diff_metrics(alp, None, None) == {}

    fully_masked = torch.zeros_like(alp)
    masked_metrics = compute_minibatch_rollout_logprob_diff_metrics(alp, rlp, fully_masked)
    assert set(masked_metrics) == {
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MAX_KEY,
        MINIBATCH_ROLLOUT_LOGPROB_DIFF_MIN_KEY,
    }
    assert all(v == 0.0 for v in masked_metrics.values())


def test_finalize_std_noop_without_moments():
    """The std derivation is a no-op when the moment keys are absent (e.g. critic training)."""
    metrics = {"foo": 1.0}
    finalize_minibatch_rollout_logprob_diff_std(metrics)
    assert metrics == {"foo": 1.0}


def test_std_aggregates_correctly_across_microbatches_and_minibatches():
    """Reduce per-micro-batch moments over micro-batches then mini-batches, derive the std, and
    check it equals the exact pooled population std (exact because all sizes are equal)."""
    # Four equal-size micro-batches grouped into two mini-batches of two micro-batches each.
    diffs = [
        torch.tensor([0.1, 0.3, 0.5, 0.2]),
        torch.tensor([1.0, 0.8, 0.6, 0.4]),
        torch.tensor([0.2, 0.2, 0.9, 0.7]),
        torch.tensor([0.3, 0.5, 0.1, 0.6]),
    ]

    def moments(d):
        return {
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY: d.mean().item(),
            MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY: d.square().mean().item(),
        }

    keys = list(moments(diffs[0]).keys())
    # Worker-level reduction over micro-batches within each mini-batch.
    mb1 = reduce_metrics({k: [moments(diffs[0])[k], moments(diffs[1])[k]] for k in keys})
    mb2 = reduce_metrics({k: [moments(diffs[2])[k], moments(diffs[3])[k]] for k in keys})

    # Trainer-level reduction across mini-batches.
    all_metrics = {k: [mb1[k], mb2[k]] for k in keys}
    reduced = reduce_metrics(all_metrics)
    finalize_minibatch_rollout_logprob_diff_std(reduced)

    # All micro-batches are equal size, so the pooled mean/std are exact.
    pooled = torch.cat(diffs)
    assert MINIBATCH_ROLLOUT_LOGPROB_DIFF_SQ_MEAN_KEY not in reduced
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_MEAN_KEY] - pooled.mean().item()) < 1e-6
    expected_std = pooled.std(unbiased=False).item()
    assert abs(reduced[MINIBATCH_ROLLOUT_LOGPROB_DIFF_STD_KEY] - expected_std) < 1e-6
