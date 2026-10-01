# Copyright 2025 SkyRL Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SFT training objectives.

Registered into the shared :class:`PolicyLossRegistry` alongside the RL losses in
``ppo_utils``, but kept separate because they answer a different question: RL losses weight
tokens by advantage, SFT losses weight them by likelihood (and optionally anchor to a frozen
reference).

Objectives, selected via ``loss_type``:

* ``cross_entropy`` -- ``-log p(y)``.
* ``dft``           -- ``-p(y).detach() * log p(y)`` (Dynamic Fine-Tuning).
* ``asft``          -- DFT + KL anchor to a frozen reference.
* ``kl_reg_sft``    -- cross-entropy + KL anchor.

The anchored losses need the reference's gold-token log-probs on the batch as
``base_action_log_probs``; see the SFT docs page for how to produce them.
"""

from typing import Optional, Tuple

import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import (
    PolicyLossType,
    reduce_loss,
    register_policy_loss,
)
from skyrl.train.config import AlgorithmConfig

# Losses whose signature accepts ``base_log_probs``; the workers only forward the
# frozen-reference log-probs for these, so every other loss keeps its existing signature.
LOSSES_WITH_BASE_LOGPROBS = frozenset({PolicyLossType.ASFT, PolicyLossType.KL_REG_SFT})


@register_policy_loss(PolicyLossType.CROSS_ENTROPY)
def cross_entropy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy loss for SFT: ``-log p(y)``, summed over masked tokens.

    The sum reduction matches Tinker's cross_entropy semantics. ``old_log_probs``,
    ``advantages`` and ``rollout_logprobs`` are ignored (RL-only).
    """
    loss = reduce_loss(-log_probs, loss_mask)
    return loss, {"clip_ratio": 0.0}


def _dft_token_weights(log_probs: torch.Tensor) -> torch.Tensor:
    """DFT per-token weight: the detached probability of the gold token.

    ``log_probs`` is already the gold token's log-prob (the model exposes selected-token
    logprobs, not full logits), so ``exp(log_probs).detach()`` matches the reference
    implementation's ``probs.gather(gold).detach()``.
    """
    return torch.exp(log_probs).detach()


@register_policy_loss(PolicyLossType.DFT)
def dft_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Dynamic Fine-Tuning (DFT) loss: ``-p(y).detach() * log p(y)``.

    ASFT without the KL anchor. ``old_log_probs``, ``advantages`` and ``rollout_logprobs``
    are ignored (RL-only).
    """
    weights = _dft_token_weights(log_probs)
    loss = reduce_loss(-weights * log_probs, loss_mask)
    return loss, {"clip_ratio": 0.0, "dft_weight_mean": weights.mean().item()}


def _anchored_loss(
    log_probs: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor],
    base_log_probs: Optional[torch.Tensor],
    *,
    use_dft_weight: bool,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Shared core for :func:`asft_loss` and :func:`kl_reg_sft_loss`.

    Both compute ``base_term + kl_coef * KL_hat(policy || reference)``; ``use_dft_weight``
    selects the NLL weight (``p(y).detach()`` for DFT, uniform for cross-entropy).
    """
    if base_log_probs is None:
        raise ValueError(
            "The anchored SFT losses require reference log-probs, but the batch carried no "
            "'base_action_log_probs'. Either set ref_source='live', or use a pretokenized "
            "dataset with a ref_logprobs column. To train without an anchor, choose "
            "loss_type='dft' (for asft) or loss_type='cross_entropy' (for kl_reg_sft)."
        )

    weights = _dft_token_weights(log_probs) if use_dft_weight else torch.ones_like(log_probs)
    base_elementwise = -weights * log_probs

    # Read the coefficient per call: the SFT trainer overrides it via loss_fn_config to ramp
    # the anchor in over the first N steps.
    kl_coef = float(getattr(config, "asft_kl_coef", 0.0))
    # Initialize every key unconditionally -- metrics are aggregated across micro-batches, and
    # a key present in some but absent in others breaks the reduction.
    metrics: dict[str, float] = {
        "clip_ratio": 0.0,
        "dft_weight_mean": weights.mean().item(),
        "asft_kl_coef_effective": kl_coef,
        "asft_kl": 0.0,
        "asft_kl_clamped_frac": 0.0,
        "asft_r_mean": 0.0,
        "asft_r_max": 0.0,
        "asft_r_gt5_frac": 0.0,
    }

    if kl_coef == 0.0:
        return reduce_loss(base_elementwise, loss_mask), metrics

    estimator = getattr(config, "asft_kl_estimator", "k3")
    # r = log pi_ref(y) - log pi_theta(y); gradient flows only through the policy.
    log_ratio = base_log_probs.detach() - log_probs

    # Clamp r before the exponential: k3's gradient is -exp(r), so a single token at r=18
    # outweighs ~1e7 typical ones and can dominate the batch. At the default r_max=10 this is
    # inert for tokens where policy and reference broadly agree.
    r_max = float(getattr(config, "asft_kl_clamp", 10.0))
    clamped_hits = None
    if r_max > 0:
        # Keep the pre-clamp indicator, not a count: the metric below restricts it to
        # loss-masked positions, and after clamping ``|log_ratio| > r_max`` is false everywhere.
        clamped_hits = log_ratio.abs() > r_max
        log_ratio = log_ratio.clamp(-r_max, r_max)

    if estimator == "k1":
        kl = -log_ratio
    elif estimator == "k3":
        # exp(r) - r - 1 >= 0: low-variance estimator of KL(theta || ref).
        kl = torch.expm1(log_ratio) - log_ratio
    else:
        raise ValueError(f"Unknown asft_kl_estimator {estimator!r}; expected 'k1' or 'k3'.")

    elementwise_loss = base_elementwise + kl_coef * kl

    if loss_mask is not None:
        # One device sync for the KL metric and all diagnostics: this runs per micro-batch
        # inside a pipeline schedule, where each .item() is a blocking sync. Masked reductions
        # only -- no sort, no boolean indexing, no quantile -- stacked and pulled across once.
        with torch.no_grad():
            mf = (loss_mask > 0).to(log_ratio.dtype)
            n = mf.sum().clamp(min=1)
            zero = torch.zeros((), dtype=log_ratio.dtype, device=log_ratio.device)
            stacked = torch.stack(
                [
                    (kl * loss_mask).sum() / loss_mask.sum().clamp(min=1),  # asft_kl
                    (log_ratio * mf).sum() / n,  # r_mean
                    (log_ratio * mf + (mf - 1) * 1e9).max(),  # r_max (masked)
                    ((log_ratio > 5).to(log_ratio.dtype) * mf).sum() / n,  # r_gt5_frac
                    # Masked like the rest: the fraction of TRAINED tokens beyond the clamp.
                    ((clamped_hits.to(log_ratio.dtype) * mf).sum() / n) if clamped_hits is not None else zero,
                ]
            )
            kl_m, r_mean, r_max_v, r_gt5, cl_frac = stacked.tolist()
        metrics["asft_kl"] = kl_m
        metrics["asft_r_mean"] = r_mean
        metrics["asft_r_max"] = r_max_v
        metrics["asft_r_gt5_frac"] = r_gt5
        metrics["asft_kl_clamped_frac"] = cl_frac
    else:
        metrics["asft_kl"] = kl.mean().item()

    return reduce_loss(elementwise_loss, loss_mask), metrics


@register_policy_loss(PolicyLossType.ASFT)
def asft_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    base_log_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Anchored SFT (ASFT): DFT-weighted NLL + KL anchor.

    ``loss = -p(y).detach() * log p(y) + kl_coef * KL_hat(policy || reference)``

    The anchor uses a token-level estimator on the gold-token log-probs rather than the paper's
    full-vocabulary KL, so it needs only the reference's gold-token log-prob. With
    ``r = base_log_probs - log_probs``, ``config.asft_kl_estimator`` selects:

    * ``"k3"`` (default): ``exp(r) - r - 1`` -- non-negative, low variance.
    * ``"k1"``: ``-r`` -- unbiased, higher variance.

    Raises ``ValueError`` if ``base_log_probs`` is missing; use ``loss_type="dft"`` to train
    without an anchor. ``old_log_probs``, ``advantages`` and ``rollout_logprobs`` are ignored.
    """
    return _anchored_loss(log_probs, config, loss_mask, base_log_probs, use_dft_weight=True)


@register_policy_loss(PolicyLossType.KL_REG_SFT)
def kl_reg_sft_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
    base_log_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """KL-regularized SFT: plain cross-entropy NLL + KL anchor (ASFT without DFT reweighting).

    ``loss = -log p(y) + kl_coef * KL_hat(policy || reference)``

    DFT's ``p(y)`` weight suppresses hard/low-probability tokens, which starves acquisition of
    capabilities the base model lacks; dropping it restores CE-style learning while keeping the
    anchor for in-distribution retention. Prefer this when the mixture teaches new capability.
    The anchor still pulls toward the base, so keep ``asft_kl_coef`` low if you need to exceed
    the base model's score.

    Same estimators and clamp as :func:`asft_loss`, and likewise raises if ``base_log_probs`` is
    missing; use ``loss_type="cross_entropy"`` to train without an anchor.
    """
    return _anchored_loss(log_probs, config, loss_mask, base_log_probs, use_dft_weight=False)
