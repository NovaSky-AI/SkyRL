import torch
from omegaconf import DictConfig
from typing import Tuple, Optional
from skyrl_train.utils.torch_utils import safe_exp_delta, masked_mean


def compute_tis_ratio(
    old_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    tis_ratio_type: str,
    off_policy_correction: DictConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute truncated importance sampling (TIS) ratio for off policy correction.

    Args:
        old_log_probs: Log probabilities from the old policy (before update).
        rollout_logprobs: Log probabilities from the rollout policy.
        loss_mask: Mask indicating valid tokens.
        tis_ratio_type: Type of TIS ratio ("token" or "sequence").
        off_policy_correction: Off-policy correction config containing cap values.

    Returns:
        Tuple of (tis_ratio, metrics):
        - tis_ratio: Tensor (float) to multiply with the loss
        - metrics: Dict with masking statistics

    Reference: https://github.com/szrlee/verl/blob/yingru/rollout_correction/docs/advance/rollout_corr_math.md
    """
    # Compute token-level importance ratio: pi_old / pi_rollout
    # In log space: old_log_probs - rollout_logprobs
    token_tis_log_ratio = old_log_probs - rollout_logprobs
    token_tis_ratio = safe_exp_delta(token_tis_log_ratio, clip=20.0, out_dtype=old_log_probs.dtype)

    metrics = {}
    if tis_ratio_type == "token":
        token_tis_ratio_cap = off_policy_correction.token_tis_ratio_clip_high
        # Compute proportion of tokens capped
        tokens_capped = (token_tis_ratio > token_tis_ratio_cap) & (loss_mask > 0)
        total_tokens = (loss_mask > 0).sum()
        metrics["tis_token_clip_high_ratio"] = (tokens_capped.sum() / total_tokens.clamp(min=1)).detach().item()
        return torch.clamp(token_tis_ratio, max=token_tis_ratio_cap).detach(), metrics
    elif tis_ratio_type == "sequence":
        # Compute sequence-level importance ratio as product of token ratios (sum of log ratios)
        seq_tis_log_ratio = (token_tis_log_ratio * loss_mask).sum(dim=-1, keepdim=True)
        seq_tis_ratio = safe_exp_delta(seq_tis_log_ratio, clip=20.0, out_dtype=old_log_probs.dtype)
        seq_tis_ratio_cap = off_policy_correction.sequence_tis_ratio_clip_high
        # Compute proportion of sequences capped
        num_sequences = seq_tis_ratio.shape[0]
        seqs_capped = (seq_tis_ratio > seq_tis_ratio_cap).sum()
        metrics["tis_seq_clip_high_ratio"] = (seqs_capped / num_sequences).detach().item()
        return torch.clamp(seq_tis_ratio, max=seq_tis_ratio_cap).detach(), metrics
    else:
        raise ValueError(f"Unknown tis_ratio_type: {tis_ratio_type}")


def compute_outlier_token_mask(
    old_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    off_policy_correction: DictConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute outlier token mask that masks out sequences with any token having
    importance ratio outside acceptable bounds.

    This is applied independently of TIS ratio type or sequence mask type,
    whenever off policy correction is enabled.

    Args:
        old_log_probs: Log probabilities from the old policy (before update).
        rollout_logprobs: Log probabilities from the rollout policy.
        loss_mask: Mask indicating valid tokens.
        off_policy_correction: Off-policy correction config containing threshold values.

    Returns:
        Tuple of (outlier_mask, metrics):
        - outlier_mask: Tensor (bool) to mask out sequences with any token having importance ratio outside acceptable bounds
        - metrics: Dict with masking statistics
    """
    metrics = {}
    # Compute token-level importance ratio
    token_tis_log_ratio = old_log_probs - rollout_logprobs
    token_tis_ratio = safe_exp_delta(token_tis_log_ratio, clip=20.0, out_dtype=old_log_probs.dtype)

    # Check per-token bounds
    token_mask_low = off_policy_correction.outlier_token_is_threshold_low
    token_mask_high = off_policy_correction.outlier_token_is_threshold_high
    if token_mask_high is not None:
        token_over_high = (token_tis_ratio > token_mask_high) & (loss_mask > 0)
    else:
        # no high threshold; so nothing is "over high"
        token_over_high = torch.zeros_like(loss_mask, dtype=torch.bool)

    if token_mask_low is not None:
        token_under_low = (token_tis_ratio < token_mask_low) & (loss_mask > 0)
    else:
        # no low threshold; so nothing is "under low"
        token_under_low = torch.zeros_like(loss_mask, dtype=torch.bool)
    token_in_bounds = ~token_over_high & ~token_under_low

    # A sequence is valid if all tokens are in bounds (considering only masked positions)
    all_tokens_valid = (token_in_bounds | (loss_mask == 0)).all(dim=-1, keepdim=True)

    # Compute metrics
    num_sequences = float(all_tokens_valid.shape[0])
    # Sequence has any token over high threshold
    seq_has_over_high = token_over_high.any(dim=-1)
    # Sequence has any token under low threshold
    seq_has_under_low = token_under_low.any(dim=-1)

    metrics["outlier_seq_masked_ratio"] = ((~all_tokens_valid.squeeze(-1)).sum() / num_sequences).detach().item()
    metrics["outlier_seq_over_high_ratio"] = (seq_has_over_high.sum() / num_sequences).detach().item()
    metrics["outlier_seq_under_low_ratio"] = (seq_has_under_low.sum() / num_sequences).detach().item()

    return all_tokens_valid.float(), metrics


def compute_sequence_mask(
    old_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sequence_mask_metric: str,
    off_policy_correction: DictConfig,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute sequence mask for off policy correction.

    This masks out sequences with importance ratios that fall outside acceptable bounds,
    helping to filter out off-policy samples that may destabilize training.

    Args:
        old_log_probs: Log probabilities from the old policy (before update).
        rollout_logprobs: Log probabilities from the rollout policy.
        loss_mask: Mask indicating valid tokens.
        sequence_mask_metric: Metric to use for sequence masking ("geometric" or "product").
        off_policy_correction: Off-policy correction config containing cap values.

    Returns:
        Tuple of (sequence_mask, metrics):
        - sequence_mask: Tensor (float) to multiply with the loss
        - metrics: Dict with masking statistics
    """
    # Compute token-level importance ratio
    token_tis_log_ratio = old_log_probs - rollout_logprobs
    metrics = {}

    if sequence_mask_metric == "geometric":
        # Compute geometric mean of importance ratios per sequence
        num_tokens = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        seq_tis_log_ratio = (token_tis_log_ratio * loss_mask).sum(dim=-1, keepdim=True)
        geo_mean_ratio = safe_exp_delta(seq_tis_log_ratio / num_tokens, clip=20.0, out_dtype=old_log_probs.dtype)
        geo_cap_high = off_policy_correction.geo_mask_high
        geo_cap_low = off_policy_correction.geo_mask_low
        seq_over_high = geo_mean_ratio > geo_cap_high
        seq_under_low = geo_mean_ratio < geo_cap_low
        geo_sequence_mask = ~seq_over_high & ~seq_under_low

        num_sequences = float(geo_mean_ratio.shape[0])
        metrics["geo_sequence_mask_masked_ratio"] = ((~geo_sequence_mask).sum() / num_sequences).detach().item()
        metrics["geo_sequence_mask_over_high_ratio"] = (seq_over_high.sum() / num_sequences).detach().item()
        metrics["geo_sequence_mask_under_low_ratio"] = (seq_under_low.sum() / num_sequences).detach().item()

        return geo_sequence_mask.float(), metrics
    elif sequence_mask_metric == "product":
        # Mask out sequences with product of importance ratios outside the cap
        seq_tis_log_ratio = (token_tis_log_ratio * loss_mask).sum(dim=-1, keepdim=True)
        seq_tis_ratio = safe_exp_delta(seq_tis_log_ratio, clip=20.0, out_dtype=old_log_probs.dtype)
        seq_cap_high = off_policy_correction.product_mask_high
        seq_cap_low = off_policy_correction.product_mask_low
        seq_over_high = seq_tis_ratio > seq_cap_high
        seq_under_low = seq_tis_ratio < seq_cap_low
        seq_in_bounds = ~seq_over_high & ~seq_under_low

        num_sequences = float(seq_tis_ratio.shape[0])
        metrics["product_sequence_mask_masked_ratio"] = ((~seq_in_bounds).sum() / num_sequences).detach().item()
        metrics["product_sequence_mask_over_high_ratio"] = (seq_over_high.sum() / num_sequences).detach().item()
        metrics["product_sequence_mask_under_low_ratio"] = (seq_under_low.sum() / num_sequences).detach().item()

        return seq_in_bounds.float(), metrics
    else:
        raise ValueError(f"Unknown sequence_mask_metric: {sequence_mask_metric}")


def compute_off_policy_correction(
    old_log_probs: torch.Tensor,
    rollout_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    off_policy_correction: DictConfig,
) -> Tuple[Optional[torch.Tensor], dict, torch.Tensor]:
    """
    Compute TIS ratio, sequence mask, and outlier token mask for off policy correction.

    This is a convenience function that combines compute_tis_ratio, compute_sequence_mask,
    and compute_outlier_token_mask.

    Args:
        old_log_probs: Log probabilities from the old policy (before update).
        rollout_logprobs: Log probabilities from the rollout policy.
        loss_mask: Mask indicating valid tokens.
        off_policy_correction: Off-policy correction config.

    Returns:
        Tuple of (tis_ratio, metrics, loss_mask):
        - tis_ratio: Tensor (float) to multiply with the loss
        - metrics: Dict with masking statistics
        - loss_mask: Mask indicating valid tokens after applying off policy correction

    References:
    - https://github.com/szrlee/verl/blob/yingru/rollout_correction/docs/advance/rollout_corr_math.md
    - https://fengyao.notion.site/off-policy-rl
    """
    tis_ratio_type = off_policy_correction.tis_ratio_type
    sequence_mask_metric = off_policy_correction.sequence_mask_metric

    # Check if TIS ratio correction is enabled
    apply_tis = tis_ratio_type is not None
    # Check if sequence mask is enabled
    apply_sequence_mask = sequence_mask_metric is not None

    # Early return if no correction needed
    if not apply_tis and not apply_sequence_mask:
        return None, {}, loss_mask

    is_ratio = safe_exp_delta(old_log_probs - rollout_logprobs, clip=20.0, out_dtype=old_log_probs.dtype)
    metrics = {}
    metrics["is_ratio_mean"] = masked_mean(is_ratio, loss_mask).mean().detach().item()
    metrics["is_ratio_std"] = (is_ratio * loss_mask).std().detach().item()
    metrics["is_ratio_max"] = (is_ratio * loss_mask).max().detach().item()
    metrics["is_ratio_min"] = (is_ratio * loss_mask).min().detach().item()

    # Apply outlier token mask whenever off policy correction is enabled
    # This rejects sequences with any token having importance ratio outside acceptable bounds
    outlier_mask, outlier_metrics = compute_outlier_token_mask(
        old_log_probs, rollout_logprobs, loss_mask, off_policy_correction
    )
    loss_mask = loss_mask * outlier_mask
    metrics.update(outlier_metrics)

    # Initialize tis_ratio to None (only set if TIS is enabled)
    tis_ratio = None

    # Apply TIS ratio if enabled
    if apply_tis:
        tis_ratio, tis_metrics = compute_tis_ratio(
            old_log_probs, rollout_logprobs, loss_mask, tis_ratio_type, off_policy_correction
        )
        metrics.update(tis_metrics)

    # Apply sequence mask if enabled
    if apply_sequence_mask:
        sequence_mask, sequence_mask_metrics = compute_sequence_mask(
            old_log_probs, rollout_logprobs, loss_mask, sequence_mask_metric, off_policy_correction
        )
        loss_mask = loss_mask * sequence_mask
        metrics.update(sequence_mask_metrics)

    return tis_ratio, metrics, loss_mask


def apply_off_policy_correction(
    loss: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_logprobs: Optional[torch.Tensor],
    loss_mask: torch.Tensor,
    off_policy_correction: DictConfig,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Apply off-policy correction to the loss tensor.

    This is a convenience wrapper around compute_off_policy_correction that
    handles the common pattern of applying the TIS ratio and updating metrics.

    Args:
        loss: The policy loss tensor to correct.
        old_log_probs: Log probabilities from the old policy (before update).
        rollout_logprobs: Log probabilities from the rollout policy (None if not available).
        loss_mask: Mask indicating valid tokens per sequence.
        off_policy_correction: Off-policy correction config.

    Returns:
        Tuple of (corrected_loss, updated_loss_mask, metrics):
        - corrected_loss: Loss tensor after applying TIS ratio (if enabled)
        - updated_loss_mask: Loss mask after applying sequence/outlier masks
        - metrics: Dict with off-policy correction statistics
    """
    metrics = {}
    if rollout_logprobs is not None:
        tis_ratio, off_policy_metrics, loss_mask = compute_off_policy_correction(
            old_log_probs, rollout_logprobs, loss_mask, off_policy_correction
        )
        if tis_ratio is not None:
            loss = loss * tis_ratio
        metrics.update(off_policy_metrics)
    return loss, loss_mask, metrics
