from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch


def get_scheduled_coeff(
    *,
    initial: float,
    final: float,
    step: int,
    total_steps: int,
    schedule: str = "constant",
    transition_step: int = 0,
    steps: Optional[list[int]] = None,
    values: Optional[list[float]] = None,
) -> float:
    if schedule == "constant":
        return initial
    if schedule == "linear_decay":
        t = min(step / max(total_steps, 1), 1.0)
        return initial + (final - initial) * t
    if schedule == "step_decay":
        if steps is not None and values is not None:
            coeff = values[0]
            for s, v in zip(steps, values):
                if step >= s:
                    coeff = v
            return coeff
        return initial if step < transition_step else final
    return initial


def compute_world_model_loss(
    action_log_probs: torch.Tensor,
    world_loss_mask: Optional[torch.Tensor],
    config: Any,
    warning_mask: Optional[torch.Tensor] = None,
    env_mask: Optional[torch.Tensor] = None,
    full_observation_count: Optional[torch.Tensor] = None,
    token_normalization_denominator: Optional[Union[float, torch.Tensor]] = None,
) -> Tuple[Optional[torch.Tensor], dict[str, Optional[torch.Tensor]]]:
    if world_loss_mask is None or config.world_model_coeff <= 0:
        return None, {}

    device = action_log_probs.device
    world_loss_mask = world_loss_mask.to(device).float()
    world_ce = -action_log_probs
    selected_tokens = world_loss_mask.sum()

    if selected_tokens <= 0:
        zero = action_log_probs.sum() * 0.0
        return zero, {
            "world_loss_unscaled": zero.detach(),
            "world_loss_scaled": zero.detach(),
            "world_policy_loss_ratio": None,
            "world_tokens_selected": selected_tokens.detach(),
            "world_tokens_warning": None,
            "world_tokens_env": None,
            "world_tokens_full_observation": selected_tokens.detach(),
            "world_ce_selected_per_token": zero.detach(),
            "world_ce_warning_per_token": None,
            "world_ce_env_per_token": None,
        }

    if config.loss_reduction == "sequence_mean":
        if config.world_loss_normalization == "full_observation_tokens" and full_observation_count is not None:
            denom = full_observation_count.to(device).float() + 1e-3
        else:
            denom = (world_loss_mask.sum(dim=-1, keepdim=True) + 1e-3).expand_as(world_loss_mask)
        world_loss_unscaled = ((world_ce * world_loss_mask) / denom).sum()
        full_obs_counts = torch.where(world_loss_mask > 0, denom, torch.zeros_like(denom))
        world_tokens_full_observation = full_obs_counts.max(dim=1).values.sum()
    elif config.loss_reduction == "seq_mean_token_sum_norm":
        if config.max_seq_len is None:
            raise ValueError("max_seq_len must be set when using seq_mean_token_sum_norm world loss.")
        world_loss_unscaled = (world_ce * world_loss_mask).sum() / config.max_seq_len
        world_tokens_full_observation = selected_tokens
    else:
        if token_normalization_denominator is None:
            token_denominator = selected_tokens + 1e-3
        elif isinstance(token_normalization_denominator, torch.Tensor):
            token_denominator = token_normalization_denominator.to(device).float()
        else:
            token_denominator = torch.tensor(float(token_normalization_denominator), device=device)
        world_loss_unscaled = (world_ce * world_loss_mask).sum() / token_denominator
        world_tokens_full_observation = selected_tokens

    world_loss_scaled = config.world_model_coeff * world_loss_unscaled
    selected_den = selected_tokens + 1e-3
    metrics: dict[str, Optional[torch.Tensor]] = {
        "world_loss_unscaled": world_loss_unscaled,
        "world_loss_scaled": world_loss_scaled,
        "world_policy_loss_ratio": None,
        "world_tokens_selected": selected_tokens,
        "world_tokens_warning": None,
        "world_tokens_env": None,
        "world_tokens_full_observation": world_tokens_full_observation,
        "world_ce_selected_per_token": (world_ce * world_loss_mask).sum() / selected_den,
        "world_ce_warning_per_token": None,
        "world_ce_env_per_token": None,
    }
    if warning_mask is not None:
        warning_mask = warning_mask.to(device).float()
        warning_tokens = warning_mask.sum()
        metrics["world_tokens_warning"] = warning_tokens
        metrics["world_ce_warning_per_token"] = (world_ce * warning_mask).sum() / (warning_tokens + 1e-3)
    if env_mask is not None:
        env_mask = env_mask.to(device).float()
        env_tokens = env_mask.sum()
        metrics["world_tokens_env"] = env_tokens
        metrics["world_ce_env_per_token"] = (world_ce * env_mask).sum() / (env_tokens + 1e-3)
    return world_loss_scaled, metrics
