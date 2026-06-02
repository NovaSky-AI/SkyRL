from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional

import ray
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase
from skyrl.train.dataset.replay_buffer import Experience

from .loss import compute_world_model_loss, get_scheduled_coeff


class EchoFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    """FSDP policy worker that adds ECHO's auxiliary world-modeling loss."""

    def get_aux_policy_loss_context(self, data: TrainingInputBatch) -> Dict[str, Any]:
        world_loss_mask = data.get("world_loss_mask")
        token_denominator = None
        if world_loss_mask is not None:
            token_denominator = float(world_loss_mask.sum().item() + 1e-3)
        return {
            "world_loss_weight": 1.0 / max(len(data), 1),
            "world_token_normalization_denominator": token_denominator,
        }

    def compute_aux_policy_loss(
        self,
        *,
        action_log_probs: torch.Tensor,
        policy_loss: torch.Tensor,
        experience: Experience,
        loss_config: Any,
        microbatch_weight: float,
        grad_sum_correction_factor: float,
        aux_policy_loss_context: Optional[Dict[str, Any]] = None,
    ):
        extras = experience.extras or {}
        metadata = experience.metadata or {}
        global_step = int(metadata.get("world_model_schedule_step", metadata.get("global_step", 0)))
        total_training_steps = int(metadata.get("total_training_steps", 0))
        world_model_coeff = get_scheduled_coeff(
            initial=loss_config.world_model_coeff,
            final=loss_config.world_model_coeff_end,
            step=global_step,
            total_steps=total_training_steps,
            schedule=loss_config.world_model_coeff_schedule,
            transition_step=loss_config.world_model_coeff_transition_step,
            steps=loss_config.world_model_coeff_steps,
            values=loss_config.world_model_coeff_values,
        )
        loss_config = replace(loss_config, world_model_coeff=world_model_coeff)
        context = aux_policy_loss_context or {}
        world_loss_scaled, world_metrics = compute_world_model_loss(
            action_log_probs,
            extras.get("world_loss_mask"),
            config=loss_config,
            warning_mask=extras.get("world_warning_mask"),
            env_mask=extras.get("world_env_mask"),
            full_observation_count=extras.get("world_full_observation_count"),
            token_normalization_denominator=context.get("world_token_normalization_denominator"),
        )
        if world_loss_scaled is None:
            coeff_tensor = torch.tensor(float(world_model_coeff), device=action_log_probs.device)
            return None, {
                "status/world_model_coeff": coeff_tensor,
            }

        if loss_config.loss_reduction in ("sequence_mean", "seq_mean_token_sum_norm"):
            aux_loss = world_loss_scaled * context.get("world_loss_weight", microbatch_weight)
        else:
            aux_loss = world_loss_scaled

        world_metrics = dict(world_metrics)
        world_metrics["status/world_model_coeff"] = torch.tensor(
            float(world_model_coeff), device=action_log_probs.device
        )
        if world_metrics.get("world_loss_scaled") is not None:
            world_metrics["world_policy_loss_ratio"] = (
                world_metrics["world_loss_scaled"].detach().abs() / (policy_loss.detach().abs() + 1e-8)
            )
        return aux_loss, world_metrics


PolicyWorker = ray.remote(num_gpus=1)(EchoFSDPPolicyWorkerBase)
