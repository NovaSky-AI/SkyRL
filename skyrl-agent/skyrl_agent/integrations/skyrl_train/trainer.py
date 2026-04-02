"""
SkyRL Trainer overrides needed for SkyRL-Agent integration.

SkyRL-Agent repeats trajectories inside `AgentRunner`, so it needs a custom
generator-input shape and a slightly looser validation rule than the default
trainer path.
"""

from typing import Any, Dict, List

import numpy as np
import torch

from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorOutput,
    TrainingPhase,
    TrajectoryID,
)
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.trainer_utils import validate_generator_output as validate_generator_output_impl


@register_advantage_estimator("loop")
def compute_advantages_and_returns_loop(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    values: torch.Tensor,
    config,
    gamma,
    lambd,
    grpo_norm_by_std,
    **kwargs,
):
    from collections import defaultdict

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    id2samples = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2samples[index[i]].append((i, scores[i]))
        for group in id2samples.values():
            group_size = len(group)
            total_score = sum(score for _, score in group)
            for i, score in group:  # i is original index
                loo_baseline = 0
                if group_size == 1:
                    print("Cannot compute LOO advantage using 1 sample. 0 baseline is used")
                else:
                    loo_baseline = (total_score - score) / (group_size - 1)
                scores[i] = score - loo_baseline
        scores = scores.unsqueeze(-1) * response_mask
        return scores, scores


class SkyRLAgentPPOTrainer(RayPPOTrainer):
    def prepare_generator_input(
        self,
        prompts: List[Any],
        training_phase: TrainingPhase,
        global_step: int | None,
    ) -> tuple[GeneratorInput, List[str]]:
        """Prepare generator input without repeating prompts for SkyRL-Agent."""
        if training_phase == "eval":
            n_samples_per_prompt = self.cfg.generator.eval_n_samples_per_prompt
            sampling_params = self.cfg.generator.eval_sampling_params
        else:
            n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt
            sampling_params = self.cfg.generator.sampling_params

        all_prompts = [prompt["prompt"] for prompt in prompts]
        env_extras = [prompt["env_extras"] for prompt in prompts]
        all_envs = [
            prompt["env_class"] if prompt["env_class"] is not None else self.cfg.environment.env_class
            for prompt in prompts
            for _ in range(n_samples_per_prompt)
        ]

        trajectory_ids = []
        uids = []
        for prompt in prompts:
            uid = prompt["uid"]
            for repetition_id in range(n_samples_per_prompt):
                trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=repetition_id))
                uids.append(uid)

        generator_input: GeneratorInput = {
            "prompts": all_prompts,
            "env_classes": all_envs,
            "env_extras": env_extras,
            "sampling_params": get_sampling_params_for_backend(
                self.cfg.generator.inference_engine.backend,
                sampling_params,
            ),
            "trajectory_ids": trajectory_ids,
            "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
        }

        return generator_input, uids

    def get_eval_metadata(
        self,
        generator_input: GeneratorInput,
        uids: List[str],
        generator_output: GeneratorOutput,
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Expand prompt-level metadata to match the evaluated trajectories."""
        trajectory_ids = generator_input.get("trajectory_ids")
        env_extras = generator_input.get("env_extras")
        if trajectory_ids is None or env_extras is None:
            return super().get_eval_metadata(generator_input, uids, generator_output)

        env_classes_by_traj_id = {
            (traj_id.instance_id, traj_id.repetition_id): env_class
            for traj_id, env_class in zip(trajectory_ids, generator_input["env_classes"])
        }

        instance_id_order = []
        seen_instance_ids = set()
        for traj_id in trajectory_ids:
            if traj_id.instance_id in seen_instance_ids:
                continue
            seen_instance_ids.add(traj_id.instance_id)
            instance_id_order.append(traj_id.instance_id)

        assert len(instance_id_order) == len(env_extras), (
            f"Mismatch between unique trajectory instance IDs ({len(instance_id_order)}) "
            f"and env_extras ({len(env_extras)})"
        )
        env_extras_by_instance_id = {
            instance_id: env_extra for instance_id, env_extra in zip(instance_id_order, env_extras)
        }

        output_trajectory_ids = generator_output.get("trajectory_ids") or trajectory_ids
        output_env_classes: List[str] = []
        output_env_extras: List[Dict[str, Any]] = []
        output_uids: List[str] = []
        for traj_id in output_trajectory_ids:
            key = (traj_id.instance_id, traj_id.repetition_id)
            assert key in env_classes_by_traj_id, f"Trajectory ID {traj_id.to_string()} not found in input"
            assert traj_id.instance_id in env_extras_by_instance_id, (
                f"Trajectory instance {traj_id.instance_id} missing env_extras"
            )
            output_env_classes.append(env_classes_by_traj_id[key])
            output_env_extras.append(env_extras_by_instance_id[traj_id.instance_id])
            output_uids.append(traj_id.instance_id)

        return output_env_classes, output_env_extras, output_uids

    def validate_generator_output(
        self,
        input_batch: GeneratorInput,
        generator_output: GeneratorOutput,
    ) -> None:
        """Allow prompt lists that are not repeated by `n_samples_per_prompt`."""
        step_wise = self.cfg.generator.step_wise_trajectories or generator_output.get("is_last_step") is not None
        num_prompts = len(input_batch["trajectory_ids"]) if input_batch.get("trajectory_ids") is not None else len(
            generator_output["response_ids"]
        )
        validate_generator_output_impl(num_prompts, generator_output, step_wise=step_wise)
