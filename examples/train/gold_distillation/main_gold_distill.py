"""
GOLD (General On-policy Logit Distillation) Trainer for Cross-Tokenizer Distillation.

This implements GOLD as a supervised distillation method where the loss (JSD on matched
tokens + L1 on sorted unmatched tokens) is computed directly inside the policy worker
with gradients flowing through the student model.

The teacher model is loaded inside the policy worker to avoid passing ~15GB+ of teacher
logits through the data pipeline. The trainer prepares teacher tokenization and alignment
data, which are small tensors that propagate through metadata.

Reference:
- HuggingFace blog: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- TRL GOLDTrainer: https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py
"""

from typing import List
import os
import sys

import ray
from loguru import logger
from skyrl.train.config import SkyRLTrainConfig
from .gold_config import GOLDSkyRLTrainConfig
from transformers import AutoTokenizer
import torch

from skyrl.train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    validate_cfg,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl.train.generators.base import GeneratorInterface

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import CriticWorker

from .gold_workers import GOLDPolicyWorker, GOLDRefWorker
from .gold_utils import build_teacher_inputs_from_texts


class GOLDDistillationTrainer(RayPPOTrainer):
    """
    Trainer for GOLD supervised distillation.

    Key differences from standard PPO:
    - Teacher tokenization and alignment are prepared here (small tensors)
    - Teacher model forward happens inside the policy worker (not here)
    - Rewards/advantages are zeroed out (GOLD is supervised, not RL)
    - The custom _forward_backward_micro in GOLDPolicyWorker computes the actual loss
    """

    def __init__(
        self,
        cfg,
        ref_tokenizer,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)
        self.ref_tokenizer = ref_tokenizer

    def update_ref_with_policy(self):
        """No-op: teacher model is frozen, never updated from student."""
        pass

    def _decode_sequences_to_texts(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[List[str], List[str]]:
        """Decode student sequences into prompt and completion texts."""
        prompt_texts = []
        completion_texts = []
        batch_size = sequences.size(0)

        prompt_offset = sequences.size(1) - loss_mask.size(1)

        for i in range(batch_size):
            seq = sequences[i]
            attn_mask = attention_mask[i].bool()
            loss_m = loss_mask[i]

            response_positions = loss_m.nonzero(as_tuple=True)[0]
            if len(response_positions) > 0:
                response_start = response_positions[0].item()
                response_end = response_positions[-1].item() + 1
            else:
                response_start = loss_m.size(0)
                response_end = loss_m.size(0)

            prompt_ids = seq[: prompt_offset + response_start][attn_mask[: prompt_offset + response_start]].tolist()
            completion_ids = seq[prompt_offset + response_start : prompt_offset + response_end][
                attn_mask[prompt_offset + response_start : prompt_offset + response_end]
            ].tolist()

            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

            prompt_texts.append(prompt_text)
            completion_texts.append(completion_text)

        return prompt_texts, completion_texts

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Prepare data for GOLD supervised distillation.

        Instead of running ref model forward to get base log probs, this method:
        1. Computes student action_log_probs (for logging/metrics)
        2. Decodes student sequences to text
        3. Re-tokenizes with teacher tokenizer
        4. Stores teacher tokenization as tensors in the TrainingInputBatch dict
           (so they auto-chunk per micro-batch alongside student data)

        The actual teacher forward and loss computation happen inside the policy worker's
        custom _gold_forward_backward_micro.
        """
        from skyrl.backends.skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch

        data_fwd_pass = training_input.select(keys=["sequences", "attention_mask"], metadata_keys=["response_length"])

        def collect_results(actor_infos, results, key):
            from skyrl.backends.skyrl_train.training_batch import TrainingOutputBatch

            ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(actor_infos, results)
            return ret_outputs[key]

        action_log_probs = None
        values = None

        # Calculate critic values (if critic exists)
        if self.colocate_all and self.critic_model is not None:
            self.critic_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        if self.critic_model is not None:
            value_refs = self.critic_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
            if self.colocate_all:
                all_rank_values = ray.get(value_refs)
                values = collect_results(self.critic_model.actor_infos, all_rank_values, key="output")
                self.critic_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        # Calculate student action log probs (for metrics/logging only)
        if self.colocate_all:
            self.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        action_log_probs_refs = self.policy_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
        if self.colocate_all:
            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(self.policy_model.actor_infos, all_rank_action_log_probs, key="output")
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        # Wait for non-colocated results
        if not self.colocate_all:
            if self.critic_model is not None:
                all_rank_values = ray.get(value_refs)
                values = collect_results(self.critic_model.actor_infos, all_rank_values, key="output")

            all_rank_action_log_probs: List[TrainingOutputBatch] = ray.get(action_log_probs_refs)
            action_log_probs = collect_results(self.policy_model.actor_infos, all_rank_action_log_probs, key="output")

        # Empty cache
        if not self.colocate_all:
            empty_cache_refs = self.policy_model.async_run_ray_method("pass_through", "empty_cache")
            if self.critic_model is not None:
                empty_cache_refs.extend(self.critic_model.async_run_ray_method("pass_through", "empty_cache"))
            ray.get(empty_cache_refs)

        sequences_all: torch.Tensor = training_input["sequences"]
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        # Set base_action_log_probs to None (GOLD doesn't use ref log probs for RL)
        training_input["base_action_log_probs"] = None
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values

        # Prepare teacher tokenization data and store as proper tensors in the batch dict
        # so they get auto-chunked alongside student data during micro-batching.
        sequences = training_input["sequences"]
        attention_mask = training_input["attention_mask"]
        loss_mask = training_input["loss_mask"]

        prompt_texts, completion_texts = self._decode_sequences_to_texts(sequences, attention_mask, loss_mask)

        teacher_input_ids, teacher_labels, teacher_attention_mask, _ = build_teacher_inputs_from_texts(
            self.ref_tokenizer,
            prompt_texts,
            completion_texts,
        )

        # Store as tensors in the dict (NOT metadata) so they auto-chunk per micro-batch
        training_input["gold_teacher_input_ids"] = teacher_input_ids
        training_input["gold_teacher_labels"] = teacher_labels
        training_input["gold_teacher_attention_mask"] = teacher_attention_mask.long()

        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """
        Return zero rewards. GOLD is supervised distillation — the actual loss is
        computed in the policy worker's _forward_backward_micro, not through rewards.
        """
        loss_mask = data["loss_mask"]
        data["rewards"] = torch.zeros_like(loss_mask, dtype=torch.float32)
        return data


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    """Pass through zero rewards as zero advantages."""
    return token_level_rewards, token_level_rewards


class GOLDDistillationExp(BasePPOExp):
    def __init__(self, cfg: SkyRLTrainConfig):
        super().__init__(cfg)
        self.ref_tokenizer = self.get_tokenizer(model_path=cfg.trainer.ref.model.path)

    def get_trainer(self, *args, **kwargs):
        return GOLDDistillationTrainer(*args, **kwargs)

    def get_tokenizer(self, padding_side="left", model_path=None):
        if model_path is None:
            model_path = self.cfg.trainer.policy.model.path

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _setup_trainer(self):
        """
        Set up the GOLD distillation trainer.

        Builds models with GOLDPolicyWorker (which loads teacher model internally)
        and GOLDRefWorker (kept for pipeline compatibility).
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "megatron":
            raise NotImplementedError("Megatron not supported for GOLD distillation")

        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        ref_tokenizer = self.ref_tokenizer
        if self.cfg.generator.inference_engine.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(
            inference_engines,
            tokenizer,
            self.cfg.trainer.policy.model.path,
            self.cfg.trainer.policy.model.lora,
            self.cfg.generator.inference_engine,
        )

        generator: GeneratorInterface = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            ref_tokenizer=ref_tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build models: GOLDPolicyWorker loads teacher internally, CriticWorker optional
        trainer.build_models(GOLDPolicyWorker, CriticWorker, GOLDRefWorker)
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = GOLDDistillationExp(cfg)
    exp.run()


def main() -> None:
    cfg = GOLDSkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
