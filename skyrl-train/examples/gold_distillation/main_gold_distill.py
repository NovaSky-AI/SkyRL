"""
GOLD (General On-policy Logit Distillation) Trainer for Cross-Tokenizer Distillation.

This example extends the on-policy distillation example (PR #585) to support
student-teacher pairs with different tokenizers. It loads a separate teacher
tokenizer and logs cross-tokenizer statistics.

Reference:
- HuggingFace blog: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- TRL GOLDTrainer: https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py
"""

from typing import List
import os

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    validate_cfg,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.generators.base import GeneratorInterface

from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
)
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.workers.fsdp.fsdp_worker import CriticWorker
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch

from examples.gold_distillation.gold_workers import GOLDPolicyWorker, GOLDRefWorker
from examples.gold_distillation.gold_utils import (
    build_teacher_inputs_from_texts,
    build_alignment_groups_from_ids,
    merge_probabilities_with_alignment_groups,
)


class GOLDDistillationTrainer(RayPPOTrainer):
    """
    Custom trainer for GOLD (General On-policy Logit Distillation).

    Extends OnPolicyDistillationTrainer to support different tokenizers between
    student and teacher models using GOLD/ULD for cross-tokenizer
    distillation.
    """

    def __init__(
        self,
        cfg,
        ref_tokenizer,
        **kwargs,
    ):
        super().__init__(cfg, **kwargs)

        self.ref_tokenizer = ref_tokenizer
        self.gold_temperature = self.cfg.trainer.algorithm.get("gold_temperature", 1.0)

    def update_ref_with_policy(self):
        """
        We want this to be a no-op for cross-tokenizer distillation.
        """
        pass

    def _decode_sequences_to_texts(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple[List[str], List[str]]:
        """
        Decode student sequences into prompt and completion texts.

        Args:
            sequences: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            loss_mask: Loss mask indicating response tokens [batch, seq_len]

        Returns:
            prompt_texts: List of prompt strings
            completion_texts: List of completion strings
        """
        prompt_texts = []
        completion_texts = []
        batch_size = sequences.size(0)

        for i in range(batch_size):
            seq = sequences[i]
            attn_mask = attention_mask[i].bool()
            loss_m = loss_mask[i]

            # Find where the response starts (first 1 in loss_mask)
            response_positions = loss_m.nonzero(as_tuple=True)[0]
            if len(response_positions) > 0:
                response_start = response_positions[0].item()
                response_end = response_positions[-1].item() + 1
            else:
                # No response tokens, treat entire sequence as prompt
                response_start = seq.size(0)
                response_end = seq.size(0)

            # Use attention_mask to filter real tokens (exclude padding)
            prompt_ids = seq[:response_start][attn_mask[:response_start]].tolist()
            completion_ids = seq[response_start:response_end][attn_mask[response_start:response_end]].tolist()

            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

            prompt_texts.append(prompt_text)
            completion_texts.append(completion_text)

        return prompt_texts, completion_texts

    def _get_teacher_logits(
        self,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run teacher model forward pass to get logits.

        NOTE: For cross-tokenizer GOLD distillation, the ref model worker must be configured
        to return full logits (shape [batch, seq_len, vocab_size]) instead of log_probs.

        Args:
            teacher_input_ids: Teacher-tokenized input IDs [batch, seq_len]
            teacher_attention_mask: Attention mask [batch, seq_len]

        Returns:
            Teacher logits [batch, seq_len, vocab_size] or None if unavailable
        """
        # Prepare data for ref model forward pass
        data_fwd_pass = TrainingInputBatch(
            {
                "sequences": teacher_input_ids,
                "attention_mask": teacher_attention_mask,
            }
        )
        # The ref model expects response_length in metadata
        data_fwd_pass.metadata = {"response_length": teacher_input_ids.size(1)}

        # Run forward pass on ref model
        if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
            self.ref_model.backload_to_gpu()

        logits_refs = self.ref_model.async_run_ray_method("mesh", "forward", data=data_fwd_pass)
        all_rank_outputs: List[TrainingOutputBatch] = ray.get(logits_refs)

        # Collect results
        ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(
            self.ref_model.actor_infos, all_rank_outputs
        )

        if self.cfg.trainer.placement.colocate_policy_ref or self.colocate_all:
            self.ref_model.offload_to_cpu()
            ray.get(self.ref_model.async_run_ray_method("pass_through", "empty_cache"))

        output = ret_outputs["output"]

        # Check if we got logits (3D) or log_probs (2D)
        if output.dim() == 2:
            # Got log_probs instead of logits - worker not configured for GOLD
            return None
        return output

    def _get_student_logits(
        self,
        student_input_ids: torch.Tensor,
        student_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run student/policy model forward pass to get logits.

        NOTE: For cross-tokenizer GOLD distillation, the policy model worker must be configured
        to return full logits (shape [batch, seq_len, vocab_size]) instead of log_probs.

        Args:
            student_input_ids: Student-tokenized input IDs [batch, seq_len]
            student_attention_mask: Attention mask [batch, seq_len]

        Returns:
            Student logits [batch, seq_len, vocab_size] or None if unavailable
        """
        # Prepare data for policy model forward pass
        data_fwd_pass = TrainingInputBatch(
            {
                "sequences": student_input_ids,
                "attention_mask": student_attention_mask,
            }
        )
        data_fwd_pass.metadata = {"response_length": student_input_ids.size(1)}

        # Run forward_logits pass on policy model to get logits (not log probs)
        if self.colocate_all:
            self.policy_model.backload_to_gpu(backload_optimizer=False, backload_model=True)

        logits_refs = self.policy_model.async_run_ray_method("mesh", "forward_logits", data=data_fwd_pass)
        all_rank_outputs: List[TrainingOutputBatch] = ray.get(logits_refs)

        # Collect results
        ret_outputs: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(
            self.policy_model.actor_infos, all_rank_outputs
        )

        if self.colocate_all:
            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)

        output = ret_outputs["output"]

        # Check if we got logits (3D) or log_probs (2D)
        if output.dim() == 2:
            # Got log_probs instead of logits - worker not configured for GOLD
            return None
        return output

    def _log_alignment_sample(
        self,
        student_token_ids: list[int],
        teacher_token_ids: list[int],
        student_groups: list[list[int]],
        teacher_groups: list[list[int]],
    ):
        """
        Log a sample alignment for debugging/visualization.

        Args:
            student_token_ids: Student token IDs
            teacher_token_ids: Teacher token IDs
            student_groups: Student alignment groups
            teacher_groups: Teacher alignment groups
        """
        # Decode tokens
        student_decoded = [self.tokenizer.decode([tid]) for tid in student_token_ids]
        teacher_decoded = [self.ref_tokenizer.decode([tid]) for tid in teacher_token_ids]

        logger.info("=" * 60)
        logger.info("GOLD Alignment Sample:")
        logger.info(f"Student tokens ({len(student_token_ids)}): {student_decoded}")
        logger.info(f"Teacher tokens ({len(teacher_token_ids)}): {teacher_decoded}")
        logger.info(f"Alignment groups ({len(student_groups)}):")

        for i, (s_group, t_group) in enumerate(zip(student_groups, teacher_groups)):
            s_tokens = [student_decoded[idx] for idx in s_group]
            t_tokens = [teacher_decoded[idx] for idx in t_group]
            s_text = "".join(s_tokens)
            t_text = "".join(t_tokens)
            logger.info(f"  Group {i}: '{s_text}' ({s_tokens}) ↔ '{t_text}' ({t_tokens})")
        logger.info("=" * 60)

    def _log_probability_comparison(
        self,
        student_probs: torch.Tensor,
        teacher_probs: torch.Tensor,
        top_k: int = 5,
    ):
        """
        Log top-k probabilities for student vs teacher.

        Args:
            student_probs: Student probability distribution [vocab_size]
            teacher_probs: Teacher probability distribution [vocab_size]
            top_k: Number of top probabilities to log
        """
        s_top_probs, s_top_ids = student_probs.topk(top_k)
        t_top_probs, t_top_ids = teacher_probs.topk(top_k)

        logger.info("Probability Comparison (first aligned group):")
        logger.info(f"  Student top-{top_k}: {list(zip(s_top_ids.tolist(), s_top_probs.tolist()))}")
        logger.info(f"  Teacher top-{top_k}: {list(zip(t_top_ids.tolist(), t_top_probs.tolist()))}")

    def _compute_gold_rewards(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        student_loss_mask: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token GOLD rewards using ULD methodology.

        This aligns token spans between student and teacher, then computes
        per-token rewards based on how well the student matches the teacher's
        probability distribution.

        Args:
            student_logits: Student logits [batch, student_seq_len, student_vocab]
            teacher_logits: Teacher logits [batch, teacher_seq_len, teacher_vocab]
            student_input_ids: Student token IDs [batch, student_seq_len]
            teacher_input_ids: Teacher token IDs [batch, teacher_seq_len]
            student_loss_mask: Mask for student response tokens [batch, student_seq_len]
            teacher_labels: Teacher labels with -100 for prompt [batch, teacher_seq_len]

        Returns:
            Per-token rewards [batch, loss_mask_seq_len] (negative of GOLD loss)
        """
        batch_size = student_logits.size(0)
        # Use loss_mask sequence length for rewards to ensure shape compatibility
        loss_mask_seq_len = student_loss_mask.size(1)
        device = student_logits.device

        # Initialize rewards tensor with loss_mask shape for compatibility
        rewards = torch.zeros(batch_size, loss_mask_seq_len, device=device)

        # Tracking metrics for logging
        alignment_successes = 0
        total_student_tokens = 0
        total_teacher_tokens = 0
        total_alignment_groups = 0
        tokens_per_student_group = []
        tokens_per_teacher_group = []
        sample_logged = False  # Log only first sample for debugging

        for i in range(batch_size):
            # Get student response region
            student_mask = student_loss_mask[i].bool()
            if not student_mask.any():
                continue

            student_positions = student_mask.nonzero(as_tuple=True)[0]
            student_start = student_positions[0].item()
            student_end = student_positions[-1].item() + 1

            # Get teacher response region (non -100 labels)
            teacher_mask = teacher_labels[i].ne(-100)
            if not teacher_mask.any():
                continue

            teacher_positions = teacher_mask.nonzero(as_tuple=True)[0]
            teacher_start = teacher_positions[0].item()
            teacher_end = teacher_positions[-1].item() + 1

            # Extract response logits
            student_resp_logits = student_logits[i, student_start:student_end]
            teacher_resp_logits = teacher_logits[i, teacher_start:teacher_end]

            # Convert to probabilities with temperature
            student_probs = F.softmax(student_resp_logits / self.gold_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_resp_logits / self.gold_temperature, dim=-1)

            # Get token IDs for alignment
            student_token_ids = student_input_ids[i, student_start:student_end].tolist()
            teacher_token_ids = teacher_input_ids[i, teacher_start:teacher_end].tolist()

            # Track token counts
            total_student_tokens += len(student_token_ids)
            total_teacher_tokens += len(teacher_token_ids)

            if len(student_token_ids) > 0 and len(teacher_token_ids) > 0:
                # Build alignment groups using greedy text matching
                student_groups, teacher_groups = build_alignment_groups_from_ids(
                    self.tokenizer, self.ref_tokenizer, student_token_ids, teacher_token_ids
                )

                if student_groups and teacher_groups:
                    # Track successful alignment
                    alignment_successes += 1
                    total_alignment_groups += len(student_groups)

                    # Track tokens per group
                    for s_group in student_groups:
                        tokens_per_student_group.append(len(s_group))
                    for t_group in teacher_groups:
                        tokens_per_teacher_group.append(len(t_group))

                    # Log sample alignment for first batch item (debugging)
                    if not sample_logged and i == 0:
                        self._log_alignment_sample(student_token_ids, teacher_token_ids, student_groups, teacher_groups)

                    # Merge probabilities for aligned spans
                    student_aligned = merge_probabilities_with_alignment_groups(student_probs, student_groups)
                    teacher_aligned = merge_probabilities_with_alignment_groups(teacher_probs, teacher_groups)

                    # Log probability comparison for first group of first sample
                    if not sample_logged and i == 0 and len(student_aligned) > 0:
                        self._log_probability_comparison(student_aligned[0], teacher_aligned[0])
                        sample_logged = True

                    # Compute per-group L1 loss and distribute to student tokens
                    min_groups = min(student_aligned.size(0), teacher_aligned.size(0))
                    for g in range(min_groups):
                        # Sort probabilities (ULD approach)
                        s_sorted = student_aligned[g].sort(descending=True).values
                        t_sorted = teacher_aligned[g].sort(descending=True).values

                        # Pad to same vocabulary size
                        max_vocab = max(s_sorted.size(0), t_sorted.size(0))
                        if s_sorted.size(0) < max_vocab:
                            s_sorted = F.pad(s_sorted, (0, max_vocab - s_sorted.size(0)))
                        if t_sorted.size(0) < max_vocab:
                            t_sorted = F.pad(t_sorted, (0, max_vocab - t_sorted.size(0)))

                        # Compute L1 loss for this group
                        group_loss = F.l1_loss(s_sorted, t_sorted, reduction="sum")

                        # Distribute loss to student tokens in this group as negative reward
                        # Lower loss = better match = higher reward
                        if g < len(student_groups):
                            for tok_idx in student_groups[g]:
                                abs_idx = student_start + tok_idx
                                if tok_idx < (student_end - student_start) and abs_idx < loss_mask_seq_len:
                                    rewards[i, abs_idx] = -group_loss / len(student_groups[g])
                else:
                    # Alignment failed - leave rewards as zeros for this sample
                    logger.warning(
                        f"GOLD alignment failed for batch {i}: empty alignment groups. "
                        f"Student tokens: {len(student_token_ids)}, Teacher tokens: {len(teacher_token_ids)}. "
                        "Rewards will be zero for this sample."
                    )
            else:
                # Should not happen due to earlier guards, but log if it does
                logger.warning(
                    f"GOLD alignment skipped for batch {i}: empty token IDs. "
                    f"Student tokens: {len(student_token_ids)}, Teacher tokens: {len(teacher_token_ids)}. "
                    "Rewards will be zero for this sample."
                )

        # Log alignment statistics
        if batch_size > 0:
            alignment_success_rate = alignment_successes / batch_size
            avg_student_tokens = total_student_tokens / batch_size
            avg_teacher_tokens = total_teacher_tokens / batch_size
            tokenizer_ratio = avg_teacher_tokens / avg_student_tokens if avg_student_tokens > 0 else 0.0
            avg_alignment_groups = total_alignment_groups / alignment_successes if alignment_successes > 0 else 0
            avg_tokens_per_student_group = (
                sum(tokens_per_student_group) / len(tokens_per_student_group) if tokens_per_student_group else 0
            )
            avg_tokens_per_teacher_group = (
                sum(tokens_per_teacher_group) / len(tokens_per_teacher_group) if tokens_per_teacher_group else 0
            )

            self.all_metrics.update(
                {
                    "gold/alignment_success_rate": alignment_success_rate,
                    "gold/avg_student_tokens": avg_student_tokens,
                    "gold/avg_teacher_tokens": avg_teacher_tokens,
                    "gold/tokenizer_ratio": tokenizer_ratio,
                    "gold/avg_alignment_groups": avg_alignment_groups,
                    "gold/avg_tokens_per_student_group": avg_tokens_per_student_group,
                    "gold/avg_tokens_per_teacher_group": avg_tokens_per_teacher_group,
                }
            )

        return rewards

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Calculate values, log probs, and prepare for GOLD reward computation.

        For cross-tokenizer distillation, this method also:
        1. Decodes student sequences to text
        2. Re-tokenizes with teacher tokenizer
        3. Stores necessary data for GOLD reward computation in apply_reward_kl_penalty

        For same-tokenizer case, uses the standard approach.
        """
        # Call parent implementation for standard log probs and values
        training_input = super().fwd_logprobs_values_reward(training_input)

        sequences = training_input["sequences"]
        attention_mask = training_input["attention_mask"]
        loss_mask = training_input["loss_mask"]

        # Decode student sequences to prompt and completion texts
        prompt_texts, completion_texts = self._decode_sequences_to_texts(sequences, attention_mask, loss_mask)

        # Re-tokenize with teacher tokenizer
        (
            teacher_input_ids,
            teacher_labels,
            teacher_attention_mask,
            teacher_prompt_length,
        ) = build_teacher_inputs_from_texts(
            self.ref_tokenizer,
            prompt_texts,
            completion_texts,
        )

        # Store teacher tokenization in metadata for use in apply_reward_kl_penalty
        if training_input.metadata is None:
            training_input.metadata = {}

        training_input.metadata["gold_teacher_input_ids"] = teacher_input_ids
        training_input.metadata["gold_teacher_labels"] = teacher_labels
        training_input.metadata["gold_teacher_attention_mask"] = teacher_attention_mask

        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """
        Computes rewards for distillation.

        For cross-tokenizer distillation (GOLD/ULD methodology):
        - Gets student and teacher logits
        - Aligns token spans between different tokenizations
        - Computes per-token rewards based on sorted probability matching

        For same-tokenizer distillation:
        - Uses standard KL penalty approach

        NOTE: For cross-tokenizer GOLD distillation to work properly, the model workers
        must be configured to return full logits instead of log_probs. If logits are
        not available, falls back to KL-based approach with a warning.
        """
        loss_masks_all: torch.Tensor = data["loss_mask"]

        # Get stored teacher tokenization from metadata
        teacher_input_ids = data.metadata["gold_teacher_input_ids"]
        teacher_labels = data.metadata["gold_teacher_labels"]
        teacher_attention_mask = data.metadata["gold_teacher_attention_mask"]

        # Move to appropriate device
        device = data["sequences"].device
        teacher_input_ids = teacher_input_ids.to(device)
        teacher_labels = teacher_labels.to(device)
        teacher_attention_mask = teacher_attention_mask.to(device)

        # Get student logits
        student_logits = self._get_student_logits(
            data["sequences"],
            data["attention_mask"],
        )

        # Get teacher logits on teacher-tokenized inputs
        teacher_logits = self._get_teacher_logits(
            teacher_input_ids,
            teacher_attention_mask,
        )

        # Check if we got full logits (required for GOLD/ULD)
        if student_logits is None or teacher_logits is None:
            raise RuntimeError(
                "GOLD distillation requires model workers to return full logits. "
                "Ensure you're using GOLDPolicyWorker and GOLDRefWorker. "
                "See gold_workers.py for reference."
            )
        else:
            # Proper GOLD/ULD methodology with full logits
            self.all_metrics.update({"gold/using_logits": 1.0})

            # Compute GOLD rewards
            rewards = self._compute_gold_rewards(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_input_ids=data["sequences"],
                teacher_input_ids=teacher_input_ids,
                student_loss_mask=loss_masks_all,
                teacher_labels=teacher_labels,
            )

            # Apply loss mask
            rewards = rewards * loss_masks_all

        # Log metrics
        if (loss_masks_all > 0).any():
            valid_rewards = rewards[loss_masks_all > 0]
            avg_reward = valid_rewards.mean().item()
            min_reward = valid_rewards.min().item()
            max_reward = valid_rewards.max().item()
            reward_std = valid_rewards.std().item()

            # Compute percentiles for reward distribution
            reward_percentile_25 = torch.quantile(valid_rewards, 0.25).item()
            reward_percentile_75 = torch.quantile(valid_rewards, 0.75).item()

            self.all_metrics.update(
                {
                    "gold/avg_reward": avg_reward,
                    "gold/min_reward": min_reward,
                    "gold/max_reward": max_reward,
                    "gold/reward_std": reward_std,
                    "gold/reward_percentile_25": reward_percentile_25,
                    "gold/reward_percentile_75": reward_percentile_75,
                }
            )
        else:
            self.all_metrics.update(
                {
                    "gold/avg_reward": 0.0,
                    "gold/min_reward": 0.0,
                    "gold/max_reward": 0.0,
                    "gold/reward_std": 0.0,
                    "gold/reward_percentile_25": 0.0,
                    "gold/reward_percentile_75": 0.0,
                }
            )

        data["rewards"] = rewards

        return data


# TODO: Is this accurate for GOLD? Taken from on-policy distillation example.
@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


# TODO: Is this accurate for GOLD? Taken from on-policy distillation example.
@register_policy_loss("importance_sampling")
def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    # as defined here: https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling
    loss = -torch.exp(log_probs - old_log_probs) * advantages
    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    return loss, 0.0


class GOLDDistillationExp(BasePPOExp):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.ref_tokenizer = self.get_tokenizer(model_path=cfg.trainer.ref.model.path)

    def get_trainer(self, *args, **kwargs):
        return GOLDDistillationTrainer(*args, **kwargs)

    def get_tokenizer(self, padding_side="left", model_path=None):
        """Same as get_tokenizer in BasePPOExp, but allows for a different model path"""
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
        Largely a copy of _setup_trainer in BasePPOExp, but creates a separate tokenizer
        for the ref model (teacher) to support cross-tokenizer distillation.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "megatron":
            # Megatron support would require another worker like GOLDRefWorker and GOLDPolicyWorker for FSDP
            raise NotImplementedError("Megatron not supported in this implementation")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        ref_tokenizer = self.ref_tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)

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

        # Build the models with GOLD-specific workers that return logits
        trainer.build_models(GOLDPolicyWorker, CriticWorker, GOLDRefWorker)
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = GOLDDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
