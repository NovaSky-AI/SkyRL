"""
GOLD Workers for supervised cross-tokenizer distillation.

The key architectural decision: the teacher model is loaded directly inside the
policy worker. During each micro-batch training step, the teacher runs forward
(no grad, tiny memory) and the student runs forward (with grad), then the GOLD
loss is computed directly. This mirrors TRL's approach and avoids the impractical
alternative of passing full teacher logits (~15GB+ per batch) through the data pipeline.

Teacher tokenization data (input_ids, labels, attention_mask) is stored as proper
tensors in the TrainingInputBatch dict so they get automatically sliced/chunked
alongside the student data. We override forward_backward to avoid the standard
BatchIterator → Experience conversion, which would drop these extra keys.
"""

from collections import defaultdict

import ray
import torch
from loguru import logger
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPRefWorkerBase
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import all_reduce_metrics, reduce_metrics

from .gold_utils import (
    build_alignment_groups_from_ids,
    compute_vocabulary_mapping,
    compute_gold_loss,
)


class GOLDFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    """
    FSDP policy worker for GOLD supervised distillation.

    Loads the teacher model alongside the student model. During training,
    computes GOLD loss directly from both models' logits in a single
    micro-batch step.
    """

    def init_model(self, model_path, num_training_steps: int = None):
        """Initialize student model via super(), then load teacher model and compute vocab mapping."""
        # Initialize student model (FSDP-wrapped)
        super().init_model(model_path, num_training_steps=num_training_steps)

        # Load teacher model path from config
        teacher_model_path = getattr(self.cfg.algorithm, "gold_teacher_model_path", None)
        if teacher_model_path is None:
            teacher_model_path = self.cfg.ref.model.path
        logger.info(f"GOLD: Loading teacher model from {teacher_model_path}")

        # Load teacher tokenizer and student tokenizer
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id

        self.student_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
            self.student_tokenizer.pad_token_id = self.student_tokenizer.eos_token_id

        # Load teacher model (frozen, not FSDP-wrapped — replicated on each GPU)
        device = torch.cuda.current_device()
        self.teacher_model = HFModelWrapper(
            teacher_model_path,
            use_flash_attention_2=self.cfg.flash_attn,
            bf16=True,
            model_config_kwargs=self.cfg.ref.model_config_kwargs,
        )
        self.teacher_model.to(device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

        # Compute vocabulary mapping once
        logger.info("GOLD: Computing vocabulary mapping between student and teacher tokenizers...")
        s_matched, t_matched, s_unmatched, t_unmatched = compute_vocabulary_mapping(
            self.student_tokenizer, self.teacher_tokenizer
        )
        self.student_matched_indices = torch.tensor(s_matched, dtype=torch.long, device=device)
        self.teacher_matched_indices = torch.tensor(t_matched, dtype=torch.long, device=device)
        self.student_unmatched_indices = torch.tensor(s_unmatched, dtype=torch.long, device=device)
        self.teacher_unmatched_indices = torch.tensor(t_unmatched, dtype=torch.long, device=device)
        logger.info(
            f"GOLD: Vocab mapping computed — {len(s_matched)} matched, "
            f"{len(s_unmatched)} student unmatched, {len(t_unmatched)} teacher unmatched"
        )

    def forward_backward(self, data: TrainingInputBatch, loss_fn=None, loss_fn_config=None):
        """
        Override forward_backward to work directly with TrainingInputBatch chunks.

        The standard path uses BatchIterator which converts to Experience objects,
        dropping extra dict keys (gold_teacher_*). We chunk the TrainingInputBatch
        directly to preserve all tensor keys.
        """
        micro_batch_size = self.cfg.micro_train_batch_size_per_gpu
        all_metrics = defaultdict(list)

        # Chunk the TrainingInputBatch directly (preserves all tensor keys)
        micro_batches = data.chunk(micro_batch_size)

        for micro_batch in micro_batches:
            metrics = self._gold_forward_backward_micro(micro_batch)
            self._micro_batches_accumulated += 1

            for k, v in metrics.items():
                all_metrics[k].append(v)

        return reduce_metrics(dict(all_metrics))

    def _gold_forward_backward_micro(self, micro_batch: TrainingInputBatch):
        """
        Custom forward-backward for GOLD supervised distillation.

        Instead of the standard RL policy loss, this:
        1. Extracts teacher tokenization data from the batch (properly chunked)
        2. Runs teacher forward (no grad) to get teacher logits
        3. Runs student forward (with grad) to get student logits
        4. Computes GOLD loss (JSD on matched + L1 on unmatched vocab)
        5. Backpropagates through student model
        """
        self.model.train()
        device = torch.cuda.current_device()
        micro_batch.to(device)

        # Extract config
        cfg = self.cfg.algorithm
        student_temperature = getattr(cfg, "gold_student_temperature", 1.0)
        teacher_temperature = getattr(cfg, "gold_teacher_temperature", 1.0)
        beta = getattr(cfg, "gold_beta", 0.0)
        matched_weight = getattr(cfg, "gold_matched_weight", 1.0)
        unmatched_weight = getattr(cfg, "gold_unmatched_weight", 1.0)
        distillation_weight = getattr(cfg, "gold_distillation_weight", 1.0)
        crossentropy_weight = getattr(cfg, "gold_crossentropy_weight", 0.0)

        # Get teacher data (auto-chunked along with student data)
        gold_teacher_input_ids = micro_batch["gold_teacher_input_ids"]
        gold_teacher_attention_mask = micro_batch["gold_teacher_attention_mask"]
        gold_teacher_labels = micro_batch["gold_teacher_labels"]

        # Get student data
        sequences = micro_batch["sequences"]
        attention_mask = micro_batch["attention_mask"]
        loss_mask = micro_batch["loss_mask"]
        num_actions = micro_batch.metadata["response_length"]

        # Compute prompt offset (student sequences include prompt + response)
        prompt_offset = sequences.size(1) - loss_mask.size(1)

        batch_size = sequences.size(0)
        total_loss = torch.zeros((), device=device, dtype=torch.float32)
        batch_metrics = {
            "jsd_loss": 0.0,
            "l1_loss": 0.0,
            "ce_loss": 0.0,
            "total_loss": 0.0,
            "alignment_groups": 0.0,
            "alignment_success_rate": 0.0,
        }
        valid_samples = 0

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_position_ids = gold_teacher_attention_mask.long().cumsum(-1) - 1
                teacher_position_ids.masked_fill_(gold_teacher_attention_mask == 0, 1)
                teacher_output = self.teacher_model.model(
                    gold_teacher_input_ids,
                    attention_mask=gold_teacher_attention_mask,
                    position_ids=teacher_position_ids,
                )
                teacher_logits_full = teacher_output["logits"]  # [B, T_seq, T_vocab]

            # Student forward (with grad)
            student_position_ids = attention_mask.long().cumsum(-1) - 1
            student_position_ids.masked_fill_(attention_mask == 0, 1)
            student_output = self.model.model(
                sequences,
                attention_mask=attention_mask,
                position_ids=student_position_ids,
            )
            student_logits_full = student_output["logits"]  # [B, S_seq, S_vocab]

            # Compute GOLD loss per sample in the batch
            for i in range(batch_size):
                # Get student response region
                student_mask = loss_mask[i].bool()
                if not student_mask.any():
                    continue

                student_positions = student_mask.nonzero(as_tuple=True)[0]
                student_start = student_positions[0].item()
                student_end = student_positions[-1].item() + 1

                # Get teacher response region
                teacher_mask = gold_teacher_labels[i].ne(-100)
                if not teacher_mask.any():
                    continue

                teacher_positions = teacher_mask.nonzero(as_tuple=True)[0]
                teacher_start = teacher_positions[0].item()
                teacher_end = teacher_positions[-1].item() + 1

                # Extract response logits
                student_resp_logits = student_logits_full[
                    i, prompt_offset + student_start : prompt_offset + student_end
                ]
                teacher_resp_logits = teacher_logits_full[i, teacher_start:teacher_end].detach()

                # Get token IDs for alignment
                student_token_ids = sequences[
                    i, prompt_offset + student_start : prompt_offset + student_end
                ].tolist()
                teacher_token_ids = gold_teacher_input_ids[i, teacher_start:teacher_end].tolist()

                if not student_token_ids or not teacher_token_ids:
                    continue

                # Build alignment groups
                student_groups, teacher_groups = build_alignment_groups_from_ids(
                    self.student_tokenizer, self.teacher_tokenizer,
                    student_token_ids, teacher_token_ids,
                )

                if not student_groups or not teacher_groups:
                    continue

                valid_samples += 1
                batch_metrics["alignment_groups"] += len(student_groups)

                # Get student labels for optional CE loss
                student_labels = None
                if crossentropy_weight > 0.0:
                    student_labels = sequences[
                        i, prompt_offset + student_start + 1 : prompt_offset + student_end + 1
                    ]

                # Compute GOLD loss for this sample
                sample_loss, sample_metrics = compute_gold_loss(
                    student_logits=student_resp_logits,
                    teacher_logits=teacher_resp_logits,
                    student_alignment_groups=student_groups,
                    teacher_alignment_groups=teacher_groups,
                    student_matched_indices=self.student_matched_indices,
                    teacher_matched_indices=self.teacher_matched_indices,
                    student_unmatched_indices=self.student_unmatched_indices,
                    teacher_unmatched_indices=self.teacher_unmatched_indices,
                    student_labels=student_labels,
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                    beta=beta,
                    matched_weight=matched_weight,
                    unmatched_weight=unmatched_weight,
                    distillation_weight=distillation_weight,
                    crossentropy_weight=crossentropy_weight,
                )

                total_loss = total_loss + sample_loss
                for k, v in sample_metrics.items():
                    batch_metrics[k] += v

        # Average over valid samples
        if valid_samples > 0:
            total_loss = total_loss / valid_samples
            for k in ["jsd_loss", "l1_loss", "ce_loss", "total_loss", "alignment_groups"]:
                batch_metrics[k] /= valid_samples
            batch_metrics["alignment_success_rate"] = valid_samples / batch_size

        # Backward
        self.strategy.backward(total_loss, self.model, self.optimizer)

        # Build status dict
        status = {
            "gold/loss": total_loss.item(),
            "gold/jsd_loss": batch_metrics["jsd_loss"],
            "gold/l1_loss": batch_metrics["l1_loss"],
            "gold/ce_loss": batch_metrics["ce_loss"],
            "gold/alignment_groups": batch_metrics["alignment_groups"],
            "gold/alignment_success_rate": batch_metrics["alignment_success_rate"],
            "response_length": num_actions,
            "policy_lr": self.scheduler.get_last_lr()[0],
        }

        # All-reduce metrics across DP workers
        status = all_reduce_metrics(status, self.strategy)
        return status


class GOLDFSDPRefWorkerBase(FSDPRefWorkerBase):
    """
    Ref worker for GOLD distillation.

    Kept for pipeline compatibility — the trainer still builds a ref model group,
    but it is not used during training (teacher runs inside the policy worker).
    The forward pass returns logits for any auxiliary use (e.g., KL regularization).
    """

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        """Forward pass that returns logits instead of log probs."""
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        attention_mask = micro_batch["attention_mask"]

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            output = self.model.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
            logits = output["logits"]

        logits = logits.to("cpu")
        output_batch = TrainingOutputBatch({"output": logits})
        output_batch.metadata = micro_batch.metadata
        return output_batch


# Ray remote wrappers
GOLDPolicyWorker = ray.remote(num_gpus=1)(GOLDFSDPPolicyWorkerBase)
GOLDRefWorker = ray.remote(num_gpus=1)(GOLDFSDPRefWorkerBase)
