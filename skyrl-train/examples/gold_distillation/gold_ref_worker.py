"""
GOLD Reference Worker that returns logits instead of log probabilities.

This custom ref worker is needed for cross-tokenizer distillation because:
1. GOLD/ULD loss requires full logits for sorted probability comparison
2. The teacher model may use a different tokenizer, so we need to handle
   differently tokenized inputs

Adapted from the base RefWorkerBase in skyrl_train/workers/worker.py.
"""

import torch
import torch.nn as nn

from skyrl_train.workers.worker import RefWorkerBase
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch


class GOLDRefWorkerBase(RefWorkerBase):
    """
    Reference worker that returns full logits instead of log probabilities.

    This is required for GOLD (General On-policy Logit Distillation) because
    the ULD loss needs to compare sorted probability distributions across
    different vocabulary sizes.
    """

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        """
        Forward pass that returns logits instead of log probs.

        Args:
            micro_batch: Input batch containing sequences, attention_mask, etc.

        Returns:
            TrainingOutputBatch with "output" containing logits [batch, seq_len, vocab_size]
        """
        device = torch.cuda.current_device()
        micro_batch.to(device)
        sequences = micro_batch["sequences"]
        attention_mask = micro_batch["attention_mask"]

        # Compute position_ids from attention_mask
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Call the underlying HuggingFace model directly to get logits
            # self.model is HFModelWrapper, self.model.model is the actual HF model
            if hasattr(self.model, "model"):
                # HFModelWrapper case
                output = self.model.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
            else:
                # Direct HF model case
                output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

            logits = output["logits"]  # [batch, seq_len, vocab_size]

        logits = logits.to("cpu")
        output_batch = TrainingOutputBatch(
            {"output": logits, "return_type": "logits"},
        )
        output_batch.metadata = micro_batch.metadata
        return output_batch


def forward_with_logits(
    model: nn.Module,
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Helper function to run forward pass and get logits directly.

    Args:
        model: HFModelWrapper or raw HuggingFace model
        sequences: Input token IDs [batch, seq_len]
        attention_mask: Attention mask [batch, seq_len]
        temperature: Temperature for scaling logits (default: 1.0)

    Returns:
        Logits tensor [batch, seq_len, vocab_size]
    """
    # Compute position_ids from attention_mask
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)

    with torch.no_grad(), torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
        # Get the underlying HF model
        if hasattr(model, "model"):
            hf_model = model.model
        else:
            hf_model = model

        output = hf_model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        logits = output["logits"]

        # Apply temperature scaling if needed
        if temperature != 1.0:
            logits = logits / temperature

    return logits


class TeacherInputBatch:
    """
    Helper class to hold teacher-tokenized inputs alongside student inputs.

    This is used when the teacher model uses a different tokenizer than the student.
    """

    def __init__(
        self,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        teacher_labels: torch.Tensor,
        teacher_prompt_length: int,
        prompt_texts: list[str],
        completion_texts: list[str],
    ):
        self.teacher_input_ids = teacher_input_ids
        self.teacher_attention_mask = teacher_attention_mask
        self.teacher_labels = teacher_labels
        self.teacher_prompt_length = teacher_prompt_length
        self.prompt_texts = prompt_texts
        self.completion_texts = completion_texts

    def to(self, device):
        """Move tensors to specified device."""
        self.teacher_input_ids = self.teacher_input_ids.to(device)
        self.teacher_attention_mask = self.teacher_attention_mask.to(device)
        self.teacher_labels = self.teacher_labels.to(device)
        return self
