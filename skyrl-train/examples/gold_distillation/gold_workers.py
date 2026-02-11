"""
GOLD Workers that return logits instead of log probabilities.

These custom workers are needed for cross-tokenizer distillation because:
1. GOLD/ULD loss requires full logits for sorted probability comparison
2. The teacher model may use a different tokenizer, so we need to handle
   differently tokenized inputs

Both workers extend their FSDP base classes to inherit FSDP-specific functionality
while overriding _forward_micro_batch to return logits instead of log_probs.
"""

import ray
import torch

from skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase, FSDPRefWorkerBase
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch


def _forward_micro_batch_logits(model, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
    """
    Shared forward pass implementation that returns logits instead of log probs.

    Args:
        model: HFModelWrapper containing the HuggingFace model
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
        # model is HFModelWrapper, model.model is the actual HF model
        output = model.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        logits = output["logits"]  # [batch, seq_len, vocab_size]

    logits = logits.to("cpu")
    return TrainingOutputBatch({"output": logits}, metadata=micro_batch.metadata)


class GOLDFSDPPolicyWorkerBase(FSDPPolicyWorkerBase):
    """
    FSDP-compatible policy worker that returns full logits instead of log probabilities.

    This is required for GOLD (General On-policy Logit Distillation) because
    the ULD loss needs to compare sorted probability distributions across
    different vocabulary sizes.

    Inherits FSDP functionality from FSDPPolicyWorkerBase:
    - offload_to_cpu / backload_to_gpu for memory management
    - init_model for FSDP-wrapped model initialization
    - forward with resharding after forward pass
    - ppo_train for policy gradient updates
    """

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        """Forward pass that returns logits instead of log probs."""
        return _forward_micro_batch_logits(self.model, micro_batch)


class GOLDFSDPRefWorkerBase(FSDPRefWorkerBase):
    """
    FSDP-compatible reference worker that returns full logits instead of log probabilities.

    This is required for GOLD (General On-policy Logit Distillation) because
    the ULD loss needs to compare sorted probability distributions across
    different vocabulary sizes.

    Inherits FSDP functionality from FSDPRefWorkerBase:
    - offload_to_cpu / backload_to_gpu for memory management
    - init_model for FSDP-wrapped model initialization
    - forward with resharding after forward pass
    """

    def _forward_micro_batch(self, micro_batch: TrainingInputBatch) -> TrainingOutputBatch:
        """Forward pass that returns logits instead of log probs."""
        return _forward_micro_batch_logits(self.model, micro_batch)


# Ray remote wrappers
GOLDPolicyWorker = ray.remote(num_gpus=1)(GOLDFSDPPolicyWorkerBase)
GOLDRefWorker = ray.remote(num_gpus=1)(GOLDFSDPRefWorkerBase)
