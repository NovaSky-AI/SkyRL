"""
GOLD (General On-policy Logit Distillation) for Cross-Tokenizer Distillation.

This module provides utilities and trainers for distilling knowledge from
teacher models with different tokenizers to student models.
"""

from .gold_utils import (
    ULDLoss,
    build_teacher_inputs_from_texts,
    build_alignment_groups_from_ids,
    merge_probabilities_with_alignment_groups,
    generalized_jsd_loss,
    compute_per_token_gold_loss,
)

from .gold_ref_worker import (
    GOLDRefWorkerBase,
    forward_with_logits,
    TeacherInputBatch,
)

from .main_gold_distill import (
    GOLDDistillationTrainer,
    GOLDDistillationExp,
)

__all__ = [
    # Utilities
    "ULDLoss",
    "build_teacher_inputs_from_texts",
    "build_alignment_groups_from_ids",
    "merge_probabilities_with_alignment_groups",
    "generalized_jsd_loss",
    "compute_per_token_gold_loss",
    # Ref worker
    "GOLDRefWorkerBase",
    "forward_with_logits",
    "TeacherInputBatch",
    # Trainer
    "GOLDDistillationTrainer",
    "GOLDDistillationExp",
]
