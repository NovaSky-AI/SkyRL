"""GOLD-specific config extensions.

Subclasses the core SkyRL config to add GOLD distillation fields
without modifying the main config.py.
"""

from dataclasses import dataclass, field
from typing import Optional

from skyrl.train.config.config import (
    AlgorithmConfig,
    TrainerConfig,
    SkyRLTrainConfig,
)


@dataclass
class GOLDAlgorithmConfig(AlgorithmConfig):
    """AlgorithmConfig extended with GOLD distillation knobs."""

    gold_teacher_model_path: Optional[str] = None
    """Path to the teacher model for GOLD distillation. If ``None``, uses ``ref.model.path``."""
    gold_student_temperature: float = 1.0
    """Temperature applied to student logits before softmax in GOLD loss."""
    gold_teacher_temperature: float = 1.0
    """Temperature applied to teacher logits before softmax in GOLD loss."""
    gold_crossentropy_weight: float = 0.0
    """Weight for the cross-entropy loss component in GOLD hybrid loss."""
    gold_distillation_weight: float = 1.0
    """Weight for the distillation (JSD + L1) loss component in GOLD."""
    gold_matched_weight: float = 1.0
    """Weight for the JSD loss on matched (shared vocabulary) tokens."""
    gold_unmatched_weight: float = 1.0
    """Weight for the L1 loss on unmatched (disjoint vocabulary) tokens."""
    gold_beta: float = 0.0
    """JSD interpolation parameter: 0=forward KL, 1=reverse KL, 0.5=symmetric JSD."""


@dataclass
class GOLDTrainerConfig(TrainerConfig):
    """TrainerConfig with GOLD algorithm config."""

    algorithm: GOLDAlgorithmConfig = field(default_factory=GOLDAlgorithmConfig)


@dataclass
class GOLDSkyRLTrainConfig(SkyRLTrainConfig):
    """Top-level config for GOLD distillation experiments."""

    trainer: GOLDTrainerConfig = field(default_factory=GOLDTrainerConfig)
