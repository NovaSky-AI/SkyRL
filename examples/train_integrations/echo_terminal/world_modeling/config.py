from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from skyrl.train.config import AlgorithmConfig, DataConfig, SkyRLTrainConfig, TrainerConfig


@dataclass
class EchoDataConfig(DataConfig):
    train_data: List[str | dict[str, Any]] = field(default_factory=list)
    val_data: List[str | dict[str, Any]] = field(default_factory=list)


@dataclass
class EchoAlgorithmConfig(AlgorithmConfig):
    world_model_coeff: float = 0.0
    world_loss_normalization: Literal["selected_tokens", "full_observation_tokens"] = "full_observation_tokens"
    world_model_coeff_schedule: Literal["constant", "linear_decay", "step_decay"] = "constant"
    world_model_coeff_end: float = 0.0
    world_model_coeff_transition_step: int = 0
    world_model_coeff_steps: Optional[List[int]] = None
    world_model_coeff_values: Optional[List[float]] = None
    wm_filter_min_valid_tool_call_pct: Optional[float] = None
    wm_filter_min_parse_clean_pct: Optional[float] = None
    wm_filter_min_correct_pct: Optional[float] = None


@dataclass
class EchoTrainerConfig(TrainerConfig):
    algorithm: EchoAlgorithmConfig = field(default_factory=EchoAlgorithmConfig)


@dataclass
class EchoSkyRLTrainConfig(SkyRLTrainConfig):
    data: EchoDataConfig = field(default_factory=EchoDataConfig)
    trainer: EchoTrainerConfig = field(default_factory=EchoTrainerConfig)
