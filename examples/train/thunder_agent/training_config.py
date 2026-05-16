"""
ThunderAgent integration config for SkyRL.

Extends InferenceEngineConfig with ThunderAgent-specific scheduling options,
then wires through the Harbor training config.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from skyrl.train.config import GeneratorConfig, InferenceEngineConfig, make_config
from skyrl.train.utils.rate_limiter import RateLimiterConfig


@dataclass
class ThunderAgentInferenceEngineConfig(InferenceEngineConfig):
    """InferenceEngineConfig extended with ThunderAgent scheduling options."""

    thunder_agent_mode: str = "tr"
    """ThunderAgent router mode: 'default' (pure proxy) or 'tr' (capacity scheduling)."""
    thunder_agent_acting_token_weight: float = 1.0
    """Weight for acting tokens in capacity calculation."""
    thunder_agent_scheduler_interval: float = 5.0
    """Interval in seconds between scheduler checks."""
    thunder_agent_use_acting_token_decay: bool = False
    """Use exponential decay for acting tokens in resume logic."""
    thunder_agent_profile_enabled: bool = False
    """Enable per-program profiling."""
    thunder_agent_metrics_enabled: bool = False
    """Enable backend metrics monitoring."""
    thunder_agent_metrics_interval: float = 5.0
    """Interval in seconds between metrics fetches."""


@dataclass
class ThunderAgentGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with ThunderAgent-aware inference engine config."""

    inference_engine: ThunderAgentInferenceEngineConfig = field(default_factory=ThunderAgentInferenceEngineConfig)


_ThunderAgentBaseConfig = make_config(generator_cls=ThunderAgentGeneratorConfig)


@dataclass
class ThunderAgentHarborGeneratorConfig(ThunderAgentGeneratorConfig):
    """GeneratorConfig with Harbor rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class ThunderAgentHarborConfig(_ThunderAgentBaseConfig):
    """Config for Harbor + ThunderAgent fully-async training."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: ThunderAgentHarborGeneratorConfig = field(default_factory=ThunderAgentHarborGeneratorConfig)
    max_train_tasks: Optional[int] = None
    max_eval_tasks: Optional[int] = None
