"""Generator config extensions for the Recursive Language Model (RLM) environment.

These fields are RLM-specific and live outside the base ``GeneratorConfig`` so that
non-RLM training runs do not surface them. Wired into a ``SkyRLTrainConfig`` via
``make_config(generator_cls=RLMGeneratorConfig)`` in the RLM entry points.
"""

from dataclasses import dataclass
from typing import Optional

from skyrl.train.config import GeneratorConfig


@dataclass
class RLMGeneratorConfig(GeneratorConfig):
    train_child_trajectories: bool = False
    """Include child RLM agent trajectories in the training batch, with reward propagated from the parent."""
    enable_child_agents: bool = True
    """When False, skip subcall_fn injection for RLM envs so the top-level agent runs without
    child-spawning capability (single-paper mode)."""
    frozen_openrouter_model: Optional[str] = None
    """When set, in-REPL ``llm_query`` calls use this frozen model via OpenRouter
    instead of the policy engine."""
