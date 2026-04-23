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
    child-spawning capability (single-paper mode). Also uses the shorter ``repl_timeout`` instead
    of ``parent_repl_timeout``."""
    judge_reward_model: Optional[str] = None
    """When set, replace F1 reward with an LLM judge score (via OpenAI-compatible API) for RLM envs.
    Requires ``OPENAI_API_KEY``. E.g. 'openai/gpt-4.1-nano'."""
    judge_reward_base_url: str = "https://api.openai.com/v1"
    """Base URL for the OpenAI-compatible endpoint used by the judge reward model."""
    child_openrouter_model: Optional[str] = None
    """When set, child RLM agents (depth >= 1) use this model via an OpenAI-compatible API instead of
    the policy inference engine. Requires ``OPENROUTER_API_KEY``."""
    child_openrouter_base_url: str = "https://openrouter.ai/api/v1"
    """Base URL for the OpenAI-compatible chat completions endpoint used by child agents."""
