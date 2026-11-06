"""A Harbor run's config: SkyRL's, plus the Harbor ``TrialConfig`` template each trial is built from."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig
from skyrl.train.utils.rate_limiter import RateLimiterConfig

# NOTE (sumanthrh): We use a YAML to store the defaults for the Harbor trial configuration
# TODO: Convert to a dataclass
HARBOR_DEFAULT_CONFIG = Path(__file__).parent / "default_trial_config.yaml"


def deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_trial_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Harbor's defaults (``HARBOR_DEFAULT_CONFIG``) with ``overrides`` (``harbor_trial_config`` from the CLI) on top."""
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f) or {}
    return deep_merge(defaults, overrides)


@dataclass
class HarborGeneratorConfig(GeneratorConfig):
    """GeneratorConfig with Harbor-specific rate limiting."""

    rate_limit: RateLimiterConfig = field(default_factory=RateLimiterConfig)


@dataclass
class HarborSkyRLConfig(SkyRLTrainConfig):
    """SkyRLTrainConfig with Harbor trial configuration."""

    harbor_trial_config: Dict[str, Any] = field(default_factory=dict)
    generator: HarborGeneratorConfig = field(default_factory=HarborGeneratorConfig)
