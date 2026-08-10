"""Training entrypoint for the Harbor TITO example."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, cast

import ray
import yaml

from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborGeneratorConfig,
    HarborSkyRLConfig,
    _deep_merge,
)
from examples.train_integrations.harbor_tito.harbor_generator import TITOHarborGenerator
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

TITO_HARBOR_OVERRIDES = Path(__file__).parent.parent / "harbor_trial_config" / "tito.yaml"


@dataclass
class TITOHarborGeneratorConfig(HarborGeneratorConfig):
    """Harbor generator configuration with optional trace parity checks."""

    tito_validate_rollout_details: bool = True
    tito_trace_log_dir: Optional[str] = None
    tito_renderer_config: Optional[Dict[str, Any]] = None


@dataclass
class TITOHarborSkyRLConfig(HarborSkyRLConfig):
    """SkyRL configuration for the dedicated Harbor TITO example."""

    generator: TITOHarborGeneratorConfig = field(default_factory=TITOHarborGeneratorConfig)


def _load_tito_harbor_config(overrides: dict) -> dict:
    with open(HARBOR_DEFAULT_CONFIG) as config_file:
        defaults = yaml.safe_load(config_file)
    with open(TITO_HARBOR_OVERRIDES) as config_file:
        tito_defaults = yaml.safe_load(config_file)
    return _deep_merge(_deep_merge(defaults, tito_defaults), overrides)


class TITOHarborExp(HarborExp):
    """Use the TITO Harbor generator with the baseline Harbor datasets."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return TITOHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    TITOHarborExp(cfg).run()


def main() -> None:
    cfg = cast(TITOHarborSkyRLConfig, TITOHarborSkyRLConfig.from_cli_overrides(sys.argv[1:]))
    cfg.harbor_trial_config = _load_tito_harbor_config(cfg.harbor_trial_config)
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set for Harbor TITO training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
