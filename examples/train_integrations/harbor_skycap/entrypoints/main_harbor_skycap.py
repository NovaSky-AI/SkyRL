"""Train on Harbor tasks, with skycap capturing each rollout's exact tokens.

The sibling ``main_harbor`` with two changes: a pool of skycap servers (Ray
actors, see ``skyrl.integration.skycap.servers``) starts in front of the inference router, and the
generator points each trial at its own trajectory on one of them. Agents that
run inside their sandbox (mini-swe-agent, ...) reach it through
``skycap.exposure``.

    uv run --isolated --extra fsdp --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \\
        trainer.policy.model.path=Qwen/Qwen3-8B generator.inference_engine.served_model_name=policy \\
        generator.step_wise_trajectories=true data.train_data="['/path/to/harbor/tasks']" ...
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import ray
import yaml
from loguru import logger

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from ..harbor_generator import HarborSkycapGenerator
from skyrl.integration.skycap.record_index import RecordLog, SkycapRecordIndex
from skyrl.integration.skycap.config import SkycapConfig, exposure_type, start_skycap
from skyrl.integration.skycap.servers import SkycapServers


@dataclass
class HarborSkycapConfig(HarborSkyRLConfig):
    skycap: SkycapConfig = field(default_factory=SkycapConfig)


class HarborSkycapExp(HarborExp):
    skycap: Optional[SkycapServers] = None
    generator: Optional[HarborSkycapGenerator] = None
    records: Optional[RecordLog] = None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if self.skycap is None:
            self.skycap = start_skycap(cfg, inference_engine_client.get_endpoint_url())
        if self.records is None and cfg.skycap.wandb.enabled and cfg.trainer.logger == "wandb":
            self.records = RecordLog()
        self.generator = HarborSkycapGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            capture_urls=self.skycap.urls,
            inference_engine_client=inference_engine_client,
            train_paths=cfg.skycap.train_paths,
            records=self.records,
            include_images=cfg.skycap.include_images,
        )
        return self.generator

    def get_trainer(self, *args, **kwargs):
        trainer = super().get_trainer(*args, **kwargs)
        if self.records is not None:
            trainer.add_callback(SkycapRecordIndex(self.records, self.cfg.skycap.wandb.phases))
        return trainer

    def run(self):
        try:
            super().run()
        finally:
            if self.skycap is not None:
                logger.info("stopping skycap, writing the trajectories still in memory")
                self.skycap.stop()
            if self.generator is not None:
                asyncio.run(self.generator.close())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    exposure_type(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
