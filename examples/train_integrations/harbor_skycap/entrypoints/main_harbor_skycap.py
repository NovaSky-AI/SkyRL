"""Train on Harbor tasks, with skycap capturing each rollout's exact tokens.

The sibling ``main_harbor`` with two changes: a pool of skycap servers (Ray
actors, see ``servers.py``) starts in front of the inference router, and the
generator points each trial at its own trajectory on one of them.

    uv run --isolated --extra fsdp --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \\
        trainer.policy.model.path=Qwen/Qwen3-8B generator.inference_engine.served_model_name=policy \\
        generator.step_wise_trajectories=true data.train_data="['/path/to/harbor/tasks']" ...
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

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
from ..servers import SkycapServers, start_servers


@dataclass
class SkycapConfig:
    num_servers: int = 1
    """skycap servers, one Ray actor each. The generator spreads trajectories over them round-robin."""
    num_cpus_per_server: float = 2.0
    """CPUs reserved per server, for rendering and HTTP."""
    placement_strategy: str = "SPREAD"
    """Ray placement-group strategy for the servers: SPREAD (default, for availability), STRICT_SPREAD, PACK."""
    record_dir: Optional[str] = None
    """Where ended trajectories are written. Defaults to ``{trainer.export_path}/skycap``; each server writes
    on its own node, so point it at a shared filesystem to have one directory for the run."""
    ttl: float = 3600.0
    """Seconds an open trajectory may be idle before skycap writes it as abandoned and releases it."""
    renderer_pool_size: int = 8
    """Renderers (tokenizer copies) each server renders prompts with in parallel."""


@dataclass
class HarborSkycapConfig(HarborSkyRLConfig):
    skycap: SkycapConfig = field(default_factory=SkycapConfig)


def start_skycap(cfg: Any, engine_url: str) -> SkycapServers:
    """skycap servers in token mode, in front of SkyRL's router."""
    ie = cfg.generator.inference_engine
    sampling = cfg.generator.sampling_params
    engine_init = dict(ie.engine_init_kwargs or {})
    settings = {
        "engine_url": engine_url,
        "tokenizer": cfg.trainer.policy.model.path,
        "renderer_pool_size": cfg.skycap.renderer_pool_size,
        "model": ie.served_model_name,
        "max_model_len": engine_init.get("max_model_len") or cfg.trainer.algorithm.max_seq_len,
        # The trainer computes logprobs with these, so every rollout is sampled with them,
        # whatever the harness asks for.
        "sampling_overrides": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
        },
        "sampling_mask": ie.enable_return_sample_support_set,
    }
    return start_servers(
        settings,
        num_servers=cfg.skycap.num_servers,
        num_cpus_per_server=cfg.skycap.num_cpus_per_server,
        placement_strategy=cfg.skycap.placement_strategy,
        record_dir=cfg.skycap.record_dir or os.path.join(cfg.trainer.export_path, "skycap"),
        ttl=cfg.skycap.ttl,
    )


class HarborSkycapExp(HarborExp):
    skycap: Optional[SkycapServers] = None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if self.skycap is None:
            self.skycap = start_skycap(cfg, inference_engine_client.get_endpoint_url())
        return HarborSkycapGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            capture_urls=self.skycap.urls,
            inference_engine_client=inference_engine_client,
        )

    def run(self):
        try:
            super().run()
        finally:
            if self.skycap is not None:
                logger.info("stopping skycap, writing the trajectories still in memory")
                self.skycap.stop()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
