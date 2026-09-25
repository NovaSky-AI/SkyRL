"""Train on Harbor tasks, with skycap capturing each rollout's exact tokens.

The sibling ``main_harbor`` with two changes: a skycap server starts in this
process in front of the inference router, and the generator points each trial
at its own trajectory on it.

    uv run --isolated --extra fsdp --extra harbor-skycap \\
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

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    get_node_ip,
)
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from ...harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from ..harbor_generator import HarborSkycapGenerator


@dataclass
class SkycapConfig:
    record_dir: Optional[str] = None
    """Where ended trajectories are written. Defaults to ``{trainer.export_path}/skycap``."""
    ttl: float = 3600.0
    """Seconds an open trajectory may be idle before skycap writes it as abandoned and releases it."""
    port: int = 0
    """Port for the skycap server; 0 picks a free one."""
    renderer_pool_size: int = 8
    """Renderers (tokenizer copies) skycap renders prompts with in parallel."""


@dataclass
class HarborSkycapConfig(HarborSkyRLConfig):
    skycap: SkycapConfig = field(default_factory=SkycapConfig)


def start_skycap(cfg: Any, engine_url: str) -> Any:
    """A skycap server in this process, in token mode, in front of SkyRL's router."""
    from skycap.tokens.backend import TokensBackend
    from skycap.tokens.renderer import RenderersRenderer

    from ..engine import SkyRLEngine
    from ..service import SkycapService

    ie = cfg.generator.inference_engine
    sampling = cfg.generator.sampling_params
    engine_init = dict(ie.engine_init_kwargs or {})
    backend = TokensBackend(
        engine_url,
        RenderersRenderer(cfg.trainer.policy.model.path, size=cfg.skycap.renderer_pool_size),
        engine=SkyRLEngine(),
        model=ie.served_model_name,
        max_model_len=engine_init.get("max_model_len") or cfg.trainer.algorithm.max_seq_len,
        # The trainer computes logprobs with these, so every rollout is sampled with them,
        # whatever the harness asks for.
        sampling_overrides={
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "min_p": sampling.min_p,
        },
        sampling_mask=ie.enable_return_sample_support_set,
    )
    node_ip = get_node_ip()
    service = SkycapService(
        backend,
        record_dir=cfg.skycap.record_dir or os.path.join(cfg.trainer.export_path, "skycap"),
        ttl=cfg.skycap.ttl,
        host=default_bind_host(node_ip),
        port=cfg.skycap.port,
        advertise_host=node_ip,
    )
    service.start()
    return service


class HarborSkycapExp(HarborExp):
    skycap: Any = None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if self.skycap is None:
            self.skycap = start_skycap(cfg, inference_engine_client.get_endpoint_url())
        return HarborSkycapGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            capture_urls=[self.skycap.url],
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
