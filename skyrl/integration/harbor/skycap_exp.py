"""Harbor training with skycap capturing each rollout's exact tokens. Needs the ``harbor`` and ``skycap`` extras.

``HarborExp`` with two changes: a pool of skycap servers (``skyrl.integration.skycap.servers``) starts in front of
the inference router, and the generator points each trial at its own trajectory on one of them.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from skyrl.integration.harbor.config import HarborSkyRLConfig
from skyrl.integration.harbor.exp import HarborExp
from skyrl.integration.harbor.skycap_generator import HarborSkycapGenerator
from skyrl.integration.skycap.config import SkycapConfig, start_skycap
from skyrl.integration.skycap.record_index import RecordLog, SkycapRecordIndex
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
