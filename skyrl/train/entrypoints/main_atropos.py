import asyncio
import os
import sys
from typing import Optional

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, skyrl_entrypoint
from skyrl.train.generators.base import GeneratorInterface
from skyrl.train.generators.atropos_shm_generator import AtroposSHMGenerator
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


class AtroposPPOExp(BasePPOExp):
    """
    Overridden PPO Experiment that uses Atropos Zero-Copy SHM for trajectories.
    """

    def get_generator(self, cfg, tokenizer, inference_engine_client) -> GeneratorInterface:
        """
        Initializes the AtroposSHMGenerator.
        """
        logger.info("Initializing AtroposSHMGenerator (Zero-Copy SHM)...")
        
        # Pull SHM settings from env or config overrides
        shm_name = os.environ.get("ATROPOS_SHM_NAME", "atropos_shm")
        shm_size = int(os.environ.get("ATROPOS_SHM_SIZE", "1000"))
        
        return AtroposSHMGenerator(
            shm_name=shm_name,
            shm_size=shm_size,
            tokenizer_name=cfg.trainer.policy.model.path,
            poll_interval=0.01, # Tight polling for high performance
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """
        Initializes the trainer. Supports switching to FullyAsyncRayPPOTrainer 
        if configured in the trainer settings.
        """
        if hasattr(cfg.trainer, "fully_async") and cfg.trainer.fully_async.enabled:
            from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
            logger.info("Using FullyAsyncRayPPOTrainer for Atropos integration.")
            return FullyAsyncRayPPOTrainer(
                cfg=cfg,
                tracker=tracker,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                inference_engine_client=inference_engine_client,
                generator=generator,
                colocate_pg=colocate_pg,
            )
        
        return super().get_trainer(
            cfg, tracker, tokenizer, train_dataset, eval_dataset, 
            inference_engine_client, generator, colocate_pg
        )


@ray.remote(num_cpus=1)
def atropos_skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """
    Ray entrypoint for Atropos-SkyRL experiments.
    """
    exp = AtroposPPOExp(cfg)
    exp.run()


def main() -> None:
    # Parse CLI args and build typed config
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])

    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    
    # Run the Atropos-specialized entrypoint
    ray.get(atropos_skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
