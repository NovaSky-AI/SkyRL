"""
Main entrypoint for evaluation-only.
"""

import asyncio

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from typing import Any

from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg)

        inference_engine_client = InferenceEngineClient(inference_engines)
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        trainer = RayPPOTrainer(
            cfg=self.cfg,
            tracker=self.get_tracker(),
            tokenizer=tokenizer,
            train_dataset=None,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        results: dict[str, Any] = await trainer.eval(eval_only=True)

        # Export to wandb if configured
        logger_cfg = self.cfg.trainer.logger
        uses_wandb = logger_cfg == "wandb" if isinstance(logger_cfg, str) else "wandb" in logger_cfg
        if uses_wandb:
            trainer.tracker.log(results, step=0)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
