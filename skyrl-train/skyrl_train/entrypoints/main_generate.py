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
)
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray
from skyrl_train.evaluate import evaluate
from skyrl_train.utils.trainer_utils import build_dataloader


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        inference_engine_client = self.get_inference_client()
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        results: dict[str, Any] = await evaluate(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

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
