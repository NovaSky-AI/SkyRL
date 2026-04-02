"""
Main entrypoint for evaluation-only.
"""

import asyncio
import sys
from typing import Any

import ray
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import (
    BasePPOExp,
)
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    async def run(self) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        inference_engine_client = self.get_inference_client()
        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)
        evaluator = self.get_evaluator(self.cfg, self.tokenizer, generator)

        if self.cfg.generator.step_wise_trajectories:
            results = await evaluator.evaluate_step_wise(
                eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
                global_step=None,
            )
        else:
            results = await evaluator.evaluate(
                eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
                global_step=None,
            )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
