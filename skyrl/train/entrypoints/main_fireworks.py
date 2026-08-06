"""Direct entrypoint for hosted Fireworks training.

The Fireworks backend does not create local policy, critic, reference, or
inference workers, so attaching the driver to a Ray cluster only adds a version
compatibility constraint. Keep the orchestration loop in this process while
the Fireworks SDK performs model computation remotely.
"""

from __future__ import annotations

import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg


class FullyAsyncFireworksExp(BasePPOExp):
    """Use SkyRL's async scheduler while Fireworks owns model compute."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        # Keep this import lazy so the synchronous hosted path does not load
        # the async scheduler implementation.
        from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

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


def experiment_class(cfg: SkyRLTrainConfig):
    """Select the SkyRL scheduling loop; neither path creates model workers."""

    return FullyAsyncFireworksExp if cfg.trainer.fully_async.enabled else BasePPOExp


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if cfg.trainer.strategy != "fireworks":
        raise ValueError("main_fireworks requires trainer.strategy='fireworks'")
    experiment_class(cfg)(cfg).run()


if __name__ == "__main__":
    main()
