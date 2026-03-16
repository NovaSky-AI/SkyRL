"""
uv run --isolated --extra fsdp -m examples.train_scripts.full_context.main_full_ctx
"""

import sys
from dataclasses import dataclass

import ray

from skyrl.train.config import TrainerConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .trainer_full_ctx import FullCtxTrainer


@dataclass
class FullCtxTrainerConfig(TrainerConfig):
    num_dummy_steps: int = 5


FullCtxConfig = make_config(trainer_cls=FullCtxTrainerConfig)


class FullCtxPPOExp(BasePPOExp):
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
        return FullCtxTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = FullCtxPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = FullCtxConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
