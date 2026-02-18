"""
uv run --isolated --extra vllm -m scripts.full_context.main_full_ctx
"""

import sys

from skyrl_train.config import SkyRLTrainConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.utils import initialize_ray
import ray
from .trainer_full_ctx import FullCtxTrainer


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
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # make sure that the training loop is not run on the head node.
    exp = FullCtxPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
