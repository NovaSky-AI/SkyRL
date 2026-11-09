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
    # "max": every sample at max prompt + max response length. "mixture": per-sample lengths, a
    # `dummy_truncated_fraction` of responses at the max, the rest uniform from the min.
    dummy_length_distribution: str = "max"
    dummy_min_prompt_length: int = 128
    dummy_max_prompt_length: int = 512
    dummy_min_response_length: int = 512
    dummy_truncated_fraction: float = 0.25
    # "repeat" (one token per batch), "random" (uniform ids), or "dataset" (tokenized train prompts).
    dummy_token_source: str = "repeat"
    dummy_seed: int = 0
    # Benchmark the trainer alone: no inference engines and no weight sync. Needs colocate_all=false.
    skip_inference_engines: bool = False


FullCtxConfig = make_config(trainer_cls=FullCtxTrainerConfig)


class FullCtxPPOExp(BasePPOExp):
    def get_inference_client(self):
        if self.cfg.trainer.skip_inference_engines:
            assert not self.cfg.trainer.placement.colocate_all, "skip_inference_engines needs colocate_all=false"
            return None
        return super().get_inference_client()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        if cfg.trainer.skip_inference_engines:
            return None
        return super().get_generator(cfg, tokenizer, inference_engine_client)

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
