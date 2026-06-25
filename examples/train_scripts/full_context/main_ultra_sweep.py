"""Entrypoint for the Nemotron-Ultra throughput/memory sweep.

uv run --isolated --extra megatron -m examples.train_scripts.full_context.main_ultra_sweep ...
"""

import sys
from dataclasses import dataclass

import ray

from skyrl.train.config import TrainerConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .trainer_ultra_sweep import UltraSweepTrainer


@dataclass
class UltraSweepTrainerConfig(TrainerConfig):
    num_dummy_steps: int = 3
    # Sweep params passed via CLI (trainer.<field>=...) so they reach the Ray
    # worker through cfg (shell env vars do NOT propagate to the entrypoint actor).
    sweep_results_file: str = "/home/ray/ultra_sweep/results.jsonl"
    sweep_tag: str = "run"
    sweep_mode: str = "uniform"  # uniform | varlen
    sweep_num_seq: int = -1  # -1 -> train_batch_size * n_samples_per_prompt
    sweep_prompt_len: int = 512
    sweep_seq_len: int = 10240  # uniform: total tokens/seq
    sweep_avg_len: int = 60000  # varlen
    sweep_std_len: int = 30000
    sweep_min_len: int = 1024
    sweep_max_len: int = 131072
    sweep_seed: int = 1234


UltraSweepConfig = make_config(trainer_cls=UltraSweepTrainerConfig)


class UltraSweepPPOExp(BasePPOExp):
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
        return UltraSweepTrainer(
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
    exp = UltraSweepPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = UltraSweepConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
