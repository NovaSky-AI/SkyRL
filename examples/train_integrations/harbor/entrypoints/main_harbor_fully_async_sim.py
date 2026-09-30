"""Run real Harbor rollouts with the fully async simulated trainer."""

import sys

import ray
import yaml

from skyrl.train.fully_async_trainer_sim import FullyAsyncTrainerSim
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)


class HarborFullyAsyncSimExp(HarborExp):
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
        return FullyAsyncTrainerSim(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg):
    HarborFullyAsyncSimExp(cfg).run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as handle:
        cfg.harbor_trial_config = _deep_merge(yaml.safe_load(handle), cfg.harbor_trial_config)

    if not cfg.trainer.fully_async.simulate_training:
        raise ValueError("Set trainer.fully_async.simulate_training=true for this entrypoint.")
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set for Harbor.")
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
