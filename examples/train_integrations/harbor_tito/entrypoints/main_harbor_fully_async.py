"""Fully asynchronous training entrypoint for the Harbor TITO example."""

import sys
from typing import cast

import ray

from examples.train_integrations.harbor.entrypoints.main_harbor_fully_async import (
    HarborFullyAsyncExp,
)
from examples.train_integrations.harbor_tito.entrypoints.main_harbor import (
    TITOHarborSkyRLConfig,
    _load_tito_harbor_config,
)
from examples.train_integrations.harbor_tito.harbor_generator import TITOHarborGenerator
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


class TITOHarborFullyAsyncExp(HarborFullyAsyncExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return TITOHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    TITOHarborFullyAsyncExp(cfg).run()


def main() -> None:
    cfg = cast(TITOHarborSkyRLConfig, TITOHarborSkyRLConfig.from_cli_overrides(sys.argv[1:]))
    cfg.harbor_trial_config = _load_tito_harbor_config(cfg.harbor_trial_config)
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set for Harbor TITO training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
