"""SFT entrypoint demonstrating the PerplexityLogger callback.

Usage:
    bash examples/train/callbacks/run_sft_with_callbacks.sh
"""

import sys

import ray

from examples.train.callbacks.perplexity_logger import PerplexityLogger
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.sft_config import (
    SFTConfig,
    build_skyrl_config_for_sft,
    validate_sft_cfg,
)
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.utils import initialize_ray


@ray.remote(num_cpus=1)
def sft_entrypoint(cfg: SFTConfig, skyrl_cfg: SkyRLTrainConfig):
    callbacks = [PerplexityLogger()]
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg, callbacks=callbacks)
    trainer.setup()
    trainer.train()
    trainer.shutdown()


def main():
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    validate_sft_cfg(cfg)
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    initialize_ray(skyrl_cfg)
    ray.get(sft_entrypoint.remote(cfg, skyrl_cfg))


if __name__ == "__main__":
    main()
