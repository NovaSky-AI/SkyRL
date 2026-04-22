"""
CLI entrypoint for SFT (Supervised Fine-Tuning) training.

Parses CLI arguments, validates the config, initializes Ray on the head
node, then dispatches training to a remote Ray task via ``sft_entrypoint``.

Usage::

    python -m skyrl.train.main_sft strategy=megatron model.path=Qwen/Qwen3-0.6B
"""

import sys

import ray

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
    """Run SFT training as a Ray task (off the head node).

    Mirrors the ``skyrl_entrypoint`` pattern from ``main_base.py``:
    the training loop runs inside a lightweight Ray task so that the
    head-node process only handles config parsing and ``ray.init()``.

    Receives the pre-built ``skyrl_cfg`` so that the trainer does not
    need to rebuild the bridge config.
    """
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
    trainer.setup()
    trainer.train()
    trainer.shutdown()


def main():
    """CLI entrypoint for SFT training.

    Parses CLI arguments, validates the config, initializes Ray on the
    head node, then dispatches training to a remote Ray task via
    ``sft_entrypoint``.

    Usage::

        python -m skyrl.train.main_sft strategy=megatron model.path=Qwen/Qwen3-0.6B
    """
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    validate_sft_cfg(cfg)
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    initialize_ray(skyrl_cfg)
    ray.get(sft_entrypoint.remote(cfg, skyrl_cfg))


if __name__ == "__main__":
    main()
