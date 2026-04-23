"""Training entry point for the Recursive Language Model (RLM) environment.

Wires ``RLMGeneratorConfig`` (RLM-specific generator fields) into the standard
``SkyRLTrainConfig`` via ``make_config``, then dispatches to ``BasePPOExp``.
"""

import sys

import ray

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .rlm_config import RLMGeneratorConfig


RLMConfig = make_config(generator_cls=RLMGeneratorConfig)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    BasePPOExp(cfg).run()


def main() -> None:
    cfg = RLMConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
