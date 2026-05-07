"""Training entry point for the Recursive Language Model (RLM) environment.

Wires ``RLMGeneratorConfig`` into the standard ``SkyRLTrainConfig`` via
``make_config`` and overrides ``BasePPOExp.get_generator`` so the RLM-specific
``RLMGymGenerator`` (with its hooks) is constructed instead of the base class.
"""

import sys

import ray

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import initialize_ray, validate_cfg

from .rlm_config import RLMGeneratorConfig
from .rlm_generator import RLMGymGenerator


RLMConfig = make_config(generator_cls=RLMGeneratorConfig)


class RLMPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return RLMGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    RLMPPOExp(cfg).run()


def main() -> None:
    cfg = RLMConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
