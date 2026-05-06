"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo
"""

import sys

import ray
import torch
from dataclasses import dataclass
from typing import List, Tuple

from skyrl.train.config import AlgorithmConfig, make_config
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.utils.reward_shaping import apply_dapo_soft_overlong_punishment


@dataclass
class DAPOAlgorithmConfig(AlgorithmConfig):
    """Extended algorithm config with DAPO-specific overlong buffer settings."""

    overlong_buffer_len: int = 512
    overlong_buffer_penalty_factor: float = 1.0


DAPOConfig = make_config(algorithm_cls=DAPOAlgorithmConfig)


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    """

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        # NOTE (sumanthrh): Given the usage of `make_config`, the algorithm config subclass for DAPO is
        # created dynamically and thus IDEs will not be able to resolve the attributes
        # For better typing, you can always define a custom subclass of DAPOConfig manually.
        # See examples/train_integrations/harbor for an example.
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
        # modify rewards here
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"
        max_response_length = self.cfg.trainer.algorithm.max_response_length
        if max_response_length is None:
            # Legacy fallback for examples that still only configure generator-side max generation length.
            max_response_length = self.cfg.generator.sampling_params.max_generate_length

        generator_output["rewards"] = apply_dapo_soft_overlong_punishment(
            response_ids=response_ids,
            rewards=rewards,
            overlong_buffer_len=overlong_buffer_len,
            overlong_buffer_penalty_factor=overlong_buffer_penalty_factor,
            max_response_length=max_response_length,
        )

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = DAPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = DAPOConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
