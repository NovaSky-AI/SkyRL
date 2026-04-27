"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo_fully_async
"""

import sys

import ray
import torch
from typing import List, Tuple

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.utils.reward_shaping import apply_dapo_soft_overlong_punishment

from examples.train.algorithms.dapo.main_dapo import DAPOConfig


class FullyAsyncDAPOTrainer(FullyAsyncRayPPOTrainer):
    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        """
        Overrides the postprocess_generator_output method to additionally apply DAPO specific soft overlong punishment to rewards.

        Handles both sequence-level rewards (List[float]) and per-token rewards (List[List[float]]).

        NOTE(Charlie): this is different from DAPOTrainer.postprocess_generator_output because we have
        batched=false in fully async mode, so we need to handle both sequence-level rewards and per-token rewards.
        """
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
        # modify rewards here
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        max_response_length = self.cfg.trainer.algorithm.max_response_length
        if max_response_length is None:
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


class FullyAsyncDAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return FullyAsyncDAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = FullyAsyncDAPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = DAPOConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
