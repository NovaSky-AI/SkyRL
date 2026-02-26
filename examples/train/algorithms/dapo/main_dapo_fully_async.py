"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo_fully_async
"""

import ray
import hydra
import torch
from typing import List
from omegaconf import DictConfig
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg

from skyrl.train.generators.base import GeneratorOutput



class FullyAsyncDAPOTrainer(FullyAsyncRayPPOTrainer):
    @torch.no_grad()
    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """
        Overrides the postprocess_generator_output method to additionally apply DAPO specific soft overlong punishment to rewards.

        Handles both sequence-level rewards (List[float]) and per-token rewards (List[List[float]]).

        NOTE(Charlie): this is different from DAPOTrainer.postprocess_generator_output because we have
        batched=false in fully async mode, so we need to handle both sequence-level rewards and per-token rewards.
        """
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer.len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer.penalty_factor
        # modify rewards here
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        response_lengths = [len(response) for response in response_ids]
        max_response_length = self.cfg.generator.sampling_params.max_generate_length

        # Determine if rewards are per-token (List[List[float]]) or sequence-level (List[float])
        is_per_token = rewards and isinstance(rewards[0], list)

        # apply soft overlong punishment
        for i, response_length in enumerate(response_lengths):
            # max_exceed_length is the beginning of the overlong buffer
            max_exceed_length = max_response_length - overlong_buffer_len
            # if the response is within the overlong buffer, apply the penalty
            if response_length > max_exceed_length and response_length <= max_response_length:
                exceed_length = response_length - max_exceed_length
                penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                if is_per_token:
                    # Subtract penalty from the last token's reward
                    rewards[i][-1] -= penalty
                else:
                    rewards[i] -= penalty
            elif response_length > max_response_length:
                if is_per_token:
                    rewards[i] = [0.0] * len(rewards[i])
                else:
                    rewards[i] = 0.0

        generator_output["rewards"] = rewards

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class FullyAsyncDAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return FullyAsyncDAPOTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = FullyAsyncDAPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()