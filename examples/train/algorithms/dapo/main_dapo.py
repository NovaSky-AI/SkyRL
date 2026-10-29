"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo
"""

import sys

import ray
import torch
from dataclasses import dataclass
from typing import List, Tuple, Union

from skyrl.train.config import AlgorithmConfig, make_config
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorOutput


@dataclass
class DAPOAlgorithmConfig(AlgorithmConfig):
    """Extended algorithm config with DAPO-specific overlong buffer settings."""

    overlong_buffer_len: int = 512
    overlong_buffer_penalty_factor: float = 1.0


DAPOConfig = make_config(algorithm_cls=DAPOAlgorithmConfig)


def apply_soft_overlong_punishment(
    rewards: Union[List[float], List[List[float]]],
    response_lengths: List[int],
    max_response_length: int,
    overlong_buffer_len: int,
    overlong_buffer_penalty_factor: float,
) -> Union[List[float], List[List[float]]]:
    """Applies DAPO soft overlong punishment to rewards.

    Responses longer than ``max_response_length - overlong_buffer_len`` are penalized linearly with the number
    of tokens in the overlong buffer, up to ``overlong_buffer_penalty_factor``. Responses longer than
    ``max_response_length`` get a reward of 0.

    Rewards are either sequence-level (``List[float]``, e.g. from batched generation) or per-token
    (``List[List[float]]``, e.g. from non-batched generation). For per-token rewards, the penalty is subtracted from
    the last token's reward and over-length responses get all-zero rewards.

    Args:
        rewards: Sequence-level or per-token rewards, one entry per response.
        response_lengths: Number of response tokens for each response.
        max_response_length: Maximum number of generated tokens per response.
        overlong_buffer_len: Length of the overlong buffer at the end of the response budget.
        overlong_buffer_penalty_factor: Penalty for a response that fills the overlong buffer.

    Returns:
        A new list of rewards. Per-token reward lists are copied before modification, so the input lists are
        not mutated.
    """
    is_per_token = len(rewards) > 0 and isinstance(rewards[0], list)
    # max_exceed_length is the beginning of the overlong buffer
    max_exceed_length = max_response_length - overlong_buffer_len
    penalized_rewards = list(rewards)
    for i, response_length in enumerate(response_lengths):
        # if the response is within the overlong buffer, apply the penalty
        if max_exceed_length < response_length <= max_response_length:
            exceed_length = response_length - max_exceed_length
            penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor
            if is_per_token:
                token_rewards = list(rewards[i])
                token_rewards[-1] -= penalty
                penalized_rewards[i] = token_rewards
            else:
                penalized_rewards[i] = rewards[i] - penalty
        # if the response is outside the overlong buffer, set the reward to 0
        elif response_length > max_response_length:
            # with `generator.apply_overlong_filtering=true`, loss masks are already set to 0 for these responses
            penalized_rewards[i] = [0.0] * len(rewards[i]) if is_per_token else 0.0
    return penalized_rewards


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    Handles both sequence-level rewards (batched generation) and per-token rewards (non-batched generation).
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
        if self.cfg.generator.step_wise_trajectories:
            raise ValueError("`DAPOTrainer` does not support `generator.step_wise_trajectories=true`.")
        response_lengths = [len(response) for response in generator_output["response_ids"]]
        # NOTE: this is only valid for single turn generation
        max_response_length = self.cfg.generator.sampling_params.max_generate_length
        generator_output["rewards"] = apply_soft_overlong_punishment(
            generator_output["rewards"],
            response_lengths,
            max_response_length,
            overlong_buffer_len,
            overlong_buffer_penalty_factor,
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
