"""
uv run --isolated --extra fsdp -m examples.train.algorithms.dapo.main_dapo_fully_async
"""

import sys

import ray
import torch
from typing import List, Optional, Tuple

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.entrypoints.main_base import BasePPOExp

from skyrl.train.generators.base import GeneratorOutput

from examples.train.algorithms.dapo.main_dapo import DAPOConfig


class FullyAsyncDAPOTrainer(FullyAsyncRayPPOTrainer):
    @torch.no_grad()
    def postprocess_generator_output(
        self,
        generator_output: GeneratorOutput,
        uids: List[str],
        metrics_generator_output: Optional[GeneratorOutput] = None,
        metrics_uids: Optional[List[str]] = None,
    ) -> Tuple[GeneratorOutput, List[str]]:
        """
        Overrides the postprocess_generator_output method to additionally apply DAPO specific soft overlong punishment to rewards.

        Handles both sequence-level rewards (List[float]) and per-token rewards (List[List[float]]).

        NOTE(Charlie): this is different from DAPOTrainer.postprocess_generator_output because we have
        batched=false in fully async mode, so we need to handle both sequence-level rewards and per-token rewards.
        """
        # Apply the penalty to the trained output, and also to the kept+dropped metrics view
        # (a separate concatenation) so reward metrics reflect the penalized rewards.
        self._apply_soft_overlong_punishment(generator_output)
        if metrics_generator_output is not None:
            self._apply_soft_overlong_punishment(metrics_generator_output)

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(
            generator_output, uids, metrics_generator_output=metrics_generator_output, metrics_uids=metrics_uids
        )

    def _apply_soft_overlong_punishment(self, generator_output: GeneratorOutput) -> None:
        """Apply DAPO soft overlong punishment to ``generator_output["rewards"]`` in place.

        Handles both sequence-level rewards (List[float]) and per-token rewards (List[List[float]]).
        """
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer_len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer_penalty_factor
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
                    # Subtract penalty from the last token's reward. Reassign the slot with a fresh
                    # list instead of mutating in place: generator_output and metrics_generator_output
                    # share the same inner reward-list objects (concatenate_generator_outputs only
                    # copies the outer list), so an in-place mutation would penalize the shared list
                    # twice -- once per call -- corrupting both the trained rewards and the metrics.
                    penalized = list(rewards[i])
                    penalized[-1] -= penalty
                    rewards[i] = penalized
                else:
                    rewards[i] -= penalty
            elif response_length > max_response_length:
                if is_per_token:
                    rewards[i] = [0.0] * len(rewards[i])
                else:
                    rewards[i] = 0.0

        generator_output["rewards"] = rewards


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
