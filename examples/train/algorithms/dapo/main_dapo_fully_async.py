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

from examples.train.algorithms.dapo.main_dapo import DAPOConfig, apply_soft_overlong_punishment


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
        """Replaces ``generator_output["rewards"]`` with the penalized rewards from ``apply_soft_overlong_punishment``.

        ``apply_soft_overlong_punishment`` does not modify its input lists, so per-token reward lists shared
        between the trained output and the metrics output are penalized once per output.
        """
        response_lengths = [len(response) for response in generator_output["response_ids"]]
        generator_output["rewards"] = apply_soft_overlong_punishment(
            generator_output["rewards"],
            response_lengths,
            self.cfg.generator.sampling_params.max_generate_length,
            self.cfg.trainer.algorithm.overlong_buffer_len,
            self.cfg.trainer.algorithm.overlong_buffer_penalty_factor,
        )


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
