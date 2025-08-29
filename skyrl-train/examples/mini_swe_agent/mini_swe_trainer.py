import torch

from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
)
from skyrl_train.trainer import RayPPOTrainer


class MiniSWEPPOTrainer(RayPPOTrainer):
    @torch.no_grad()
    async def generate(
        self,
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate rollouts.

        If colocate_all is enabled:
        - before calling this method, the policy model should be on CPU and inference engine should
            be awake (i.e. on GPU).
        - after calling this method, the same model placement still holds.
        """
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        # NOTE (sumanthrh): We filter out instances that failed, so the number of prompts at input can differ from number of trajectories at output
        # we simply ignore validation for now
        # validate_generator_output(input_batch, generator_output)

        if not len(generator_output["prompt_token_ids"]):
            raise ValueError(
                "Found no valid generation. This likely means that there is something wrong with your environment setup. Please ensure that you can successfully generate trajectories with `mini-swe-agent` in your Ray cluster"
            )

        return generator_output
