import torch
import math

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

        # NOTE (sumanthrh): We filter out instances that failed during generation, so the number of prompts at input can differ from number of trajectories at output
        # we simply ignore validation for now
        # validate_generator_output(input_batch, generator_output)

        # ensure that the number of trajectories meets the minimum required
        num_trajectories = len(generator_output["response_ids"])
        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        assert (
            num_trajectories >= self.policy_model.actor_infos[0].rank.dp_size
        ), f"Expected atleast {dp_size} valid trajectories for num policy workers {dp_size}, but got {num_trajectories} trajectories"
        if self.critic_model is not None:
            dp_size = math.lcm(dp_size, self.critic_model.actor_infos[0].rank.dp_size)
            assert (
                num_trajectories >= self.critic_model.actor_infos[0].rank.dp_size
            ), f"Expected atleast {self.critic_model.actor_infos[0].rank.dp_size} valid trajectories for num critic workers {self.critic_model.actor_infos[0].rank.dp_size}, but got {num_trajectories} trajectories"
        if self.ref_model is not None:
            dp_size = math.lcm(dp_size, self.ref_model.actor_infos[0].rank.dp_size)
            assert (
                num_trajectories >= self.ref_model.actor_infos[0].rank.dp_size
            ), f"Expected atleast {self.ref_model.actor_infos[0].rank.dp_size} valid trajectories for num ref workers {self.ref_model.actor_infos[0].rank.dp_size}, but got {num_trajectories} trajectories"

        # removes tail data so that workers get even shards
        for key in generator_output.keys():
            if isinstance(generator_output[key], list):
                generator_output[key] = generator_output[key][: (num_trajectories // dp_size) * dp_size]

        if not len(generator_output["prompt_token_ids"]):
            raise ValueError(
                "Found no valid generation. This likely means that there is something wrong with your environment setup. Please ensure that you can successfully generate trajectories with `mini-swe-agent` in your Ray cluster"
            )

        return generator_output
