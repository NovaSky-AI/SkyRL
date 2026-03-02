"""
Main entrypoint for DAPO training with FlashRL.
"""

import ray
import hydra
import torch
from typing import List, Union
from omegaconf import DictConfig
from skyrl_train.config import SkyRLConfig
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    validate_cfg,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.base import GeneratorOutput


def create_ray_wrapped_inference_engines_from_config_flashrl(
    cfg: Union[SkyRLConfig, DictConfig], colocate_pg, tokenizer
):
    from examples.flash_rl.flash_rl_engine import create_ray_wrapped_inference_engines_flashrl

    return create_ray_wrapped_inference_engines_flashrl(
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        model_dtype=cfg.generator.model_dtype,
        pretrain=cfg.trainer.policy.model.path,
        seed=cfg.trainer.seed,
        vllm_v1_disable_multiproc=cfg.generator.vllm_v1_disable_multiproc,
        enable_prefix_caching=cfg.generator.enable_prefix_caching,
        enforce_eager=cfg.generator.enforce_eager,
        shared_pg=colocate_pg,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=cfg.trainer.placement.colocate_all,
        async_engine=cfg.generator.async_engine,
        max_num_batched_tokens=cfg.generator.max_num_batched_tokens,
        max_num_seqs=cfg.generator.max_num_seqs,
        tokenizer=tokenizer,
        backend=cfg.generator.backend,
    )


class DAPOTrainer(RayPPOTrainer):
    """
    Custom trainer for DAPO.

    Overrides the postprocess_generator_output method to additionally apply soft overlong punishment to rewards.
    """

    @torch.no_grad()
    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """
        Overrides the postprocess_generator_output method to additionally apply DAPO specific soft overlong punishment to rewards.

        Args:
            generator_output: GeneratorOutput
            uids: List[str]

        Returns:
            GeneratorOutput
        """
        overlong_buffer_len = self.cfg.trainer.algorithm.overlong_buffer.len
        overlong_buffer_penalty_factor = self.cfg.trainer.algorithm.overlong_buffer.penalty_factor
        # modify rewards here
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"

        # get the response length
        response_lengths = [len(response) for response in response_ids]

        # get the max context length
        # NOTE: this is only valid for single turn generation
        max_response_length = self.cfg.generator.sampling_params.max_generate_length

        # apply soft overlong punishment
        for i, response_length in enumerate(response_lengths):
            # max_exceed_length is the beginning of the overlong buffer
            max_exceed_length = max_response_length - overlong_buffer_len
            # if the response is within the overlong buffer, apply the penalty
            if response_length > max_exceed_length and response_length <= max_response_length:
                exceed_length = response_length - max_exceed_length
                penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                rewards[i] -= penalty
            # if the response is outside the overlong buffer, set the reward to 0
            elif response_length > max_response_length:
                # if self.cfg.generator.apply_overlong_filtering is true, loss masks are already set to 0 for these responses
                rewards[i] = 0.0

        generator_output["rewards"] = rewards

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)

    def get_inference_client(self) -> InferenceEngineInterface:
        """Setup and return the inference engine client using FlashRL engines.

        Returns:
            InferenceEngineInterface: The inference engine client.
        """
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config_flashrl(
                self.cfg, self.colocate_pg, self.tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, self.tokenizer)

        return InferenceEngineClient(inference_engines, self.tokenizer, self.cfg)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):

    exp = DAPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    if not cfg.generator.run_engines_locally:
        raise ValueError("FlashRL only supports colocated training.")

    if cfg.trainer.strategy not in ("fsdp", "fsdp2"):
        raise ValueError(f"FlashRL only supports fsdp/fsdp2 strategy, got: {cfg.trainer.strategy}")

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
