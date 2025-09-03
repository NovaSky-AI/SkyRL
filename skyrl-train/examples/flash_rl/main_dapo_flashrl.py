"""
uv run --isolated --extra flashrl -m examples.flash_rl.main_dapo_flashrl
"""

import ray
import os
import hydra
import torch
from typing import List
from omegaconf import DictConfig
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
    validate_cfg,
    create_remote_inference_engines_from_config,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.base import GeneratorInterface
from loguru import logger
from skyrl_train.generators.base import GeneratorOutput


def create_ray_wrapped_inference_engines_from_config_flashrl(cfg: DictConfig, colocate_pg, tokenizer):
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
        max_model_len=cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length,
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
        prompt_token_ids = generator_output["prompt_token_ids"]
        response_ids = generator_output["response_ids"]
        rewards = generator_output["rewards"]

        assert not isinstance(rewards[0], list), "we assume verifiable sequence level rewards here"

        # get the prompt length
        prompt_lengths = [len(prompt) for prompt in prompt_token_ids]

        # get the response length
        response_lengths = [len(response) for response in response_ids]

        # get the max context length
        max_context_length = (
            self.cfg.generator.max_input_length + self.cfg.generator.sampling_params.max_generate_length
        )

        # apply soft overlong punishment
        for i, (prompt_length, response_length) in enumerate(zip(prompt_lengths, response_lengths)):
            # max_exceed_length is the beginning of the overlong buffer
            max_exceed_length = max_context_length - overlong_buffer_len - prompt_length
            # if the response is within the overlong buffer, apply the penalty
            if response_length > max_exceed_length and response_length <= max_context_length - prompt_length:
                exceed_length = response_length - max_exceed_length
                penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

                rewards[i] -= penalty
            # if the response is outside the overlong buffer, set the reward to 0
            elif response_length > max_context_length - prompt_length:
                # if self.cfg.generator.apply_overlong_filtering is true, loss masks are already set to 0 for these responses
                rewards[i] = 0.0

        generator_output["rewards"] = rewards

        # use base class impl for metrics and per-token reward conversion
        return super().postprocess_generator_output(generator_output, uids)


class DAPOExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return DAPOTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Setup and return the trainer.

        Instantiates the trainer and all the associated models for training.

        Returns:
            RayPPOTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import (
                PolicyWorker,
                CriticWorker,
                RefWorker,
                RewardWorker,
            )
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker, RewardWorker
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config_flashrl(
                self.cfg, self.colocate_pg, tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)

        generator: GeneratorInterface = self.get_generator(self.cfg, tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build the models
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker, RewardWorker)
        return trainer


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

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
