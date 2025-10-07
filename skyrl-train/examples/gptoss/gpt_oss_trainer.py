import asyncio
from typing import List, Optional, Tuple
from jaxtyping import Float
import ray
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.generators.base import (
    GeneratorOutput,
)
from skyrl_train.generators.utils import prepare_generator_input
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import (
    get_kl_controller,
    normalize_advantages_dict,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.trainer_utils import (
    ResumeMode,
)
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.dataset.preprocess import _verify_inputs

from examples.gptoss.attn_mask_utils import make_attention_mask
from examples.gptoss.gpt_oss_generator import GPTOSSGeneratorOutput

class GPTOSSTrainer(RayPPOTrainer):
    def train(self):
        """
        Main training loop for PPO
        """
        # Initialize weight sync state between policy model and inference engines.
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # Load policy model to GPU before loading checkpoint.
        if self.colocate_all:
            self.policy_model.backload_to_gpu()

        # Load checkpoint state if resumption is enabled.
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.global_step = self.load_checkpoints()

        if self.colocate_all:
            asyncio.run(self.inference_engine_client.wake_up(tags=["weights"]))
        with Timer("sync_weights"):
            ray.get(self.sync_policy_weights_to_inference_engines())
        if self.colocate_all:
            with Timer("offload_policy_model_to_cpu"):
                self.policy_model.offload_to_cpu()
            asyncio.run(self.inference_engine_client.wake_up(tags=["kv_cache"]))

        # Eval before training
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = asyncio.run(self.eval())
                self.tracker.log(eval_metrics, step=self.global_step, commit=True)

        # initialize kl controller
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Batches Processed")
        self.global_step += 1  # start training at global_step 1
        for epoch in range(self.cfg.trainer.epochs):
            for iter, rand_prompts in enumerate(self.train_dataloader):
                with Timer("step", self.all_timings):
                    # for colocate_all=true, inference engine is always on GPU when starting the training step

                    # 0. truncate data to have even shards
                    rand_prompts = self._remove_tail_data(rand_prompts)
                    generator_input, uids = prepare_generator_input(
                        rand_prompts,
                        self.cfg.generator.n_samples_per_prompt,
                        get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
                        self.cfg.environment.env_class,
                        "train",
                        self.global_step,
                    )

                    # 1.1 generation phase
                    with Timer("generate", self.all_timings):
                        generator_output: GeneratorOutput = asyncio.run(self.generate(generator_input))

                    # dynamic sampling
                    if self.cfg.trainer.algorithm.dynamic_sampling.type is not None:
                        generator_output, uids, keep_sampling = self.handle_dynamic_sampling(generator_output, uids)
                        if keep_sampling:  # continue sampling
                            # update progress bar for current batch (but not global step)
                            pbar.update(1)
                            continue

                    # if we are not continuing sampling, we sleep the inference engine
                    asyncio.run(self.inference_engine_client.sleep())

                    breakpoint()

                    # 1.2 postprocess rewards
                    with Timer("postprocess_generator_output", self.all_timings):
                        generator_output = self.postprocess_generator_output(generator_output, uids)

                    # 2. print example just for debugging
                    vis = self.tokenizer.decode(generator_output["response_ids"][0])
                    logger.info(f"Example:\n" f"  Input: {generator_input['prompts'][0]}\n" f"  Output:\n{vis}")

                    with Timer("convert_to_training_input", self.all_timings):
                        training_input: TrainingInputBatch = self.convert_to_training_input(generator_output, uids)
                        logger.info(f"Number of sequences: {len(training_input['sequences'])}")

                    # 1.4 inference and calculate values, log probs, rewards, kl divergence
                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)

                    # 1.5 apply kl divergence penalty to rewards
                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(training_input)

                    # 3. calculate advantages and returns
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(training_input)
                        # remove some unwanted keys
                        for key in ["rewards"]:
                            training_input.pop(key)
                        training_input.metadata.pop("uids")

                        if self.cfg.trainer.algorithm.advantage_batch_normalize:
                            training_input = normalize_advantages_dict(training_input)

                    if self.cfg.trainer.dump_data_batch:
                        # dump data to file
                        with Timer("dump_data_batch"):
                            self.dump_data(training_input, file_name=f"global_step_{self.global_step}_training_input")

                    # 4. train policy/critic model
                    # Policy model is backloaded to GPU during training
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(training_input)

                    # 5. conditionally save checkpoints and hf model
                    if self.cfg.trainer.ckpt_interval > 0 and self.global_step % self.cfg.trainer.ckpt_interval == 0:
                        with Timer("save_checkpoints", self.all_timings):
                            self.save_checkpoints()
                    if (
                        self.cfg.trainer.hf_save_interval > 0
                        and self.global_step % self.cfg.trainer.hf_save_interval == 0
                    ):
                        with Timer("save_hf_model", self.all_timings):
                            self.save_models()

                    # 6. conditionally sync policy and ref at the end of the epoch
                    if (
                        self.cfg.trainer.update_ref_every_epoch
                        and self.ref_model is not None
                        and iter == len(self.train_dataloader) - 1
                        and epoch != self.cfg.trainer.epochs - 1  # skip updating ref at the end of the last epoch
                    ):
                        with Timer("update_ref_with_policy", self.all_timings):
                            self.update_ref_with_policy()

                    # 7. sync weights to inference engines
                    if self.colocate_all:
                        asyncio.run(self.inference_engine_client.wake_up(tags=["weights"]))
                    with Timer("sync_weights", self.all_timings):
                        ray.get(self.sync_policy_weights_to_inference_engines())
                    if self.colocate_all:
                        with Timer("offload_policy_model_to_cpu"):
                            self.policy_model.offload_to_cpu()
                        asyncio.run(self.inference_engine_client.wake_up(tags=["kv_cache"]))

                # 8. set logs
                logger.info(status)
                # log epoch info
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with Timer("eval", self.all_timings):
                        eval_metrics = asyncio.run(self.eval())
                        self.all_metrics.update(eval_metrics)

                log_payload = {
                    **self.all_metrics,
                    **{f"timing/{k}": v for k, v in self.all_timings.items()},
                }
                self.tracker.log(log_payload, step=self.global_step, commit=True)
                self.all_metrics = {}
                self.all_timings = {}

                # update progress bar after logging
                pbar.update(1)

                self.global_step += 1

                del training_input, generator_output

        pbar.close()
        if self.colocate_all:
            asyncio.run(self.inference_engine_client.sleep())
            self.policy_model.backload_to_gpu()
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                self.save_checkpoints()
                logger.info("Saved final checkpoint.")
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                self.save_models()
                logger.info("Saved final model.")
        logger.info("Training done!")

    def convert_to_training_input(self, generator_output: GPTOSSGeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training"""
        prompt_ids: List[List[int]] = generator_output["prompt_token_ids"]
        response_ids: List[List[int]] = generator_output["response_ids"]
        rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        logprobs: Optional[List[List[float]]] = generator_output.get("rollout_logprobs", None)
        kinds = generator_output["kinds"]
        steps = generator_output["steps"]

        (
            sequences_tensor,
            attention_masks_tensor,
            response_masks_tensor,
            rewards_tensor,
            loss_masks_tensor,
            rollout_logprobs_tensor,
        ) = convert_prompts_responses_to_batch_tensors(
            self.tokenizer,
            prompt_ids,
            response_ids,
            rewards,
            loss_masks,
            kinds,
            steps,
            logprobs,
        )
        # attention_masks4d_tensor = self._get_attention_mask(kinds_tensor, steps_tensor)
        # sanity check for tis
        if self.cfg.trainer.algorithm.use_tis:
            assert (
                rollout_logprobs_tensor is not None
            ), "expected non-null rollout logprobs tensor with  `trainer.algorithm.use_tis` as `True`"
            assert rollout_logprobs_tensor.shape == loss_masks_tensor.shape, "Logprobs should look like responses"
        training_input = TrainingInputBatch(
            {
                "sequences": sequences_tensor,  # Full trajectories (padded and concatenated prompts and responses)
                "attention_mask": attention_masks_tensor,
                "response_mask": response_masks_tensor,
                "rewards": rewards_tensor,
                "loss_mask": loss_masks_tensor,
                "rollout_logprobs": rollout_logprobs_tensor,   
            },
        )
        training_input.metadata = {
            "uids": uids,
        }
        # padded response length
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        training_input.metadata["avg_response_length"] = sum(
            len(sample_response_ids) for sample_response_ids in response_ids
        ) / len(response_ids)
        return training_input
