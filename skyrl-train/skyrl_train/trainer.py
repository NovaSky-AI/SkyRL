import copy
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ray
import torch
from jaxtyping import Float
from loguru import logger
from omegaconf import DictConfig
from ray import ObjectRef
from ray.util.placement_group import PlacementGroup, placement_group
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl_train.dataset import PromptDataset
from skyrl_train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
)
from skyrl_train.distributed.dispatch import (
    ActorInfo,
    MeshRank,
)
from skyrl_train.evaluate import evaluate, evaluate_step_wise
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl_train.generators.utils import (
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import (
    Timer,
    get_ray_pg_ready_with_timeout,
    ppo_utils,
    trainer_utils,
)
from skyrl_train.utils.trainer_utils import calculate_per_source_reward_metrics
from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl_train.utils.io import io
from skyrl_train.utils.logging_utils import log_example
from skyrl_train.utils.ppo_utils import (
    AdaptiveKLController,
    FixedKLController,
    compute_approx_kl,
    get_kl_controller,
    masked_mean,
    normalize_advantages_dict,
)
from skyrl_train.utils.tracking import Tracking
from skyrl_train.utils.trainer_utils import (
    GLOBAL_STEP_PREFIX,
    DynamicSamplingState,
    ResumeMode,
    build_dataloader,
    cleanup_old_checkpoints,
    extract_step_from_path,
    run_on_each_node,
    validate_consistency_for_latest_checkpoint,
    validate_generator_output,
    zero_variance_filter,
)
from skyrl_train.utils.alerts import check_tool_error_rate_alert
from skyrl_train.utils.utils import configure_ray_worker_logging
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl_train.workers.worker_utils import reduce_metrics


class RayPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        tracker: Tracking,
        tokenizer: AutoTokenizer,
        train_dataset: Optional[PromptDataset],
        inference_engine_client: InferenceEngineClient,
        generator: GeneratorInterface,
        colocate_pg: Optional[PlacementGroup] = None,
        eval_dataset: Optional[PromptDataset] = None,
    ):
        self.cfg = cfg
        self.colocate_all = cfg.trainer.placement.colocate_all
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.inference_engine_client = inference_engine_client
        self.generator = generator
        self.train_dataloader = None
        self.total_training_steps = None
        self._build_train_dataloader_and_compute_training_steps()

        self.eval_dataloader = (
            build_dataloader(self.cfg, eval_dataset, is_train=False) if eval_dataset is not None else None
        )
        self.colocate_pg = colocate_pg

        self.resume_mode = ResumeMode(cfg.trainer.resume_mode)

        self.all_metrics = {}
        self.all_timings = {}
        self.global_step = 0

        # initialized in `build_models`
        self.policy_model: PPORayActorGroup = None
        self.critic_model: Optional[PPORayActorGroup] = None
        self.ref_model: Optional[PPORayActorGroup] = None
        # used for checkpoint cleanup
        self._node_ids: Optional[List[str]] = None

        self.dynamic_sampling_state: Optional[DynamicSamplingState] = None

        self.reward_kl_controller: Optional[Union[FixedKLController, AdaptiveKLController]] = None
        self.dispatch: WorkerDispatch = None
        configure_ray_worker_logging()

    @property
    def has_critic(self) -> bool:
        """Check if critic model is configured."""
        return self.cfg.trainer.critic.model.path is not None

    def _build_train_dataloader_and_compute_training_steps(self):
        """
        Hook for constructing the training dataloader. Subclasses can override
        this to customize dataloader behavior. For instance, fully async training
        needs a batch size of 1, among other features.
        Defaults to `trainer_utils.build_dataloader` with `is_train=True`.
        """
        self.train_dataloader = build_dataloader(self.cfg, self.train_dataset, is_train=True)
        self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        if self.cfg.generator.step_wise_trajectories:
            eval_metrics = await evaluate_step_wise(
                eval_dataloader=self.eval_dataloader,
                generator=self.generator,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
            )
        else:
            eval_metrics = await evaluate(
                eval_dataloader=self.eval_dataloader,
                generator=self.generator,
                cfg=self.cfg,
                global_step=self.global_step,
                tokenizer=self.tokenizer,
            )
        return eval_metrics

    async def train(self):
        """
        Main training loop for PPO
        """
        # Initialize weight sync state between policy model and inference engines.
        with Timer("init_weight_sync_state"):
            self.init_weight_sync_state()

        # Load checkpoint state if resumption is enabled.
        if self.resume_mode != ResumeMode.NONE:
            with Timer("load_checkpoints"):
                self.global_step, _ = self.load_checkpoints()

        self.dispatch.prepare_for_weight_sync()
        if self.colocate_all:
            await self.inference_engine_client.wake_up(tags=["weights"])
        with Timer("sync_weights"):
            self.dispatch.broadcast_to_inference_engines(self.inference_engine_client)
        self.dispatch.finish_weight_sync()
        if self.colocate_all:
            await self.inference_engine_client.wake_up(tags=["kv_cache"])

        # Eval before training - always log at step 0 regardless of resumed checkpoint
        if self.cfg.trainer.eval_interval > 0 and self.cfg.trainer.eval_before_train:
            with Timer("eval", self.all_timings):
                eval_metrics = await self.eval()
                self.tracker.log(eval_metrics, step=0, commit=True)

        # initialize kl controller
        if self.cfg.trainer.algorithm.use_kl_in_reward:
            self.reward_kl_controller = get_kl_controller(self.cfg.trainer.algorithm)

        # main training loop
        pbar = tqdm(total=self.total_training_steps, initial=self.global_step, desc="Training Batches Processed")
        start_epoch = self.global_step // len(self.train_dataloader)
        self.global_step += 1  # start training at global_step 1
        for epoch in range(start_epoch, self.cfg.trainer.epochs):
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
                        generator_output: GeneratorOutput = await self.generate(generator_input)

                    if self.cfg.generator.step_wise_trajectories:
                        # NOTE: We use instance_ids from `trajectory_ids` here instead of re-using `uids`
                        # this is because in step-wise training, len(uids) != len(generator_output["response_ids"])
                        uids = [trajectory_id.instance_id for trajectory_id in generator_output["trajectory_ids"]]

                    # dynamic sampling
                    if self.cfg.trainer.algorithm.dynamic_sampling.type is not None:
                        generator_output, uids, keep_sampling = self.handle_dynamic_sampling(generator_output, uids)
                        if keep_sampling:  # continue sampling
                            # update progress bar for current batch (but not global step)
                            pbar.update(1)
                            continue

                    if self.colocate_all:
                        # if we are not continuing sampling, we sleep the inference engine
                        await self.inference_engine_client.sleep()

                    # 1.2 postprocess rewards
                    with Timer("postprocess_generator_output", self.all_timings):
                        # Extract data_sources for per-environment metrics
                        data_sources = [
                            env_extra.get("data_source") for env_extra in generator_input.get("env_extras", [])
                        ]
                        generator_output = self.postprocess_generator_output(generator_output, uids, data_sources)

                    # 2. print example just for debugging
                    vis = self.tokenizer.decode(generator_output["response_ids"][0])
                    log_example(
                        logger,
                        prompt=generator_input["prompts"][0],
                        response=vis,
                        reward=generator_output["rewards"][0],
                    )

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
                    self.dispatch.prepare_for_weight_sync()
                    if self.colocate_all:
                        await self.inference_engine_client.wake_up(tags=["weights"])
                    with Timer("sync_weights", self.all_timings):
                        self.dispatch.broadcast_to_inference_engines(self.inference_engine_client)
                    self.dispatch.finish_weight_sync()
                    if self.colocate_all:
                        await self.inference_engine_client.wake_up(tags=["kv_cache"])

                # 8. set logs
                logger.info(status)
                # log epoch info
                self.all_metrics.update({"trainer/epoch": epoch, "trainer/global_step": self.global_step})
                if self.cfg.trainer.eval_interval > 0 and (
                    self.global_step % self.cfg.trainer.eval_interval == 0
                    or self.global_step == self.total_training_steps
                ):
                    with Timer("eval", self.all_timings):
                        eval_metrics = await self.eval()
                        self.all_metrics.update(eval_metrics)

                log_payload = {
                    **self.all_metrics,
                    **{f"timing/{k}": v for k, v in self.all_timings.items()},
                }
                self.tracker.log(log_payload, step=self.global_step, commit=True)

                # Check for high tool error rate and alert via Slack
                run_name = self.cfg.trainer.get("experiment_name", None)
                check_tool_error_rate_alert(self.all_metrics, self.global_step, threshold=0.5, run_name=run_name)

                self.all_metrics = {}
                self.all_timings = {}

                # update progress bar after logging
                pbar.update(1)

                self.global_step += 1

                del training_input, generator_output

        pbar.close()
        if self.colocate_all:
            await self.inference_engine_client.sleep()
        if self.cfg.trainer.ckpt_interval > 0:
            with Timer("save_checkpoints", self.all_timings):
                self.save_checkpoints()
                logger.info("Saved final checkpoint.")
        if self.cfg.trainer.hf_save_interval > 0:
            with Timer("save_hf_model", self.all_timings):
                self.save_models()
                logger.info("Saved final model.")
        logger.info("Training done!")

    def _remove_tail_data(self, entries: List[Any]) -> List[Any]:
        """Remove tail data to have even shards in terms of *effective* samples.

        Each prompt produces `n_samples_per_prompt` samples. For data-parallel
        training we care that the total number of samples is nicely splittable
        across the (combined) data-parallel size of all enabled models.
        """
        lcm_dp_size = self.dispatch.get_lcm_dp_size()

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        # We want the largest m <= len(entries) such that:
        #   (m * n_samples_per_prompt) % lcm_dp_size == 0
        #
        # Let g = gcd(lcm_dp_size, n_samples_per_prompt). Then this is equivalent
        # to requiring m to be a multiple of (lcm_dp_size / g).
        stride = lcm_dp_size // math.gcd(lcm_dp_size, n_samples_per_prompt)
        if stride <= 1:
            # Every prompt count is valid, keep all entries.
            return entries

        kept_prompts = (len(entries) // stride) * stride
        return entries[:kept_prompts]

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        """
        Initialize the actors for training, and handle colocation logic
        """
        cfg = self.cfg
        pg = None

        use_ref_model = cfg.trainer.algorithm.use_kl_loss or cfg.trainer.algorithm.use_kl_in_reward

        if cfg.trainer.placement.colocate_all:
            num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
            num_critic_gpus = cfg.trainer.placement.critic_num_gpus_per_node * cfg.trainer.placement.critic_num_nodes
            num_ref_gpus = cfg.trainer.placement.ref_num_gpus_per_node * cfg.trainer.placement.ref_num_nodes
            num_rollout_gpus = (
                cfg.generator.num_inference_engines
                * cfg.generator.inference_engine_tensor_parallel_size
                * cfg.generator.inference_engine_pipeline_parallel_size
                * cfg.generator.inference_engine_data_parallel_size
            )
            assert (
                num_policy_gpus == num_rollout_gpus
            ), "num_policy_gpus and num_rollout_gpus must be the same when colocating all models"
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.2 if pg else 1,
                colocate_all=True,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            if use_ref_model:
                assert (
                    num_policy_gpus == num_ref_gpus
                ), "num_policy_gpus and num_ref_gpus must be the same when colocating policy and ref model"
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                assert (
                    num_policy_gpus == num_critic_gpus
                ), "num_policy_gpus and num_critic_gpus must be the same when colocating policy and critic model"
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    pg=pg,
                    num_gpus_per_actor=0.2,
                    colocate_all=True,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        else:
            if cfg.trainer.placement.colocate_policy_ref and use_ref_model:
                assert (
                    cfg.trainer.placement.policy_num_nodes == cfg.trainer.placement.ref_num_nodes
                    and cfg.trainer.placement.policy_num_gpus_per_node == cfg.trainer.placement.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate policy and ref model."

                bundles = [
                    {
                        "GPU": cfg.trainer.placement.policy_num_gpus_per_node,
                        "CPU": cfg.trainer.placement.policy_num_gpus_per_node,
                    }
                    for _ in range(cfg.trainer.placement.policy_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

            policy_model = PPORayActorGroup(
                cfg,
                cfg.trainer.placement.policy_num_nodes,
                cfg.trainer.placement.policy_num_gpus_per_node,
                PolicyWorker,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
                colocate_all=False,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            )
            if use_ref_model:
                ref_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.ref_num_nodes,
                    cfg.trainer.placement.ref_num_gpus_per_node,
                    RefWorker,
                    pg=pg,
                    num_gpus_per_actor=0.25 if pg else 1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.ref.sequence_parallel_size,
                )
            else:
                ref_model = None

            if cfg.trainer.critic.model.path:
                critic_model = PPORayActorGroup(
                    cfg,
                    cfg.trainer.placement.critic_num_nodes,
                    cfg.trainer.placement.critic_num_gpus_per_node,
                    CriticWorker,
                    num_gpus_per_actor=1,
                    colocate_all=False,
                    sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
                )
            else:
                critic_model = None

        policy_steps_per_train_batch = (
            cfg.trainer.train_batch_size // cfg.trainer.policy_mini_batch_size * cfg.trainer.update_epochs_per_batch
        )
        critic_steps_per_train_batch = 0
        if cfg.trainer.critic.model.path:
            critic_steps_per_train_batch = (
                cfg.trainer.train_batch_size // cfg.trainer.critic_mini_batch_size * cfg.trainer.update_epochs_per_batch
            )
        if not cfg.trainer.placement.colocate_all:
            refs = []
            if ref_model is not None:
                refs.extend(ref_model.async_init_model(cfg.trainer.ref.model.path))
            refs.extend(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps * policy_steps_per_train_batch,
                )
            )
            if cfg.trainer.critic.model.path:
                refs.extend(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps * critic_steps_per_train_batch,
                    )
                )
            ray.get(refs)
            ray.get(policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))
        else:
            if ref_model is not None:
                ray.get(ref_model.async_init_model(cfg.trainer.ref.model.path))
                ref_model.offload_to_cpu()
            ray.get(
                policy_model.async_init_model(
                    cfg.trainer.policy.model.path,
                    num_training_steps=self.total_training_steps * policy_steps_per_train_batch,
                )
            )
            ray.get(policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))
            policy_model.offload_to_cpu()
            if cfg.trainer.critic.model.path:
                ray.get(
                    critic_model.async_init_model(
                        cfg.trainer.critic.model.path,
                        num_training_steps=self.total_training_steps * critic_steps_per_train_batch,
                    )
                )
                critic_model.offload_to_cpu()

        self.policy_model: PPORayActorGroup = policy_model
        self.critic_model: Optional[PPORayActorGroup] = critic_model
        self.ref_model: Optional[PPORayActorGroup] = ref_model

        # Create unified dispatch that manages all actor groups
        self.dispatch = WorkerDispatch(
            cfg=self.cfg,
            policy_actor_group=policy_model,
            critic_actor_group=critic_model,
            ref_actor_group=ref_model,
        )

        # Mark all models as offloaded if colocate_all (they were offloaded above)
        if self.colocate_all:
            self.dispatch.mark_all_offloaded()

        logger.info("init policy/ref/critic models done")

    def init_weight_sync_state(self):
        """
        Setup the connection between policy model and inference engine for weight syncing.
        """
        ray.get(
            self.policy_model.async_run_ray_method(
                "pass_through", "init_weight_sync_state", self.inference_engine_client
            )
        )
        logger.info("Initialized weight sync state for policy model and inference engines.")

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training"""
        prompt_ids: List[List[int]] = generator_output["prompt_token_ids"]
        response_ids: List[List[int]] = generator_output["response_ids"]
        rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        logprobs: Optional[List[List[float]]] = generator_output.get("rollout_logprobs", None)

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
            logprobs,
        )
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
                "is_last_step": (
                    torch.tensor(generator_output["is_last_step"], dtype=torch.bool)
                    if generator_output.get("is_last_step", None) is not None
                    else None
                ),
            },
        )
        training_input.metadata = {"uids": uids}
        # padded response length
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        if self.cfg.generator.step_wise_trajectories:
            assert (
                "trajectory_ids" in generator_output
            ), "Expected `trajectory_ids` in generator output for step wise training"
            training_input.metadata["trajectory_ids"] = [
                trajectory_id.to_string() for trajectory_id in generator_output["trajectory_ids"]
            ]
            training_input.metadata["avg_response_length"] = sum(
                len(sample_response_ids)
                for sample_response_ids, is_last_step in zip(response_ids, generator_output["is_last_step"])
                if is_last_step
            ) / len(response_ids)
        else:
            training_input.metadata["avg_response_length"] = sum(
                len(sample_response_ids) for sample_response_ids in response_ids
            ) / len(response_ids)

        logger.info(f"Number of sequences before padding: {len(training_input['sequences'])}")
        training_input = self.pad_batch(training_input)
        logger.info(f"Number of sequences after padding: {len(training_input['sequences'])}")

        return training_input

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
        # NOTE: we assume that .generate returns samples in the same order as passed in
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        if not self.cfg.generator.step_wise_trajectories:
            validate_generator_output(len(input_batch["prompts"]), generator_output)

        return generator_output

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str], data_sources: Optional[List[str]] = None
    ) -> GeneratorOutput:
        """
        Converts to per token rewards and computes pass@N.

        In the future algorithm specific reward or loss mask post processing should be done here.
        """
        generator_output_for_metrics = generator_output
        uids_for_metrics = uids
        data_sources_for_metrics = data_sources
        if self.cfg.generator.step_wise_trajectories:
            generator_output_for_metrics = defaultdict(list)
            for key in generator_output:
                if isinstance(generator_output[key], list):
                    generator_output_for_metrics[key] = [
                        generator_output[key][i]
                        for i in range(len(generator_output[key]))
                        if generator_output["is_last_step"][i]
                    ]
            uids_for_metrics = [
                uid for uid, is_last_step in zip(uids, generator_output["is_last_step"]) if is_last_step
            ]
            if data_sources:
                data_sources_for_metrics = [
                    ds for ds, is_last_step in zip(data_sources, generator_output["is_last_step"]) if is_last_step
                ]

        # only use `generator_output_for_metrics` for metrics calculation
        # For step-wise training, we only calculate metrics for the last step of each trajectory
        overall_metrics = get_metrics_from_generator_output(
            generator_output_for_metrics,
            uids_for_metrics,
        )

        # these use the full generator output
        rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
        responses: List[List[int]] = generator_output["response_ids"]
        per_token_rewards: List[List[float]] = []

        # Check if rewards are already token-level (List[List[float]]) or response-level (List[float])
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: rewards is List[List[float]]
            per_token_rewards = rewards
        else:
            if self.cfg.trainer.algorithm.zero_variance_filter:
                kept_indices_set = set(zero_variance_filter(rewards, uids))
                generator_output["loss_masks"] = [
                    [0] * len(mask) if i not in kept_indices_set else mask
                    for i, mask in enumerate(generator_output["loss_masks"])
                ]
            # Response-level rewards: rewards is List[float], convert to per-token rewards
            for reward, response in zip(rewards, responses):
                per_token_reward = [0.0] * len(response)
                per_token_reward[-1] = float(reward)
                per_token_rewards.append(per_token_reward)

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        reward_metrics = {
            f"reward/avg_pass_at_{n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "reward/avg_raw_reward": overall_metrics["avg_score"],
            "reward/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }

        # Calculate per-data-source metrics if data_sources provided
        if data_sources_for_metrics:
            per_source_metrics = calculate_per_source_reward_metrics(
                generator_output_for_metrics,
                uids_for_metrics,
                data_sources_for_metrics,
                n_samples_per_prompt,
            )
            reward_metrics.update(per_source_metrics)

        self.all_metrics.update(reward_metrics)
        logger.info(
            f"reward/avg_pass_at_{n_samples_per_prompt}: {overall_metrics['pass_at_n']}, reward/avg_raw_reward: {overall_metrics['avg_score']}, reward/mean_positive_reward: {overall_metrics['mean_positive_reward']}"
        )
        # re-assign reward but now it's per token rewards
        generator_output["rewards"] = per_token_rewards
        return generator_output

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        """Calculate advantages and returns for the data batch.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["response_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["loss_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `.metadata["uids"]`: List[str]

        Adds:
            - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        token_level_rewards = data["rewards"]

        if self.cfg.generator.step_wise_trajectories:
            is_last_step = data["is_last_step"].bool()
            response_mask = data["response_mask"]
            index = np.array(data.metadata["uids"])
            adv_estimator = self.cfg.trainer.algorithm.advantage_estimator
            config = self.cfg.trainer.algorithm
            values = data["values"]
            gamma = self.cfg.trainer.algorithm.gamma
            lambd = self.cfg.trainer.algorithm.lambd
            grpo_norm_by_std = self.cfg.trainer.algorithm.grpo_norm_by_std
            last_step_rewards = token_level_rewards[is_last_step]
            # compatible with any advantage estimator
            last_step_advantages, last_step_returns = ppo_utils.compute_advantages_and_returns(
                token_level_rewards=last_step_rewards,
                response_mask=response_mask[is_last_step],
                index=index[is_last_step.cpu().numpy()],
                adv_estimator=adv_estimator,
                values=values[is_last_step] if values is not None else None,
                config=config,
                gamma=gamma,
                lambd=lambd,
                grpo_norm_by_std=grpo_norm_by_std,
            )
            traj_ids = (
                torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
            )
            num_groups = traj_ids[-1].item() + 1
            assert num_groups == len(
                last_step_advantages
            ), f"number of groups {num_groups} doesn't match the number of trajectories as given by `is_last_step` {len(last_step_advantages)}. The `is_last_step` tensor is likely malformed"
            advantages = last_step_advantages[traj_ids]
            returns = last_step_returns[traj_ids]
        else:
            advantages, returns = ppo_utils.compute_advantages_and_returns(
                token_level_rewards=token_level_rewards,
                response_mask=data["response_mask"],
                index=data.metadata["uids"],
                adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
                config=self.cfg.trainer.algorithm,
                values=data["values"],
                gamma=self.cfg.trainer.algorithm.gamma,
                lambd=self.cfg.trainer.algorithm.lambd,
                grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            )
        data["returns"] = returns
        data["advantages"] = advantages

        # remove padding while calculating metrics
        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)

        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        if self.cfg.generator.step_wise_trajectories:
            avg_rewards: float = return_sums[data["is_last_step"][: num_samples - pad_size]].mean().item()
        else:
            avg_rewards: float = return_sums.mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size, ...], data["response_mask"][: num_samples - pad_size].bool()
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_final_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )
        return data

    def dump_data(self, data: TrainingInputBatch, file_name: str):
        """
        Dump data to pickle file
        """
        data_save_dir = Path(self.cfg.trainer.export_path) / "dumped_data"
        data_save_dir.mkdir(parents=True, exist_ok=True)
        data.save(data_save_dir / f"{file_name}.pkl")

    def pad_batch(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Pad the batch to be divisible by dp size"""
        import math

        dp_size = self.dispatch.get_lcm_dp_size()
        pad_size = math.ceil(training_input.batch_size / dp_size) * dp_size - training_input.batch_size
        new_tensors = {}
        training_input.metadata["pad_size"] = pad_size
        if pad_size == 0:
            return training_input
        for key, tensor in training_input.items():
            if tensor is not None:
                additional_dims = tuple(tensor.shape[1:]) if len(tensor.shape) > 1 else ()

                if key == "is_last_step":
                    padding_tensor = torch.ones(pad_size, *additional_dims, dtype=tensor.dtype, device=tensor.device)
                elif key == "loss_mask":
                    # ensures that padding tensors don't count towards the loss
                    padding_tensor = torch.zeros(pad_size, *additional_dims, dtype=tensor.dtype, device=tensor.device)
                else:
                    # ensures all padding tensors are in a valid format by cloning `pad_size` from the original input
                    # `pad_size` is guaranteed to be smaller than batch_size
                    padding_tensor = tensor[:pad_size].clone()
                new_tensors[key] = torch.cat([tensor, padding_tensor], dim=0)

        new_training_input = TrainingInputBatch(new_tensors)
        new_training_input.metadata = {}
        new_training_input.metadata["uids"] = training_input.metadata["uids"] + [f"pad{i}" for i in range(pad_size)]
        if "trajectory_ids" in training_input.metadata:
            new_training_input.metadata["trajectory_ids"] = training_input.metadata["trajectory_ids"] + [
                f"pad{i}" for i in range(pad_size)
            ]
        for key, value in training_input.metadata.items():
            if key not in ["uids", "trajectory_ids"]:
                new_training_input.metadata[key] = copy.deepcopy(value)
        return new_training_input

    @torch.no_grad()
    def fwd_logprobs_values_reward(
        self,
        training_input: TrainingInputBatch,
    ):
        """
        Calculate values from the critic, log probs from the policy and ref model.

        Dispatch handles offload/backload automatically for all colocation configurations.

        Expects:
            - `["sequences"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `["attention_mask"]`: Integer[torch.Tensor, "batch_size seqlen"]
            - `.metadata["response_length"]`: Int

        Adds:
            - `["base_action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["action_log_probs"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["values"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        data_fwd_pass = training_input.select(keys=["sequences", "attention_mask"], metadata_keys=["response_length"])

        values = None
        base_log_probs = None
        action_log_probs = None

        # Critic forward (dispatch handles offload/backload automatically)
        if self.has_critic:
            critic_output = self.dispatch.forward("critic", data_fwd_pass)
            values = critic_output["output"]

        # Ref forward
        if self.ref_model is not None:
            ref_output = self.dispatch.forward("ref", data_fwd_pass)
            base_log_probs = ref_output["output"]
            self.dispatch.empty_cache("ref")

        # Policy forward
        policy_output = self.dispatch.forward("policy", data_fwd_pass)
        action_log_probs = policy_output["output"]

        # Empty cache after all forward passes
        self.dispatch.empty_cache()

        sequences_all: torch.Tensor = training_input["sequences"]
        # NOTE (sumanthrh): The slicing is needed to make sure that the batch dimension doesn't change for the tensordict.
        base_log_probs = base_log_probs[: len(sequences_all)] if base_log_probs is not None else None
        action_log_probs = action_log_probs[: len(sequences_all)]
        values = values[: len(sequences_all)] if values is not None else None

        training_input["base_action_log_probs"] = base_log_probs
        training_input["action_log_probs"] = action_log_probs
        training_input["values"] = values

        if self.cfg.generator.sampling_params.logprobs is not None:
            # calculates the difference in probs between inference and trainer components
            # only consider response tokens
            logprobs_diff = (
                training_input["rollout_logprobs"][training_input["loss_mask"] > 0]
                - action_log_probs[training_input["loss_mask"] > 0]
            )
            prob_diff = logprobs_diff.exp().abs()
            prob_diff_mean = prob_diff.mean().item()
            prob_diff_std = prob_diff.std().item()
            self.all_metrics.update(
                {
                    "policy/rollout_train_prob_diff_mean": prob_diff_mean,
                    "policy/rollout_train_prob_diff_std": prob_diff_std,
                }
            )
        return training_input

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Applies a penalty for KL divergence between the policy log probs and the base model log probs to the rewards."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        rewards: torch.Tensor = data["rewards"]
        base_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # single batched computation
        kl: Float[torch.Tensor, "batch_size seqlen"] = compute_approx_kl(  # type: ignore
            action_log_probs,
            base_action_log_probs,
            loss_mask=loss_masks_all,
            kl_estimator_type=self.cfg.trainer.algorithm.kl_estimator_type,
        )
        kl_max: Float[torch.Tensor, "batch_size"] = torch.max(kl.abs(), dim=-1)[0]  # noqa: F821
        kl_mean: Float[torch.Tensor, "batch_size"] = masked_mean(kl, loss_masks_all, dim=-1)  # noqa: F821

        # NOTE (erictang000): only supporting custom rewards currently
        kl_loss_coef = (
            self.reward_kl_controller.value
            if self.reward_kl_controller is not None
            else self.cfg.trainer.algorithm.kl_loss_coef
        )
        rewards = rewards - kl * max(0, kl_loss_coef)
        data["rewards"] = rewards

        avg_kl: float = kl_mean.mean().item()
        avg_kl_max: float = kl_max.mean().item()

        # update the kl controller
        if self.reward_kl_controller is not None:
            self.reward_kl_controller.update(current=avg_kl, n_steps=kl.shape[0])  # n_steps is just the batch size
        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}

        data.metadata["metrics"].update(
            {
                "avg_kl": avg_kl,
                "avg_kl_max": avg_kl_max,
                "kl_loss_coef": kl_loss_coef,
            }
        )

        self.all_metrics.update(
            {
                "loss/avg_kl": avg_kl,
                "loss/avg_kl_max": avg_kl_max,
                "loss/kl_loss_coef": kl_loss_coef,
            }
        )

        return data

    def sync_policy_weights_to_inference_engines(self) -> List[ObjectRef]:
        return self.policy_model.async_run_ray_method(
            "pass_through", "broadcast_to_inference_engines", self.inference_engine_client
        )

    def _execute_training_step(self, model: str, data: TrainingInputBatch) -> Dict[str, float]:
        """
        Execute training step for FSDP strategy using forward_backward + optim_step.

        The trainer loops over epochs and mini-batches. Workers handle micro-batching
        internally for gradient accumulation (memory efficiency).

        Args:
            model: Model name ("policy" or "critic")
            data: Training data batch

        Returns:
            Dict of reduced metrics from training
        """
        # Compute mini batch size from config (algorithm-level concept)
        n_samples = self.cfg.generator.n_samples_per_prompt
        if model == "policy":
            mini_batch_size = self.cfg.trainer.policy_mini_batch_size * n_samples
        else:
            mini_batch_size = self.cfg.trainer.critic_mini_batch_size * n_samples

        all_metrics: Dict[str, List[float]] = defaultdict(list)

        # Training loop over epochs and mini-batches
        for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
            num_mini_batches = len(data) // mini_batch_size
            for local_step in range(num_mini_batches):
                start_idx = local_step * mini_batch_size
                end_idx = (local_step + 1) * mini_batch_size
                mini_batch = data[start_idx:end_idx]

                status = self.dispatch.forward_backward(model, mini_batch)
                for k, v in status.items():
                    all_metrics[k].append(v)

                # Optimizer step after each mini batch
                grad_norm = self.dispatch.optim_step(model)
                if grad_norm is not None:
                    all_metrics["grad_norm"].append(grad_norm)

        # Compute gradient metrics once per training step (from accumulated stats)
        # Only track gradient metrics for policy, not critic
        if model == "policy" and self.cfg.trainer.policy.track_extra_gradient_metrics:
            grad_metrics = self.dispatch.compute_gradient_metrics(model)
            for k, v in grad_metrics.items():
                all_metrics[k].append(v)
            # Free GPU tensors (mean/var_numerator) to reclaim memory for inference
            self.dispatch.reset_gradient_stats(model)

        # Reduce metrics across all mini-batches and epochs
        reduced_metrics = reduce_metrics(all_metrics, ignore_keys=["grad:update_diversity"])
        return reduced_metrics

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """
        Run the training step for the policy and critic models.

        For Megatron strategy: uses ppo_train (training loop inside worker)
        For FSDP strategy: uses forward_backward + optim_step (training loop in trainer)
        """
        data.metadata["global_step"] = self.global_step
        critic_status = None

        if self.cfg.trainer.strategy == "megatron":
            # Megatron: training loop inside worker via ppo_train
            if self.has_critic:
                with Timer("critic_train", self.all_timings):
                    critic_status = self.dispatch.ppo_train("critic", data)
            with Timer("policy_train", self.all_timings):
                policy_status = self.dispatch.ppo_train("policy", data)
        else:
            # FSDP: training loop in trainer via forward_backward + optim_step
            if self.has_critic:
                with Timer("critic_train", self.all_timings):
                    critic_status = self._execute_training_step("critic", data)
            with Timer("policy_train", self.all_timings):
                policy_status = self._execute_training_step("policy", data)

        # Update metrics
        if critic_status is not None:
            for k, v in critic_status.items():
                self.all_metrics.update({f"critic/{k}": v})

        for k, v in policy_status.items():
            self.all_metrics.update({f"policy/{k}": v})

        self.dispatch.empty_cache()

        return policy_status

    def handle_dynamic_sampling(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str], bool]:
        """
        Handle dynamic sampling for the current batch.

        Accumulates the generator output and UIDs across batches if we are sampling repeatedly
        and applies the dynamic sampling strategy (i.e. filter, replace) to the current batch.
        If we hit the limit of max sample batches, we raise an error.

        Args:
            generator_output: Current batch generator output
            uids: Current batch UIDs

        Returns:
            processed_output: Filtered generator output
            processed_uids: Filtered UIDs
            keep_sampling: Whether to keep sampling
        """
        # Prepare sampling configuration
        max_sample_batches = self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches
        dynamic_sampling_config = {
            "type": self.cfg.trainer.algorithm.dynamic_sampling.type,
            "max_sample_batches": max_sample_batches,
            "min_replace_ratio": self.cfg.trainer.algorithm.dynamic_sampling.min_replace_ratio,
            "train_batch_size": self.cfg.trainer.train_batch_size,
            "n_samples_per_prompt": self.cfg.generator.n_samples_per_prompt,
        }

        if self.dynamic_sampling_state is None:
            self.dynamic_sampling_state: DynamicSamplingState = {
                "sample_batch_count": 1,
            }
        else:
            self.dynamic_sampling_state["sample_batch_count"] += 1

        # Handle dynamic sampling using utilities
        processed_output, processed_uids, keep_sampling, updated_state = trainer_utils.handle_dynamic_sampling(
            generator_output, uids, dynamic_sampling_config, self.dynamic_sampling_state
        )

        # Check max resample limit, and if we hit it, raise an error
        if (
            keep_sampling
            and max_sample_batches > 0
            and self.dynamic_sampling_state["sample_batch_count"] >= max_sample_batches
        ):
            raise RuntimeError(
                f"Exiting training loop due to hitting dynamic sampling limit for "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.type} strategy with "
                f"{self.cfg.trainer.algorithm.dynamic_sampling.max_sample_batches} max sample batches. "
                f"Please check your data difficulty distribution."
            )
        # Update state
        self.dynamic_sampling_state = updated_state

        if not keep_sampling:
            # Reset state when sampling is complete
            self.dynamic_sampling_state = None

        return processed_output, processed_uids, keep_sampling

    def _get_dp_group_models(self, rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        return model._actor_handlers[rank]

    def _get_mesh_rank(self, rank: int, model_type: str = "") -> MeshRank:
        model: PPORayActorGroup = getattr(self, model_type)
        actor_info: ActorInfo = model.actor_infos[rank]
        return actor_info.rank

    def save_checkpoints(self):
        """
        Save the model, optimizer, and training states to disk.

        Dispatch handles offload/backload automatically for all colocation configurations.
        """
        # Create global step folder structure
        global_step_folder = os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}")
        policy_save_dir = os.path.join(global_step_folder, "policy")
        critic_save_dir = os.path.join(global_step_folder, "critic")

        io.makedirs(global_step_folder, exist_ok=True)

        # Save policy checkpoint (dispatch handles offload/backload)
        self.dispatch.save_checkpoint("policy", policy_save_dir, self.tokenizer)

        # Save critic checkpoint (if it exists)
        if self.has_critic:
            self.dispatch.save_checkpoint("critic", critic_save_dir, self.tokenizer)

        # Save dataloader state
        dataloader_save_path = os.path.join(global_step_folder, "data.pt")
        try:
            dataloader_state_dict = self.train_dataloader.state_dict()
            with io.open_file(dataloader_save_path, "wb") as f:
                torch.save(dataloader_state_dict, f)
            logger.info(f"Saved dataloader state to {dataloader_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save dataloader state: {e}")

        # Save additional trainer state
        trainer_state = {
            "global_step": self.global_step,
            "config": self.cfg,
        }
        trainer_state_path = os.path.join(global_step_folder, "trainer_state.pt")
        with io.open_file(trainer_state_path, "wb") as f:
            torch.save(trainer_state, f)
        logger.info(f"Saved trainer state to {trainer_state_path}")

        # Atomic tracking - write this last after all saves succeed
        latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        with io.open_file(latest_checkpoint_file, "w") as f:
            f.write(str(self.global_step))

        logger.info(f"Successfully saved checkpoint for global_step_{self.global_step} to: {global_step_folder}")

        # Clean up old checkpoints after successful save
        with Timer("cleanup_old_checkpoints", self.all_timings):
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        if not self._node_ids:
            self._node_ids = self.dispatch.get_node_ids()
        run_on_each_node(
            self._node_ids,
            cleanup_old_checkpoints,
            self.cfg.trainer.ckpt_path,
            self.cfg.trainer.max_ckpts_to_keep,
        )
        # run on driver as well
        # NOTE (sumanthrh): the function will get called twice on the node with driver process, but it's ok because it's idempotent
        cleanup_old_checkpoints(self.cfg.trainer.ckpt_path, self.cfg.trainer.max_ckpts_to_keep)

    def load_checkpoints(self) -> Tuple[int, str]:
        """
        Load complete checkpoint state and return the global_step to resume from.
        Returns 0 if no checkpoint is loaded.

        If colocate_all is True, assumes that the policy model is currently on GPU.

        Returns:
            global_step: The global step to resume from.
            checkpoint_path: The path to the checkpoint.
        """
        checkpoint_path = None
        # Check if resumption is enabled
        if self.resume_mode == ResumeMode.NONE:
            logger.info("Checkpoint resumption disabled, starting training from scratch")
            return 0, None
        # first, let's get resume_path
        elif self.resume_mode == ResumeMode.LATEST:
            latest_checkpoint_file = os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
            if not io.exists(latest_checkpoint_file):
                logger.info("No checkpoint found, starting training from scratch")
                return 0, None
            with io.open_file(latest_checkpoint_file, "r") as f:
                ckpt_iteration = int(f.read().strip())
            checkpoint_path = os.path.join(self.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{ckpt_iteration}")
            # Run validation: Make sure ckpt folder is consistent with latest_ckpt_global_step.txt
            validate_consistency_for_latest_checkpoint(
                self.cfg.trainer.ckpt_path,
                ckpt_iteration,
                checkpoint_path,
                latest_checkpoint_file,
                self.cfg.trainer.ckpt_interval,
            )
        else:
            # Get and validate resume path
            checkpoint_path = Path(self.cfg.trainer.resume_path)
            if not checkpoint_path:
                raise ValueError("`trainer.resume_path` must be specified when resume_mode is 'from_path'")

            # Validate that it's a global_step directory
            if GLOBAL_STEP_PREFIX not in checkpoint_path.name:
                raise ValueError(
                    f"`trainer.resume_path` must point to a directory whose name starting with {GLOBAL_STEP_PREFIX}, got: {checkpoint_path}"
                )

        # Validate that the path exists
        if not io.exists(str(checkpoint_path)):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint from: {checkpoint_path}")

        # Extract global step from checkpoint path
        global_step = extract_step_from_path(Path(checkpoint_path))
        if global_step == -1:
            raise ValueError(f"Checkpoint path {checkpoint_path} is not a valid checkpoint path")
        logger.info(f"Resuming from global_step: {global_step}")

        # Define paths for different checkpoint components
        policy_ckpt_dir = os.path.join(checkpoint_path, "policy")
        critic_ckpt_dir = os.path.join(checkpoint_path, "critic")
        trainer_state_path = os.path.join(checkpoint_path, "trainer_state.pt")
        dataloader_state_path = os.path.join(checkpoint_path, "data.pt")

        # Validate that required checkpoint files exist
        if not io.exists(trainer_state_path):
            raise FileNotFoundError(f"Trainer state file not found: {trainer_state_path}")

        # 1. Load and validate trainer state
        with io.open_file(trainer_state_path, "rb") as f:
            trainer_state = torch.load(f, map_location="cpu", weights_only=False)
        saved_global_step = trainer_state.get("global_step", global_step)
        logger.info("Successfully loaded trainer state")
        if saved_global_step != global_step:
            logger.warning(f"Global step mismatch: path={global_step}, saved={saved_global_step}. Using path value.")

        # 2. Load dataloader state if available
        if io.exists(dataloader_state_path):
            try:
                with io.open_file(dataloader_state_path, "rb") as f:
                    dataloader_state = torch.load(f, map_location="cpu", weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state)
                logger.info("Successfully loaded dataloader state")
            except Exception as e:
                logger.warning(f"Failed to load dataloader state: {e}. Dataloader will start from beginning.")
        else:
            logger.warning(
                f"No dataloader state found at {dataloader_state_path}. Dataloader will start from beginning."
            )

        # 3. Load policy checkpoint (dispatch handles offload/backload)
        logger.info(f"Loading policy checkpoint from {policy_ckpt_dir}")
        self.dispatch.load_checkpoint(
            "policy",
            policy_ckpt_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        logger.info("Successfully loaded policy checkpoint")

        # 4. Load critic checkpoint if it exists and we have a critic model
        if self.has_critic:
            logger.info(f"Loading critic checkpoint from {critic_ckpt_dir}")
            self.dispatch.load_checkpoint(
                "critic",
                critic_ckpt_dir,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
            )
            logger.info("Successfully loaded critic checkpoint")

        logger.info(f"Successfully loaded complete checkpoint state from global_step_{global_step}")
        return global_step, str(checkpoint_path)

    def save_models(self):
        """
        Save the model parameters in HF format at `cfg.trainer.export_path`.

        Dispatch handles offload/backload automatically for all colocation configurations.
        """
        policy_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy")
        self.dispatch.save_hf_model("policy", policy_export_dir, self.tokenizer)

        if self.has_critic:
            critic_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "critic")
            self.dispatch.save_hf_model("critic", critic_export_dir, self.tokenizer)

        logger.info("Successfully saved model weights.")

    def update_ref_with_policy(self):
        """
        Update the reference model with the policy model weights (required by some algorithms).

        Dispatch handles offload/backload automatically for all colocation configurations.
        After this method, prepare_for_weight_sync() should be called to ensure policy is on GPU.
        """
        # TODO(tgriggs): Make policy-to-ref sync faster.
        policy_export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{self.global_step}", "policy")

        # Save policy model (dispatch handles GPU state)
        self.dispatch.save_hf_model("policy", policy_export_dir, self.tokenizer)

        # Re-initialize ref model from saved policy (dispatch handles offloading policy first)
        self.dispatch.init_model("ref", policy_export_dir)

        # Clean up temporary saved model files
        try:
            shutil.rmtree(policy_export_dir)
            logger.info(f"Cleaned up temporary policy export directory: {policy_export_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary policy export directory {policy_export_dir}: {e}")

        logger.info("Successfully updated ref model with policy model, training continues.")
