"""
SkyRL Trainer overrides needed for SkyRLAgent integration.

NOTE(Charlie): we need to get rid of this ASAP, hacky. Either change SkyAgent code or SkyRL code.

Two changes:
- Do not check `input_batch["prompts"]` length in `validate_generator_output()`, since it is not
  repeated n_samples_per_prompt times.
- Do not repeat the prompts in `prepare_generator_input()` since SkyAgent would repeat it.
"""

from typing import List, Dict, Any, Tuple

from loguru import logger
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import torch

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.generators.base import (
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
    BatchMetadata,
    TrainingPhase,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.dataset import PromptDataset

import asyncio
from pathlib import Path
import ray
from tqdm import tqdm

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.generators.utils import prepare_generator_input, get_metrics_from_generator_output
from skyrl_train.utils import Timer
from skyrl_train.utils.ppo_utils import (
    get_kl_controller,
    normalize_advantages_dict,
)
from skyrl_train.utils.trainer_utils import (
    validate_generator_output,
    ResumeMode,
    build_dataloader,
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    concatenate_generator_outputs,
)

from skyrl_train.utils.ppo_utils import register_advantage_estimator


@register_advantage_estimator("loop")
def compute_advantages_and_returns_loop(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    values: torch.Tensor,
    config,
    gamma,
    lambd,
    grpo_norm_by_std,
    **kwargs,
):
    from collections import defaultdict

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    id2samples = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2samples[index[i]].append((i, scores[i]))
        for group in id2samples.values():
            group_size = len(group)
            total_score = sum(score for _, score in group)
            for i, score in group:  # i is original index
                loo_baseline = 0
                if group_size == 1:
                    print("Cannot compute LOO advantage using 1 sample. 0 baseline is used")
                else:
                    loo_baseline = (total_score - score) / (group_size - 1)
                scores[i] = score - loo_baseline
        scores = scores.unsqueeze(-1) * response_mask
        return scores, scores


def validate_generator_output(input_batch: GeneratorInput, generator_output: GeneratorOutput):
    """
    Validate the generator output.

    This is exactly the same as the one in SkyRL, except that we do not use the len(input_batch["prompts"])
    since it is not repeated n_samples_per_prompt times.
    """
    if len(generator_output["response_ids"]) <= 0:
        raise RuntimeError("No outputs generated")

    # check that input prompts, response ids, and prompt token ids are all the same length
    # num_prompts = len(input_batch["prompts"])
    num_responses = len(generator_output["response_ids"])
    num_prompt_tokens = len(generator_output["prompt_token_ids"])
    # assert num_prompts == num_responses, f"Mismatch between prompts ({num_prompts}) and responses ({num_responses})"
    assert (
        num_responses == num_prompt_tokens
    ), f"Mismatch between responses ({num_responses}) and prompt_token_ids ({num_prompt_tokens})"

    # make sure all batch elements have the same length as response_ids (which should be non-zero)
    for key in generator_output:
        if isinstance(generator_output[key], list) and key in [
            "response_ids",
            "loss_masks",
            "rewards",
            "rollout_logprobs",
        ]:
            assert len(generator_output[key]) == len(
                generator_output["response_ids"]
            ), f"Generator output {key} length must be equal to response_ids length, got {len(generator_output[key])} and {len(generator_output['response_ids'])}"

    # make sure that each element of response ids and loss masks are all the same length (and token level rewards if used)
    for i, (response_ids, loss_masks, rewards) in enumerate(
        zip(generator_output["response_ids"], generator_output["loss_masks"], generator_output["rewards"])
    ):
        assert len(response_ids) == len(
            loss_masks
        ), f"Response ids and loss masks must have the same length, for sample {i} got {len(response_ids)} and {len(loss_masks)}"
        if isinstance(rewards, list):
            assert len(rewards) == len(
                response_ids
            ), f"Token rewards and response ids must have the same length, for sample {i} got {len(rewards)} and {len(response_ids)}"

        if generator_output["rollout_logprobs"]:
            assert len(response_ids) == len(
                generator_output["rollout_logprobs"][i]
            ), f"Response ids and rollout logprobs must have the same length, for sample {i} got {len(response_ids)} and {len(generator_output['rollout_logprobs'][i])}"

    # loss masks should be non-zero for at least one element for trainer
    if np.concatenate(generator_output["loss_masks"]).sum() == 0:
        logger.warning("All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!")


def build_dataloader(cfg: DictConfig, dataset: PromptDataset, is_train=True) -> StatefulDataLoader:
    """
    Build the dataloader for the training or evaluation dataset
    """
    # prepare dataloader
    batch_size = cfg.trainer.train_batch_size if is_train else cfg.trainer.eval_batch_size

    # Seed the dataloader for reproducibility.
    # NOTE: We seed the dataloader in the same fashion as VERL for exact reproducibility
    seeded_generator = torch.Generator()
    seeded_generator.manual_seed(cfg.trainer.seed)
    if is_train:
        sampler = RandomSampler(dataset, generator=seeded_generator)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        # TODO(Charlie): debug why inference http endpoint is slow when num_workers is 8
        num_workers=0 if cfg.generator.enable_http_endpoint else 8,
        drop_last=True if is_train else False,
        sampler=sampler,
    )
    if is_train:
        logger.info(f"Total steps: {len(dataloader) * cfg.trainer.epochs}")
    else:
        logger.info(f"Validation set size: {len(dataloader)}")

    return dataloader


def prepare_generator_input(
    prompts: List[Any],
    n_samples_per_prompt: int,
    sampling_params: Dict[str, Any],
    default_env_class: str,
    training_phase: TrainingPhase,
    global_step: int,
) -> Tuple[GeneratorInput, List[str]]:
    """Prepares the generator input for training and eval

    Args:
        prompts (List[Any]): list of prompts
        n_samples_per_prompt (int): how many samples to create per prompt
        sampling_params (Dict[str, Any]): sampling parameters
        default_env_class (str): env class to use if env class missing from prompts
        training_phase (TrainingPhase): training or eval
        global_step (int): current global step

    Returns:
        Tuple[GeneratorInput, List[str]]: generator input and list of uuids
    """
    # skyagent's AgentRunner will repeat trajectories internally
    all_prompts = [prompt["prompt"] for prompt in prompts]

    # all the other columns are env_extras
    env_extras = [prompt["env_extras"] for prompt in prompts]

    # But for other items like uuids or env classes - repeat by `n_samples_per_prompt` because it is used by
    # the trainer and for metrics
    all_envs = [
        prompt["env_class"] if prompt["env_class"] is not None else default_env_class
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]
    # Create TrajectoryID objects - one UID per row, repetition_id for multiple samples
    trajectory_ids = []
    uids = []
    for _, prompt in enumerate(prompts):
        uid: str = prompt["uid"]

        # Create TrajectoryID for each repetition
        for repetition_id in range(n_samples_per_prompt):
            trajectory_ids.append(TrajectoryID(instance_id=uid, repetition_id=repetition_id))
            uids.append(uid)

    generator_input: GeneratorInput = {
        "prompts": all_prompts,
        "env_classes": all_envs,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
        "trajectory_ids": trajectory_ids,
        "batch_metadata": BatchMetadata(global_step=global_step, training_phase=training_phase),
    }

    return generator_input, uids


class SkyRLAgentPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # reinitialize with new dataloader function for exact reproducibility across backends
        self.train_dataloader = build_dataloader(self.cfg, self.train_dataset, is_train=True)
        self.total_training_steps = len(self.train_dataloader) * self.cfg.trainer.epochs
        self.eval_dataloader = (
            build_dataloader(self.cfg, self.eval_dataset, is_train=False) if self.eval_dataset is not None else None
        )

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
        # Here SkyAgent would return a repeated output (n_samples_per_prompt times)
        generator_output: GeneratorOutput = await self.generator.generate(input_batch)

        # add rollout metrics to self.all_metrics
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])

        validate_generator_output(input_batch, generator_output)

        return generator_output

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
            self.policy_model.offload_to_cpu(offload_optimizer=True, offload_model=False)
            asyncio.run(self.inference_engine_client.wake_up(tags=["weights"]))
        with Timer("sync_weights"):
            ray.get(self.sync_policy_weights_to_inference_engines())
        if self.colocate_all:
            with Timer("offload_policy_model_to_cpu"):
                self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
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
                        self.policy_model.offload_to_cpu(offload_optimizer=True, offload_model=False)
                        asyncio.run(self.inference_engine_client.wake_up(tags=["weights"]))
                    with Timer("sync_weights", self.all_timings):
                        ray.get(self.sync_policy_weights_to_inference_engines())
                    if self.colocate_all:
                        with Timer("offload_policy_model_to_cpu"):
                            self.policy_model.offload_to_cpu(offload_optimizer=False, offload_model=True)
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

    @torch.no_grad()
    async def eval(self):
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        # NOTE: We've only injected the custom `prepare_generator_input` function here

        eval_dataloader = self.eval_dataloader
        generator = self.generator
        cfg = self.cfg
        global_step = self.global_step
        tokenizer = self.tokenizer

        # 1. Get all generator outputs
        generator_outputs: List[GeneratorOutput] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []
        sampling_params = cfg.generator.eval_sampling_params
        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for _, prompts in enumerate(eval_dataloader):
            pbar.update(1)
            generator_input, uids = prepare_generator_input(
                prompts,
                cfg.generator.eval_n_samples_per_prompt,
                get_sampling_params_for_backend(cfg.generator.backend, sampling_params),
                cfg.environment.env_class,
                "eval",
                global_step,
            )
            generator_output: GeneratorOutput = await generator.generate(generator_input)
            validate_generator_output(generator_input, generator_output)
            generator_outputs.append(generator_output)
            concat_all_envs.extend(generator_input["env_classes"])
            concat_env_extras.extend(generator_input["env_extras"])
            concat_uids.extend(uids)
        concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)

        # Extract data_sources from env_extras
        concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
        vis = tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Eval output example: {vis}")

        # 2. Group data by data source and calculate per-dataset metrics
        eval_metrics = calculate_per_dataset_metrics(
            concat_generator_outputs, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
        )

        # 3. Calculate overall metrics across all datasets
        overall_metrics = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
        eval_metrics.update(
            {
                "eval/all/avg_score": overall_metrics["avg_score"],
                f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
                "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
            }
        )

        # 4. Prepare dumping data
        # TODO[Ben] update this to be cloud-compatible
        if cfg.trainer.dump_eval_results:
            with Timer("dump_eval_results"):
                data_save_dir = (
                    Path(cfg.trainer.export_path)
                    / "dumped_evals"
                    / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
                )
                data_save_dir.mkdir(parents=True, exist_ok=True)
                dump_per_dataset_eval_results(
                    data_save_dir,
                    tokenizer,
                    concat_generator_outputs,
                    concat_data_sources,
                    concat_all_envs,
                    concat_env_extras,
                    eval_metrics,
                )

        return eval_metrics
