from typing import List, Optional, Union, Dict
from collections import defaultdict
from loguru import logger
import torch
import copy

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.generators.base import GeneratorOutput, GeneratorInput
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.generators.utils import get_metrics_from_generator_output
from skyrl_train.utils import ppo_utils
from skyrl_train.trainer import RayPPOTrainer

from examples.step_wise.step_wise_generator import StepWiseGeneratorOutput

from examples.step_wise.step_wise_evaluate import evaluate
import numpy as np


def compute_advantages_step_wise(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    adv_estimator: str,
    values: torch.Tensor,
    config,
    gamma,
    lambd,
    grpo_norm_by_std,
    is_last_step,
    **kwargs,
):
    """
    A custom advantage estimator where the inputs are represented as step level turns

    Assumes outcome rewards assigned to the final step
    """
    is_last_step = is_last_step.bool()

    with torch.no_grad():
        # calculate for the last step only and then broadcast to all steps
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
            **kwargs,
        )

        traj_ids = torch.cat([torch.tensor([False], device=is_last_step.device), is_last_step[:-1]]).int().cumsum(dim=0)
        num_groups = traj_ids[-1].item() + 1
        assert num_groups == len(
            last_step_advantages
        ), f"number of groups {num_groups} doesn't match the number of trajectories as given by `is_last_step` {len(last_step_advantages)}. The `is_last_step` tensor is likely malformed"
        advantages = last_step_advantages[traj_ids]
        returns = last_step_returns[traj_ids]

    return advantages, returns


class StepWiseTrainer(RayPPOTrainer):
    def convert_to_training_input(
        self, generator_output: StepWiseGeneratorOutput, uids: List[str]
    ) -> TrainingInputBatch:
        """Converts lists to a padded batch of tensors for training"""
        prompt_ids: List[List[int]] = generator_output["prompt_token_ids"]
        response_ids: List[List[int]] = generator_output["response_ids"]
        rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        logprobs: Optional[List[List[float]]] = generator_output.get("rollout_logprobs", None)

        # overrwrite uids
        uids = [f"{trajectory_id.instance_id}" for trajectory_id in generator_output["trajectory_ids"]]

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
                "is_last_step": torch.tensor(generator_output["is_last_step"]),
            },
        )
        training_input.metadata = {
            "uids": uids,
            "trajectory_ids": [
                f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}"
                for trajectory_id in generator_output["trajectory_ids"]
            ],
        }
        # padded response length
        num_trajectories = training_input["is_last_step"].sum().item()
        training_input.metadata["response_length"] = response_masks_tensor.shape[1]
        training_input.metadata["avg_response_length"] = (
            sum(
                len(sample_response_ids)
                for sample_response_ids, is_last_step in zip(response_ids, generator_output["is_last_step"])
                if is_last_step
            )
            / num_trajectories
        )

        logger.info(f"Number of sequences before padding: {len(training_input['sequences'])}")
        training_input = self.pad_batch(training_input)
        logger.info(f"Number of sequences after padding: {len(training_input['sequences'])}")
        return training_input

    @torch.no_grad()
    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        """
        Converts to per token rewards and computes pass@N.

        In the future algorithm specific reward or loss mask post processing should be done here.
        """
        # overrwrite uids
        uids = [f"{trajectory_id.instance_id}" for trajectory_id in generator_output["trajectory_ids"]]

        # only calculate for "is_last_step" in generator_output
        generator_output_last_step = defaultdict(list)
        for key in generator_output:
            if isinstance(generator_output[key], list):
                generator_output_last_step[key] = [
                    generator_output[key][i]
                    for i in range(len(generator_output[key]))
                    if generator_output["is_last_step"][i]
                ]
        uids_last_step = [uid for uid, is_last_step in zip(uids, generator_output["is_last_step"]) if is_last_step]
        mean_raw_reward, pass_at_n = get_metrics_from_generator_output(
            generator_output_last_step,
            uids_last_step,
        )

        rewards: Union[List[float], List[List[float]]] = generator_output["rewards"]
        responses: List[List[int]] = generator_output["response_ids"]
        per_token_rewards: List[List[float]] = []

        # Check if rewards are already token-level (List[List[float]]) or response-level (List[float])
        if rewards and isinstance(rewards[0], list):
            # Token-level rewards: rewards is List[List[float]]
            per_token_rewards = rewards
        else:
            # Response-level rewards: rewards is List[float], convert to per-token rewards
            for reward, response in zip(rewards, responses):
                per_token_reward = [0.0] * len(response)
                per_token_reward[-1] = float(reward)
                per_token_rewards.append(per_token_reward)

        n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt

        reward_metrics = {
            f"reward/avg_pass_at_{n_samples_per_prompt}": pass_at_n,
            "reward/avg_raw_reward": mean_raw_reward,
        }
        self.all_metrics.update(reward_metrics)
        logger.info(f"reward/avg_pass_at_{n_samples_per_prompt}: {pass_at_n}, reward/avg_raw_reward: {mean_raw_reward}")

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
            - `["values"]`: Float[torch.Tensor, "batch_size"]
            - `["rewards"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `.metadata["uids"]`: List[str]

        Adds:
            - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
            - `["returns"]`: Float[torch.Tensor, "batch_size seqlen"]
        """
        token_level_rewards = data["rewards"]
        # NOTE: as such, padding rows can be ignored here, but
        # we will need to re-pad the advantages and returns tensors
        # in any case, so we retain them for simplicity
        advantages, returns = compute_advantages_step_wise(
            token_level_rewards=token_level_rewards,
            response_mask=data["response_mask"],
            index=np.array(data.metadata["uids"]),
            adv_estimator=self.cfg.trainer.algorithm.advantage_estimator,
            config=self.cfg.trainer.algorithm,
            values=data["values"],
            gamma=self.cfg.trainer.algorithm.gamma,
            lambd=self.cfg.trainer.algorithm.lambd,
            grpo_norm_by_std=self.cfg.trainer.algorithm.grpo_norm_by_std,
            is_last_step=data["is_last_step"],
        )
        data["returns"] = returns
        data["advantages"] = advantages
        # remove padding while calculating metrics
        pad_size = data.metadata["pad_size"]
        num_samples = len(token_level_rewards)
        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        avg_rewards: float = return_sums[data["is_last_step"][: num_samples - pad_size]].mean().item()

        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")

        valid_advantages = torch.masked_select(data["advantages"], data["response_mask"].bool())[
            : num_samples - pad_size, ...
        ]
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_rewards": avg_rewards,
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

        # Skip validation.
        # NOTE (sumanthrh): `validate_generator_output` checks if the number of
        # rows in the `input_batch` is the same as that in `generator_output`
        # In step wise training, since each step of a row / trajectory
        # in `input_batch` is represented as a row in `generator_output`,
        # this check doesn't apply and it is skipped.
        # validate_generator_output(input_batch, generator_output)

        return generator_output

    def pad_batch(self, training_input: TrainingInputBatch) -> TrainingInputBatch:
        """Pad the batch to be divisible by dp size"""
        import math

        dp_size = self.policy_model.actor_infos[0].rank.dp_size
        if self.critic_model is not None:
            dp_size = math.lcm(dp_size, self.critic_model.actor_infos[0].rank.dp_size)
        if self.ref_model is not None:
            dp_size = math.lcm(dp_size, self.ref_model.actor_infos[0].rank.dp_size)

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
        new_training_input.metadata["trajectory_ids"] = training_input.metadata["trajectory_ids"] + [
            f"pad{i}" for i in range(pad_size)
        ]
        for key, value in training_input.metadata.items():
            if key not in ["uids", "trajectory_ids"]:
                new_training_input.metadata[key] = copy.deepcopy(value)
        return new_training_input

    @torch.no_grad()
    async def eval(self) -> Dict[str, float]:
        """
        Run generation and scoring on the evaluation dataset.

        The eval metrics are recorded after having finished training `self.global_step` steps.
        Metrics recorded in global_step 0 corresponds to evaluations before training.

        Returns:
            A dictionary of evaluation metrics.
        """
        eval_metrics = await evaluate(
            eval_dataloader=self.eval_dataloader,
            generator=self.generator,
            cfg=self.cfg,
            global_step=self.global_step,
            tokenizer=self.tokenizer,
        )
        return eval_metrics
