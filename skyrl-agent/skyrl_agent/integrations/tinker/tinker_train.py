import asyncio
import json
import logging
import os
import pprint
import random
import time
from datetime import datetime
from typing import Any, Literal, List, Dict, cast
from contextlib import contextmanager

import chz
import numpy as np
import tinker
import torch
import wandb
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers.models.auto.tokenization_auto import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from skyrl_agent import AutoAgentRunner

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")


@contextmanager
def timed(key: str, metrics: dict[str, Any]):
    logger.info(f"Starting {key}")
    tstart = time.time()
    yield
    logger.info(f"{key} took {time.time() - tstart:.2f} seconds")
    metrics[f"time/{key}"] = time.time() - tstart


safezip = cast(type[zip], lambda *args, **kwargs: zip(*args, **kwargs, strict=True))


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have mean 0 and std 1 (standard normalization)."""
    if not advantages or len(advantages) == 1:
        return advantages
    mean = np.mean(advantages)
    std = np.std(advantages)
    if std < 1e-8:
        return [0.0] * len(advantages)
    return [(a - mean) / (std + 1e-8) for a in advantages]


def compute_advantages_grpo(
    rewards: List[float],
    group_size: int = None,
    normalize: bool = True,
) -> List[float]:
    """
    GRPO (Group Relative Policy Optimization) advantage estimation.

    For each group of trajectories from the same prompt, compute advantages
    as deviations from the group mean. This is particularly useful for
    best-of-N sampling scenarios.

    Reference: https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        rewards: List of rewards for all trajectories
        group_size: Number of trajectories per prompt group.
                   If None, treats all as one group.
        normalize: Whether to apply additional global normalization

    Returns:
        List of advantages, same length as rewards
    """
    rewards = np.array(rewards)

    if group_size is None:
        # Treat all trajectories as one group (equivalent to simple baseline)
        group_size = len(rewards)

    n_groups = len(rewards) // group_size
    advantages = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]

        # GRPO: advantage = reward - mean(group_rewards)
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        advantages.extend(group_advantages.tolist())

    # Handle remaining trajectories if not evenly divisible
    remaining = len(rewards) % group_size
    assert remaining == 0, f"Remaining trajectories: {remaining} is not divisible by group_size: {group_size}"

    # Optional: Apply global normalization for extra stability
    if normalize:
        advantages = normalize_advantages(advantages)

    return advantages


def compute_kl_sample_train(data_D: List[tinker.Datum], training_logprobs_D: List[torch.Tensor]) -> Dict[str, float]:
    """Compute KL divergence metrics between sampling and training logprobs."""
    all_diffs: list[torch.Tensor] = []
    all_sampling_logprobs: list[torch.Tensor] = []

    for datum, training_logprobs in safezip(data_D, training_logprobs_D):
        # Get logprobs from sampling
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        # Extract only action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            logprob_diff = sampling_logprobs_actions - training_logprobs_actions
            all_diffs.append(logprob_diff)
            all_sampling_logprobs.append(sampling_logprobs_actions)

    assert all_diffs
    flat_diffs = torch.cat(all_diffs)
    kl_sample_train_v1 = flat_diffs.mean().item()
    kl_sample_train_v2 = 0.5 * (flat_diffs**2).mean().item()

    flat_sampling_logprobs = torch.cat(all_sampling_logprobs)
    entropy_sample = -flat_sampling_logprobs.mean().item()
    return {
        "optim/kl_sample_train_v1": kl_sample_train_v1,
        "optim/kl_sample_train_v2": kl_sample_train_v2,
        "optim/entropy": entropy_sample,
    }


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3-32B"
    batch_size: int = 64
    eval_batch_size: int = 1024
    learning_rate: float = 4e-5
    lora_rank: int = 16
    seed: int = 0
    max_steps: int = 200
    save_every: int = 2
    eval_every: int = 10
    resume_exp_name: str = None

    skyrl_agent_task_yaml: str = None
    dataset_file: str = None  # Path to the training dataset parquet file
    eval_dataset_file: str = None  # Path to the evaluation dataset parquet file

    # Loss function configuration
    loss_fn: Literal["importance_sampling", "ppo", "custom_ppo"] = "ppo"
    # Options:
    #   "ppo" or "importance_sampling": Use Tinker's built-in loss (forward_backward)

    # GRPO (Group Relative Policy Optimization) settings
    group_size: int = 8  # Trajectories per prompt group (None = auto-infer from task yaml)
    normalize_advantages: bool = True  # Apply global normalization after group-relative computation

    wandb_project: str | None = None
    wandb_name: str | None = None
    log_dir: str | None = None


async def save_checkpoint_async(
    training_client: tinker.TrainingClient,
    name: str,
    log_path: str,
    loop_state: dict[str, Any],
    kind: Literal["state", "sampler", "both"] = "state",
) -> dict[str, str]:
    """Save model checkpoint.
    Args:
        training_client: Training client to save from
        name: Name for the checkpoint
        log_path: Path to the log directory, where we can find checkpoints.jsonl file
    Returns:
        Path to the saved checkpoint
    """
    futures = {}
    if kind in ["state", "both"]:
        futures["state"] = await training_client.save_state_async(name)
    if kind in ["sampler", "both"]:
        futures["sampler"] = await training_client.save_weights_for_sampler_async(name)

    results = {k: await v.result_async() for k, v in futures.items()}
    paths = {k + "_path": v.path for k, v in results.items()}
    logger.info(f"Saved checkpoints: {paths}")
    full_dict = {"name": name, **loop_state, **paths}
    with open(os.path.join(log_path, "checkpoints.jsonl"), "a") as f:
        f.write(json.dumps(full_dict) + "\n")

    return paths


def collate_fn(batch):
    """Custom collate function that returns batch as-is without tensor collation.

    This is needed because the agent runner expects to handle the raw batch data
    through build_generator_input, rather than having PyTorch stack tensors.
    """
    return batch


async def main(config: Config):
    # Set random seed for reproducibility
    set_seed(config.seed)

    # Setup logging
    if config.resume_exp_name:
        wandb_name = config.resume_exp_name
    else:
        wandb_name = config.wandb_name or config.model_name.split("/")[-1]
        wandb_name += "_" + datetime.now().strftime("%m%dT%H:%M:%S")
    save_path = os.path.join("./tinker_output", wandb_name)
    os.makedirs(save_path, exist_ok=True)

    # read the most recent checkpoint
    checkpoint_path = os.path.join(save_path, "checkpoints.jsonl")
    load_state_path = None  # Initialize to None
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoints = [json.loads(line) for line in f]
        most_recent_checkpoint = max(checkpoints, key=lambda x: x["policy_iteration_step"])
        resume_from_step = most_recent_checkpoint["policy_iteration_step"]
        load_state_path = most_recent_checkpoint["state_path"]
        print(f"Resuming training from step {resume_from_step}")
    else:
        resume_from_step = 0
        print("Starting training from scratch")

    wandb.init(
        project=config.wandb_project,
        config=chz.asdict(config),
        dir=str(config.log_dir) if config.log_dir else None,
        name=wandb_name,
    )

    # dataset and dataloader
    train_dataset = load_dataset("parquet", data_files=config.dataset_file)["train"]
    eval_dataset = load_dataset("parquet", data_files=config.eval_dataset_file)["train"]

    # Calculate steps per epoch for tracking
    steps_per_epoch = (len(train_dataset) + config.batch_size - 1) // config.batch_size
    logger.info(f"Dataset size: {len(train_dataset)}, Steps per epoch: {steps_per_epoch}")

    # Create function to get dataloader for a specific epoch
    def create_train_dataloader(epoch: int):
        """Create dataloader with epoch-specific seed for different shuffle orders."""
        return DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(config.seed + epoch),  # Different shuffle per epoch
        )

    # Initialize iterator state for resuming
    current_epoch = resume_from_step // steps_per_epoch
    batch_offset_in_epoch = resume_from_step % steps_per_epoch

    train_dataloader = create_train_dataloader(current_epoch)
    train_iterator = iter(train_dataloader)

    # Skip batches within the current epoch if resuming mid-epoch
    if batch_offset_in_epoch > 0:
        logger.info(f"Resuming from epoch {current_epoch}, batch {batch_offset_in_epoch}/{steps_per_epoch}")
        for _ in range(batch_offset_in_epoch):
            next(train_iterator)

    # Setup agent (tinker training client)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name, rank=config.lora_rank
    )
    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    adam_params = types.AdamParams(learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    skyrl_agent_task_yaml_path = config.skyrl_agent_task_yaml
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # training loop
    for policy_iteration_step in range(resume_from_step, config.max_steps):
        print("=" * 10 + f" Step {policy_iteration_step} " + "=" * 10)
        metrics = {
            "step": policy_iteration_step,
            "epoch": policy_iteration_step // steps_per_epoch,
            "batch_in_epoch": policy_iteration_step % steps_per_epoch,
        }

        # save model
        if config.save_every > 0 and policy_iteration_step > 0 and policy_iteration_step % config.save_every == 0:
            await save_checkpoint_async(
                training_client,
                f"{policy_iteration_step:06d}",
                log_path=save_path,
                kind="state",
                loop_state={"policy_iteration_step": policy_iteration_step},
            )

        sampling_path = training_client.save_weights_for_sampler(name=f"{policy_iteration_step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        agent_generator = AutoAgentRunner.from_task(
            skyrl_agent_task_yaml_path, infer_engine=sampling_client, tokenizer=tokenizer
        )

        if policy_iteration_step % config.eval_every == 0:
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collate_fn
            )
            data_source_rewards = {}
            for batch in eval_dataloader:
                input_batch = batch
                rollouts = await agent_generator.run(input_batch, val_mode=True)
                traj_rewards_list = rollouts["traj_rewards"]
                for data, reward in zip(input_batch, traj_rewards_list):
                    data_source = data["data_source"]
                    if data_source not in data_source_rewards:
                        data_source_rewards[data_source] = []
                    data_source_rewards[data_source].append(reward)
            # get avg reward per data source
            for data_source, rewards in data_source_rewards.items():
                metrics[f"eval/reward/mean/{data_source}"] = np.mean(rewards)

        # Collect rollouts using AgentRunner
        print(f"🎲 Start collecting episodes at step {policy_iteration_step}")
        st = time.time()

        # Get next batch, handling epoch transitions
        try:
            input_batch = next(train_iterator)
        except StopIteration:
            # Start new epoch with different shuffle order
            current_epoch += 1
            logger.info(f"Starting epoch {current_epoch} with new shuffle order")
            train_dataloader = create_train_dataloader(current_epoch)
            train_iterator = iter(train_dataloader)
            input_batch = next(train_iterator)

        rollouts = await agent_generator.run(input_batch, val_mode=False)
        metrics["time/sample"] = time.time() - st
        # rollout time
        print(f"Rollout time: {metrics['time/sample']}")

        # Write rollout_metrics to wandb
        rollout_metrics = rollouts.get("rollout_metrics", {})
        wandb.log({f"rollout/{k}": v for k, v in rollout_metrics.items()}, step=policy_iteration_step)

        # Extract rollout data
        prompt_token_ids = rollouts["prompt_token_ids"]  # List of prompt token IDs
        response_ids = rollouts["response_ids"]  # List of response token IDs
        traj_rewards_list = rollouts["traj_rewards"]  # List of rewards (binary: 0 or 1)
        loss_masks = rollouts["loss_masks"]  # List of loss masks
        sampled_logprobs = rollouts["rollout_logprobs"]  # List of sampled logprobs
        num_steps_per_trajectory = rollouts["episode_nums"]  # List of number of steps per trajectory

        actual_batch_size = len(response_ids)
        logger.info(f"Processing {actual_batch_size} rollouts for training")

        # Compute advantages using GRPO (Group Relative Policy Optimization)
        all_returns = [float(r) for r in traj_rewards_list]

        # Determine group size for GRPO
        group_size = config.group_size
        if group_size is None:
            # Try to infer from task config
            from omegaconf import OmegaConf

            task_config = OmegaConf.load(skyrl_agent_task_yaml_path)
            group_size = task_config.generator.get("num_trajectories", 1)
            logger.info(f"Auto-inferred group_size={group_size} from task config")

        # Compute GRPO advantages
        logger.info(f"Computing GRPO advantages: group_size={group_size}, normalize={config.normalize_advantages}")
        all_advantages = compute_advantages_grpo(
            all_returns, group_size=group_size, normalize=config.normalize_advantages
        )
        # broadcast advantages to num_steps per trajectory
        step_advantages = []
        for idx, num_steps in enumerate(num_steps_per_trajectory):
            step_advantages.extend([all_advantages[idx]] * num_steps)

        metrics["reward/mean"] = np.mean(all_returns)
        metrics["reward/max"] = np.max(all_returns)
        metrics["reward/min"] = np.min(all_returns)
        metrics["advantage/mean"] = np.mean(all_advantages)
        metrics["advantage/std"] = np.std(all_advantages)

        # Prepare training datums compatible with Tinker API
        # For each trajectory, we need to provide:
        # - model_input: the full sequence (prompt + response)
        # - loss_fn_inputs: target_tokens, advantages, logprobs (if available), mask
        training_datums = []
        for idx in range(actual_batch_size):
            # Concatenate prompt and response to get full sequence
            full_sequence = prompt_token_ids[idx] + response_ids[idx]
            prompt_len = len(prompt_token_ids[idx])

            # Target tokens are same as input (autoregressive training)
            target_tokens = full_sequence[1:]
            logprobs = ([0] * prompt_len + sampled_logprobs[idx])[1:]

            # Base mask: 0 for prompt, loss_mask value for response
            mask = [0] * prompt_len + loss_masks[idx]

            # Advantages: broadcast the single advantage value across all response tokens
            advantage_value = step_advantages[idx]
            advantages = torch.zeros(len(full_sequence))
            # Only apply advantage to response tokens that are not masked
            assert len(mask) == len(full_sequence), f"Mask length mismatch: {len(mask)} vs {len(full_sequence)}"
            for i in range(prompt_len, len(full_sequence)):
                if mask[i] > 0:
                    advantages[i] = advantage_value
            advantages = advantages[1:]
            mask = mask[1:]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(logprobs)),
                    "advantages": TensorData.from_torch(advantages),
                },
            )
            training_datums.append(datum)

        # Training step
        print(f"🎈 Start training at step {policy_iteration_step}")
        st = time.time()

        # Use Tinker's built-in loss function ("ppo" or "importance_sampling")
        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn=config.loss_fn)
        # Optimize
        optim_step_future = training_client.optim_step(adam_params)
        fwd_bwd_result = fwd_bwd_future.result()

        # Extract training logprobs from loss_fn_outputs
        training_logprobs_D: list[torch.Tensor] = []
        for output in fwd_bwd_result.loss_fn_outputs:
            training_logprobs = output["logprobs"].to_torch()
            training_logprobs_D.append(training_logprobs)
        # with timed("compute_kl_sample_train", metrics):
        #     kl_sample_train_metrics = compute_kl_sample_train(training_datums, training_logprobs_D)
        #     metrics.update(kl_sample_train_metrics)

        _ = optim_step_future.result()
        metrics["time/train"] = time.time() - st

        pprint.pprint(metrics)
        wandb.log(metrics, step=policy_iteration_step)

    # Save final checkpoint
    if config.save_every > 0:
        await save_checkpoint_async(
            training_client,
            "final",
            log_path=save_path,
            kind="both",
            loop_state={"policy_iteration_step": config.max_steps},
        )

    wandb.finish()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
