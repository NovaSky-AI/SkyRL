"""
Fleet Task Training with Tinker Backend.

This entrypoint uses Tinker (hosted) for training and inference,
combined with Fleet environments via OpenEnv for rollout collection.

Usage:
    python -m integrations.fleet.entrypoints.main_fleet_tinker \
        --model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
        --tasks-file /path/to/tasks.json \
        --dataset-file /path/to/train.parquet \
        --eval-dataset-file /path/to/validation.parquet

Environment Variables:
    TINKER_API_KEY: Tinker API key for authentication (required)
    TINKER_API_URL: Tinker service URL (optional, SDK uses default if not set)
    FLEET_API_KEY: Fleet API key for environment access
    WANDB_API_KEY: Weights & Biases API key for logging

Architecture:
    1. Load tasks from JSON file (same format as SkyRL Fleet integration)
    2. For each training step:
       a. Save current model weights for sampling
       b. Create SamplingClient from Tinker
       c. Collect rollouts using FleetTaskEnv (OpenEnv) + Tinker inference
       d. Compute GRPO advantages
       e. Train using Tinker's forward_backward + optim_step
    3. Checkpoints saved via Tinker API

Metrics (matching SkyRL):
    - reward/avg_pass_at_{n}: Pass@k across all prompts
    - reward/avg_raw_reward: Average raw reward
    - reward/{env_key}/pass_at_{n}: Per-environment pass@k
    - reward/{env_key}/avg_score: Per-environment average reward
    - eval/all/pass_at_{n}: Evaluation pass@k
    - eval/{env_key}/pass_at_{n}: Per-environment eval pass@k
"""

import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import tinker
import torch
import wandb
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Use OpenEnv's FleetTaskEnv directly for async compatibility
# (SkyRL's wrapper uses asyncio.run() which can't be called from async context)
from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARN)

# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, Dict[str, Any]] = {}


def load_tasks_from_json(tasks_file: str) -> Dict[str, Any]:
    """Load tasks from JSON file with caching."""
    if tasks_file not in _TASK_CACHE:
        expanded_path = os.path.expanduser(tasks_file)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Tasks file not found: {expanded_path}")

        with open(expanded_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Invalid JSON format in {tasks_file}")

        _TASK_CACHE[tasks_file] = {t.get("key") or t.get("task_key"): t for t in tasks}

    return _TASK_CACHE[tasks_file]


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_advantages(advantages: List[float]) -> List[float]:
    """Normalize advantages to have mean 0 and std 1."""
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
    as deviations from the group mean.
    """
    rewards = np.array(rewards)

    if group_size is None:
        group_size = len(rewards)

    n_groups = len(rewards) // group_size
    advantages = []

    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size
        group_rewards = rewards[start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_advantages = group_rewards - group_mean
        advantages.extend(group_advantages.tolist())

    remaining = len(rewards) % group_size
    if remaining > 0:
        remaining_rewards = rewards[-remaining:]
        remaining_mean = remaining_rewards.mean()
        advantages.extend((remaining_rewards - remaining_mean).tolist())

    if normalize:
        advantages = normalize_advantages(advantages)

    return advantages


def compute_pass_at_n(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> float:
    """
    Compute pass@n metric matching SkyRL's implementation.

    For each unique prompt (task_key), if ANY of the n trajectories has reward > 0,
    that counts as a "pass".
    """
    uid_to_rewards = defaultdict(list)
    for r in rollouts:
        uid = r.get("task_key", "unknown")
        uid_to_rewards[uid].append(r.get("reward", 0.0))

    if not uid_to_rewards:
        return 0.0

    # Count prompts where at least one trajectory passed
    passed = sum(1 for rewards in uid_to_rewards.values() if any(r > 0 for r in rewards))
    return passed / len(uid_to_rewards)


def compute_per_env_metrics(rollouts: List[Dict[str, Any]], n_samples_per_prompt: int) -> Dict[str, float]:
    """Compute per-environment metrics matching SkyRL's pattern."""
    env_to_rollouts = defaultdict(list)
    for r in rollouts:
        env_key = r.get("env_key", "unknown")
        env_to_rollouts[env_key].append(r)

    metrics = {}
    for env_key, env_rollouts in env_to_rollouts.items():
        rewards = [r.get("reward", 0.0) for r in env_rollouts]
        pass_at_n = compute_pass_at_n(env_rollouts, n_samples_per_prompt)
        avg_score = np.mean(rewards) if rewards else 0.0
        mean_positive = np.mean([r for r in rewards if r > 0]) if any(r > 0 for r in rewards) else 0.0

        sanitized_env = env_key.replace("/", "_")
        metrics[f"reward/{sanitized_env}/pass_at_{n_samples_per_prompt}"] = pass_at_n
        metrics[f"reward/{sanitized_env}/avg_score"] = avg_score
        metrics[f"reward/{sanitized_env}/mean_positive_reward"] = mean_positive

    return metrics


def build_system_prompt(tools: List[Dict]) -> str:
    """Build system prompt with tool definitions (matching SkyRL's FleetTaskEnv)."""
    tools_json = json.dumps(tools, indent=2)
    current_date = datetime.now().strftime("%Y-%m-%d")

    return f"""You are a helpful agent. Complete the task by calling tools.

## Current Date
Today's date is {current_date}. When dates are mentioned without a year, assume the current year ({datetime.now().year}) or a future date.

## Available Tools
{tools_json}

## Tool Call Format
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

## Error Handling
If a tool call returns an error:
- Read the error message carefully
- Do NOT repeat the same call with identical arguments
- Change your approach: use different parameters, try a different tool, or break the task into smaller steps

## Response Format
EVERY response MUST end with exactly ONE of:
1. A tool call: <tool_call>...</tool_call> - to perform an action
2. Done signal: <done> - ONLY when the task is fully complete

NEVER respond with just a message. NEVER say "feel free to ask" or offer further help.
If the task is complete, say <done>. Otherwise, make a tool call."""


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from LLM response."""
    import re

    for tag in ["tool_call", "function_call"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params", {})
                if name:
                    return {"name": name, "arguments": args}
            except json.JSONDecodeError:
                pass
    return None


def tokenize_chat(tokenizer: AutoTokenizer, chat_history: List[Dict], add_generation_prompt: bool = True) -> List[int]:
    """
    Tokenize chat history and ensure we get a plain list of token IDs.

    apply_chat_template can return different types depending on the tokenizer:
    - List[int] for some tokenizers
    - BatchEncoding dict with 'input_ids' key for others

    Tinker's ModelInput.from_ints() requires a plain list of integers.
    """
    result = tokenizer.apply_chat_template(chat_history, add_generation_prompt=add_generation_prompt, tokenize=True)
    # Handle BatchEncoding (dict-like) vs plain list
    if hasattr(result, "input_ids"):
        return list(result.input_ids)
    elif isinstance(result, dict) and "input_ids" in result:
        return list(result["input_ids"])
    else:
        return list(result)


async def collect_fleet_rollout(
    task_config: Dict[str, Any],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    max_tokens: int = 2048,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Collect a single trajectory using Fleet environment and Tinker inference.

    Uses OpenEnv's FleetTaskEnv directly with async methods for compatibility.
    """
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY environment variable must be set")

    # Load full task config from JSON
    tasks = load_tasks_from_json(tasks_file)
    task_key = task_config.get("task_key") or task_config.get("key")
    full_task_config = tasks.get(task_key)
    if not full_task_config:
        raise ValueError(f"Task '{task_key}' not found in {tasks_file}")

    # Normalize task config for OpenEnv
    normalized_config = full_task_config.copy()
    if "key" in normalized_config and "task_key" not in normalized_config:
        normalized_config["task_key"] = normalized_config["key"]
    if "env_id" in normalized_config and "env_key" not in normalized_config:
        normalized_config["env_key"] = normalized_config["env_id"]

    env_key = normalized_config.get("env_key", "unknown")

    # Create OpenEnv FleetTaskEnv directly (async-compatible)
    env = OpenEnvFleetTaskEnv(
        task_config=normalized_config,
        api_key=api_key,
        ttl_seconds=600,
        max_steps=max_turns,
    )

    turns = 0
    tool_calls = 0

    try:
        # Reset environment (async)
        obs = await env.reset_async()
        tools = obs.get("tools", [])
        task_prompt = normalized_config.get("prompt", "")

        # Build chat history with system prompt
        system_prompt = build_system_prompt(tools)
        chat_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

        # Tokenize initial prompt
        prompt_ids = tokenize_chat(tokenizer, chat_history, add_generation_prompt=True)

        all_response_ids = []
        all_logprobs = []
        loss_mask = []
        done = False
        total_reward = 0.0

        while not done and turns < max_turns:
            turns += 1

            # Prepare input for Tinker
            input_ids = tokenize_chat(tokenizer, chat_history, add_generation_prompt=True)

            # Generate with Tinker
            sampling_params = types.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )

            result = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=input_ids),
                num_samples=1,
                sampling_params=sampling_params,
            ).result()

            if not result.sequences or len(result.sequences) == 0:
                logger.warning("No sequences returned from Tinker")
                break

            sequence = result.sequences[0]
            output_ids = sequence.tokens  # SampledSequence uses 'tokens', not 'token_ids'
            output_logprobs = sequence.logprobs if sequence.logprobs else []

            # Decode output
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

            # Collect trajectory data
            all_response_ids.extend(output_ids)
            if output_logprobs:
                all_logprobs.extend(output_logprobs)
            else:
                all_logprobs.extend([0.0] * len(output_ids))
            loss_mask.extend([1] * len(output_ids))

            # Update chat history with assistant response
            chat_history.append({"role": "assistant", "content": output_text})

            # Parse tool call and step environment
            tool_call = parse_tool_call(output_text)
            agent_done = "<done>" in output_text.lower()

            if tool_call:
                tool_calls += 1
                action = {
                    "tool": tool_call["name"],
                    "params": tool_call.get("arguments", {}),
                    "done": agent_done,
                }
            else:
                action = {"done": agent_done}

            # Step environment (async)
            step_obs, reward, done, info = await env.step_async(action)

            # Add observation to chat history
            tool_result = step_obs.get("observation", {})
            if tool_result:
                if isinstance(tool_result, dict):
                    obs_content = f"Tool result:\n{json.dumps(tool_result, indent=2)}"
                else:
                    obs_content = f"Tool result:\n{tool_result}"
            elif agent_done:
                obs_content = "Task marked as complete."
            elif not tool_call:
                obs_content = (
                    'No tool call found. Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> format.'
                )
            else:
                obs_content = "Action executed."

            chat_history.append({"role": "user", "content": obs_content})

            # Add observation tokens (masked out for loss)
            obs_ids = tokenizer.encode(obs_content, add_special_tokens=False)
            all_response_ids.extend(obs_ids)
            all_logprobs.extend([0.0] * len(obs_ids))
            loss_mask.extend([0] * len(obs_ids))

            total_reward = reward
            done = done or agent_done

        return {
            "prompt_ids": prompt_ids,
            "response_ids": all_response_ids,
            "logprobs": all_logprobs,
            "loss_mask": loss_mask,
            "reward": total_reward,
            "task_key": task_key,
            "env_key": env_key,
            "turns": turns,
            "tool_calls": tool_calls,
        }

    finally:
        env.close()


async def collect_batch_rollouts(
    batch: List[Dict[str, Any]],
    tasks_file: str,
    sampling_client: tinker.SamplingClient,
    tokenizer: AutoTokenizer,
    max_turns: int = 50,
    n_samples_per_prompt: int = 1,
) -> List[Dict[str, Any]]:
    """Collect rollouts for a batch of tasks."""
    rollouts = []

    for task_config in batch:
        for _ in range(n_samples_per_prompt):
            try:
                rollout = await collect_fleet_rollout(
                    task_config=task_config,
                    tasks_file=tasks_file,
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    max_turns=max_turns,
                )
                rollouts.append(rollout)
            except Exception as e:
                logger.error(f"Failed to collect rollout for {task_config.get('task_key')}: {e}")
                rollouts.append(
                    {
                        "prompt_ids": [],
                        "response_ids": [],
                        "logprobs": [],
                        "loss_mask": [],
                        "reward": 0.0,
                        "task_key": task_config.get("task_key"),
                        "env_key": task_config.get("env_key", "unknown"),
                        "turns": 0,
                        "tool_calls": 0,
                        "error": str(e),
                    }
                )

    return rollouts


def collate_fn(batch):
    """Return batch as-is without tensor collation."""
    return batch


async def save_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    save_path: str,
    step: int,
) -> Dict[str, str]:
    """Save training checkpoint."""
    state_result = await training_client.save_state_async(name)
    state_path = (await state_result.result_async()).path

    sampler_result = await training_client.save_weights_for_sampler_async(name)
    sampler_path = (await sampler_result.result_async()).path

    checkpoint_info = {
        "name": name,
        "step": step,
        "state_path": state_path,
        "sampler_path": sampler_path,
    }

    checkpoint_file = os.path.join(save_path, "checkpoints.jsonl")
    with open(checkpoint_file, "a") as f:
        f.write(json.dumps(checkpoint_info) + "\n")

    logger.info(f"Saved checkpoint {name}: state={state_path}, sampler={sampler_path}")
    return checkpoint_info


async def main(
    model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
    tasks_file: str = None,
    dataset_file: str = None,
    eval_dataset_file: str = None,
    batch_size: int = 8,
    eval_batch_size: int = 32,
    learning_rate: float = 4e-5,
    lora_rank: int = 16,
    max_steps: int = 200,
    max_turns: int = 50,
    n_samples_per_prompt: int = 4,
    save_every: int = 10,
    eval_every: int = 20,
    seed: int = 42,
    wandb_project: str = "fleet-tinker-grpo",
    wandb_name: str = None,
    resume_from: str = None,
):
    """
    Main training loop using Tinker for training/inference and Fleet for environments.
    """
    set_seed(seed)

    # Setup paths
    if wandb_name is None:
        wandb_name = f"{model_name.split('/')[-1]}_{datetime.now().strftime('%m%d_%H%M')}"
    save_path = os.path.join("./tinker_fleet_output", wandb_name)
    os.makedirs(save_path, exist_ok=True)

    # Check for resume
    resume_from_step = 0
    load_state_path = None
    checkpoint_file = os.path.join(save_path, "checkpoints.jsonl")
    if resume_from or os.path.exists(checkpoint_file):
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoints = [json.loads(line) for line in f]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x["step"])
                resume_from_step = latest["step"]
                load_state_path = latest["state_path"]
                logger.info(f"Resuming from step {resume_from_step}")

    # Initialize WandB
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "max_turns": max_turns,
            "n_samples_per_prompt": n_samples_per_prompt,
        },
    )

    # Load datasets
    train_dataset = load_dataset("parquet", data_files=dataset_file)["train"]
    eval_dataset = load_dataset("parquet", data_files=eval_dataset_file)["train"] if eval_dataset_file else None

    logger.info(f"Loaded {len(train_dataset)} training samples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} eval samples")

    # Setup Tinker
    tinker_url = os.environ.get("TINKER_API_URL")
    tinker_api_key = os.environ.get("TINKER_API_KEY")

    service_client_kwargs = {}
    if tinker_url:
        service_client_kwargs["base_url"] = tinker_url
    if tinker_api_key:
        service_client_kwargs["api_key"] = tinker_api_key

    service_client = tinker.ServiceClient(**service_client_kwargs)
    training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=lora_rank)

    if load_state_path:
        future = await training_client.load_state_async(load_state_path)
        await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    adam_params = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataloader
    def create_dataloader(epoch: int):
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + epoch),
        )

    steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
    current_epoch = resume_from_step // steps_per_epoch
    train_dataloader = create_dataloader(current_epoch)
    train_iterator = iter(train_dataloader)

    # Skip to resume position
    batch_offset = resume_from_step % steps_per_epoch
    for _ in range(batch_offset):
        try:
            next(train_iterator)
        except StopIteration:
            break

    # Training loop
    for step in range(resume_from_step, max_steps):
        step_start = time.time()
        metrics = {"step": step, "epoch": step // steps_per_epoch}

        # Save checkpoint
        if save_every > 0 and step > 0 and step % save_every == 0:
            await save_checkpoint(training_client, f"step_{step:06d}", save_path, step)

        # Get sampling weights
        sampling_path = training_client.save_weights_for_sampler(name=f"step_{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)

        # Get batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            current_epoch += 1
            train_dataloader = create_dataloader(current_epoch)
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        # Collect rollouts
        logger.info(f"Step {step}: Collecting rollouts for {len(batch)} tasks...")
        rollout_start = time.time()

        rollouts = await collect_batch_rollouts(
            batch=batch,
            tasks_file=tasks_file,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            max_turns=max_turns,
            n_samples_per_prompt=n_samples_per_prompt,
        )

        metrics["time/rollout"] = time.time() - rollout_start

        # Filter valid rollouts
        valid_rollouts = [r for r in rollouts if r.get("response_ids") and not r.get("error")]
        if not valid_rollouts:
            logger.warning(f"Step {step}: No valid rollouts, skipping")
            continue

        # Compute GRPO advantages
        rewards = [r["reward"] for r in valid_rollouts]
        advantages = compute_advantages_grpo(rewards, group_size=n_samples_per_prompt, normalize=True)

        # Compute metrics matching SkyRL
        pass_at_n = compute_pass_at_n(valid_rollouts, n_samples_per_prompt)
        mean_positive = np.mean([r for r in rewards if r > 0]) if any(r > 0 for r in rewards) else 0.0

        metrics[f"reward/avg_pass_at_{n_samples_per_prompt}"] = pass_at_n
        metrics["reward/avg_raw_reward"] = np.mean(rewards)
        metrics["reward/mean_positive_reward"] = mean_positive
        metrics["advantage/mean"] = np.mean(advantages)
        metrics["advantage/std"] = np.std(advantages)
        metrics["rollouts/valid"] = len(valid_rollouts)
        metrics["rollouts/total"] = len(rollouts)

        # Per-environment metrics
        per_env_metrics = compute_per_env_metrics(valid_rollouts, n_samples_per_prompt)
        metrics.update(per_env_metrics)

        # Log rollout metrics (turns, tool_calls per env)
        rollout_metrics = defaultdict(list)
        for r in valid_rollouts:
            env_key = r.get("env_key", "unknown").replace("/", "_")
            rollout_metrics[f"rollout/{env_key}/turns"].append(r.get("turns", 0))
            rollout_metrics[f"rollout/{env_key}/tool_calls"].append(r.get("tool_calls", 0))

        for key, values in rollout_metrics.items():
            metrics[key] = np.mean(values)

        # Prepare training data
        training_datums = []
        for idx, rollout in enumerate(valid_rollouts):
            prompt_ids = rollout["prompt_ids"]
            response_ids = rollout["response_ids"]
            logprobs = rollout["logprobs"]
            loss_mask_data = rollout["loss_mask"]

            full_sequence = prompt_ids + response_ids
            prompt_len = len(prompt_ids)

            # Target tokens (shifted by 1)
            target_tokens = full_sequence[1:]

            # Logprobs (0 for prompt, actual for response)
            full_logprobs = [0.0] * prompt_len + logprobs
            full_logprobs = full_logprobs[1:]

            # Loss mask (0 for prompt, actual for response)
            full_mask = [0] * prompt_len + loss_mask_data
            full_mask = full_mask[1:]

            # Advantages
            advantage_value = advantages[idx]
            full_advantages = torch.zeros(len(full_sequence))
            for i in range(prompt_len, len(full_sequence)):
                if i - 1 < len(full_mask) and full_mask[i - 1] > 0:
                    full_advantages[i] = advantage_value
            full_advantages = full_advantages[1:]

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=full_sequence[:-1]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(full_logprobs)),
                    "advantages": TensorData.from_torch(full_advantages),
                },
            )
            training_datums.append(datum)

        # Training step
        logger.info(f"Step {step}: Training on {len(training_datums)} sequences...")
        train_start = time.time()

        fwd_bwd_future = training_client.forward_backward(training_datums, loss_fn="ppo")
        optim_step_future = training_client.optim_step(adam_params)

        fwd_bwd_future.result()
        optim_step_future.result()

        metrics["time/train"] = time.time() - train_start
        metrics["time/total"] = time.time() - step_start

        # Log metrics
        wandb.log(metrics, step=step)
        logger.info(
            f"Step {step}: pass@{n_samples_per_prompt}={pass_at_n:.3f}, "
            f"reward={metrics['reward/avg_raw_reward']:.3f}, time={metrics['time/total']:.1f}s"
        )

        # Evaluation
        if eval_every > 0 and eval_dataset and step % eval_every == 0:
            logger.info(f"Step {step}: Running evaluation...")
            eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)

            all_eval_rollouts = []
            for eval_batch in eval_dataloader:
                eval_rollouts = await collect_batch_rollouts(
                    batch=eval_batch,
                    tasks_file=tasks_file,
                    sampling_client=sampling_client,
                    tokenizer=tokenizer,
                    max_turns=max_turns,
                    n_samples_per_prompt=1,
                )
                all_eval_rollouts.extend([r for r in eval_rollouts if not r.get("error")])

            if all_eval_rollouts:
                eval_rewards = [r["reward"] for r in all_eval_rollouts]
                eval_pass_at_1 = compute_pass_at_n(all_eval_rollouts, 1)
                eval_per_env = compute_per_env_metrics(all_eval_rollouts, 1)

                eval_metrics = {
                    "eval/all/avg_score": np.mean(eval_rewards),
                    "eval/all/pass_at_1": eval_pass_at_1,
                    "eval/all/mean_positive_reward": (
                        np.mean([r for r in eval_rewards if r > 0]) if any(r > 0 for r in eval_rewards) else 0.0
                    ),
                    "eval/num_samples": len(all_eval_rollouts),
                }
                # Add per-env eval metrics (rename from reward/ to eval/)
                for key, value in eval_per_env.items():
                    eval_key = key.replace("reward/", "eval/")
                    eval_metrics[eval_key] = value

                wandb.log(eval_metrics, step=step)
                logger.info(f"Step {step}: eval pass@1={eval_pass_at_1:.3f}, avg_score={np.mean(eval_rewards):.3f}")

    # Save final checkpoint
    await save_checkpoint(training_client, "final", save_path, max_steps)

    wandb.finish()
    logger.info("Training completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fleet Task Training with Tinker")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--tasks-file", type=str, required=True, help="Path to tasks JSON file")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to training parquet")
    parser.add_argument("--eval-dataset-file", type=str, default=None, help="Path to eval parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--n-samples-per-prompt", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="fleet-tinker-grpo")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

    args = parser.parse_args()

    asyncio.run(
        main(
            model_name=args.model_name,
            tasks_file=args.tasks_file,
            dataset_file=args.dataset_file,
            eval_dataset_file=args.eval_dataset_file,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            lora_rank=args.lora_rank,
            max_steps=args.max_steps,
            max_turns=args.max_turns,
            n_samples_per_prompt=args.n_samples_per_prompt,
            save_every=args.save_every,
            eval_every=args.eval_every,
            seed=args.seed,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
            resume_from=args.resume_from,
        )
    )
