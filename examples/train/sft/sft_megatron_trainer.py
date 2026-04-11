"""
Minimal SFT (Supervised Fine-Tuning) trainer using WorkerDispatch with Megatron backend.

This script demonstrates SFT using the same forward_backward interface as RL training,
but with loss_fn="cross_entropy" to compute simple negative log-likelihood loss.
It uses the Megatron backend with tensor and pipeline parallelism.

Usage:
    # First, make sure you have Ray installed and 4 GPUs available
    uv run --isolated --extra megatron python examples/train/sft/sft_megatron_trainer.py

This example:
1. Loads a small subset of the Alpaca dataset
2. Tokenizes examples into prompt + completion format
3. Uses WorkerDispatch.forward_backward(loss_fn="cross_entropy") for SFT
4. Demonstrates Megatron parallelism (TP=2, PP=2) for supervised fine-tuning
"""

import ray
import torch
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer
from tqdm import tqdm

from ray.util.placement_group import placement_group

from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import PolicyWorker
from skyrl.train.utils.utils import initialize_ray, validate_cfg, ResolvedPlacementGroup
from skyrl.train.utils import get_ray_pg_ready_with_timeout


def get_sft_megatron_config() -> SkyRLTrainConfig:
    """Get config with SFT-specific overrides for Megatron backend."""
    cfg = SkyRLTrainConfig()

    # Use Megatron strategy
    cfg.trainer.strategy = "megatron"

    # Model (matching the Megatron RL example)
    cfg.trainer.policy.model.path = "Qwen/Qwen3-0.6B"

    # SFT doesn't use an inference engine or ref model, so disable colocation and KL
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False

    # Megatron needs multiple GPUs for TP/PP
    cfg.trainer.placement.policy_num_gpus_per_node = 4

    # Megatron parallelism config: TP=2, PP=2 across 4 GPUs
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.context_parallel_size = 1

    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.logger = "console"

    validate_cfg(cfg)
    return cfg


def tokenize_sft_example(example: dict, tokenizer, max_length: int = 512) -> dict | None:
    """Tokenize a single SFT example (instruction + output).

    Returns dict with input_ids, attention_mask, num_actions (response length),
    or None if the example was fully truncated.
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    # Combine instruction and input
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction

    # Tokenize prompt and full sequence separately to find boundary
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    full_text = f"{prompt}\n\n{output}"
    full_tokens = tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=max_length)

    prompt_len = len(prompt_tokens["input_ids"])
    full_len = len(full_tokens["input_ids"])
    num_actions = full_len - prompt_len

    # Skip examples where response was fully truncated
    if num_actions <= 0:
        return None

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "num_actions": num_actions,
    }


def collate_sft_batch(examples: list, tokenizer) -> TrainingInputBatch:
    """Collate tokenized examples into a TrainingInputBatch.

    Creates the batch format expected by forward_backward with cross_entropy loss:
    - sequences: [batch_size, seq_len] - token IDs (left-padded)
    - attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
    - loss_mask: [batch_size, num_actions] - 1 for tokens to compute loss on
    """
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)

    sequences = []
    attention_masks = []
    loss_masks = []

    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        # Left-pad sequences (SkyRL convention)
        sequences.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + ex["attention_mask"])
        # Per-example loss_mask: 0s for padding, 1s only for this example's response tokens
        action_pad = max_num_actions - ex["num_actions"]
        loss_masks.append([0] * action_pad + [1] * ex["num_actions"])

    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.long),
        }
    )
    batch.metadata = {"response_length": max_num_actions}
    return batch


def main():
    """Run a minimal SFT training loop with Megatron backend."""
    cfg = get_sft_megatron_config()
    initialize_ray(cfg)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset...")
    # Use a small subset for demonstration
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:100]")

    logger.info("Tokenizing dataset...")
    tokenized = [tokenize_sft_example(ex, tokenizer) for ex in dataset]
    tokenized = [ex for ex in tokenized if ex is not None]  # Filter truncated
    logger.info(f"Kept {len(tokenized)} examples after filtering truncated")

    logger.info("Initializing Megatron policy worker (TP=2, PP=2, 4 GPUs)...")
    num_gpus = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
    pg = ResolvedPlacementGroup(raw_pg)

    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        ray_actor_type=PolicyWorker,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
    )
    ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path))

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)

    # Training loop
    batch_size = 4
    num_steps = 10
    logger.info(f"Starting Megatron SFT training for {num_steps} steps...")

    for step in tqdm(range(num_steps)):
        # Create batch from tokenized examples
        start_idx = (step * batch_size) % len(tokenized)
        batch_examples = tokenized[start_idx : start_idx + batch_size]
        if len(batch_examples) < batch_size:
            batch_examples = tokenized[:batch_size]  # Wrap around

        batch = collate_sft_batch(batch_examples, tokenizer)

        # Normalize loss mask so that the sum-reduction in cross_entropy_loss
        # produces a properly averaged loss.  Megatron sums over tokens within
        # each micro-batch and then averages across micro-batches, so we scale
        # by 1/(micro_batch_size * max_sequence_length) to get a per-token mean.
        micro_batch_size = cfg.trainer.micro_train_batch_size_per_gpu
        max_sequence_length = batch["sequences"].shape[-1]
        batch["loss_mask"] = batch["loss_mask"].float() / (micro_batch_size * max_sequence_length)

        # Forward-backward with cross-entropy loss (Tinker API style)
        metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")

        # Optimizer step
        grad_norm = dispatch.optim_step("policy")

        if step % 5 == 0:
            loss_val = metrics.get("final_loss", metrics.get("loss", "N/A"))
            logger.info(f"Step {step}: loss={loss_val:.4f}, grad_norm={grad_norm}")

    logger.info("Megatron SFT training complete!")
    ray.shutdown()


if __name__ == "__main__":
    main()
