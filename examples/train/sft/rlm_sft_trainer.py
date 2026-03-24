"""
SFT trainer for alphaXiv RLM agent trajectories.

Fine-tunes Qwen3.5-9B on the alphaXiv/rlm-stf-v1 dataset, which contains
pre-formatted multi-turn prompts (input_prompt) and agent responses (output_response).
Loss is computed only over output_response tokens.

The input_prompt already includes empty <think></think> blocks, so thinking
is effectively disabled during the forward pass.

Dataset schema:
  - id: str
  - input_prompt: str  (pre-formatted with chat template, ends with assistant turn)
  - output_response: str  (agent response, ends with <|im_end|>)

Usage:
    # export WANDB_API_KEY=<your_key>
    # export HF_TOKEN=<your_token>
    bash examples/train/sft/run_rlm_sft.sh
"""

import os
import shutil
from pathlib import Path

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
from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl.train.utils.utils import initialize_ray, validate_cfg, ResolvedPlacementGroup
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.tracking import Tracking

MODEL_PATH = "Qwen/Qwen3.5-9B"
DATASET_NAME = "alphaXiv/rlm-stf-v1"
WORKSPACE_ROOT = Path(__file__).parents[3]
EVAL_MD_PATH = WORKSPACE_ROOT / "EVAL.md"


def get_sft_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()

    num_gpus = int(os.environ.get("NUM_GPUS", "4"))

    cfg.trainer.policy.model.path = MODEL_PATH
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.policy.sequence_parallel_size = num_gpus
    cfg.trainer.logger = os.environ.get("LOGGER", "console")
    cfg.trainer.micro_train_batch_size_per_gpu = int(os.environ.get("MICRO_BATCH_SIZE", "2"))

    validate_cfg(cfg)
    return cfg


def tokenize_sft_example(example: dict, tokenizer, max_length: int = 4096) -> dict:
    """Tokenize an RLM SFT example.

    The input_prompt is already chat-template-formatted (with empty <think></think>
    blocks for thinking-off mode). We concatenate input_prompt + output_response,
    tokenize as a single sequence, and compute num_actions as the number of tokens
    belonging to output_response so the loss mask covers only the response.
    """
    input_prompt = example["input_prompt"]
    output_response = example["output_response"]

    full_text = input_prompt + output_response
    full_tokens = tokenizer(full_text, add_special_tokens=False)

    if len(full_tokens["input_ids"]) > max_length:
        return {"input_ids": [], "attention_mask": [], "num_actions": 0, "_valid": False}

    prompt_tokens = tokenizer(input_prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokens["input_ids"])
    full_len = len(full_tokens["input_ids"])
    num_actions = full_len - prompt_len

    if num_actions <= 0:
        return {"input_ids": [], "attention_mask": [], "num_actions": 0, "_valid": False}

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "num_actions": num_actions,
        "_valid": True,
    }


def collate_sft_batch(examples: list, tokenizer) -> TrainingInputBatch:
    """Collate tokenized examples into a TrainingInputBatch.

    Left-pads sequences (SkyRL convention). The loss_mask marks only the
    output_response tokens for gradient computation.
    """
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)

    sequences = []
    attention_masks = []
    loss_masks = []

    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        sequences.append([tokenizer.pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + ex["attention_mask"])
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
    cfg = get_sft_config()
    initialize_ray(cfg)

    logger_backend = os.environ.get("LOGGER", "console")
    project_name = os.environ.get("WANDB_PROJECT", "alphaxiv-rlm-sft")
    run_name = os.environ.get("WANDB_RUN_NAME", "sft-qwen3.5-9b")

    tracker = Tracking(
        project_name=project_name,
        experiment_name=run_name,
        backends=logger_backend,
        config=cfg,
    )

    max_length = int(os.environ.get("MAX_LENGTH", "4096"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "1"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "1e-5"))
    log_interval = int(os.environ.get("LOG_INTERVAL", "10"))
    sample_interval = int(os.environ.get("SAMPLE_INTERVAL", "50"))
    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", "10"))
    max_checkpoints = int(os.environ.get("MAX_CHECKPOINTS", "2"))
    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", str(WORKSPACE_ROOT / "exports" / "checkpoints")))

    logger.info(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Dataset size: {len(dataset)} examples")

    num_workers = min(os.cpu_count() or 1, 8)
    logger.info(f"Tokenizing with {num_workers} workers, max_length={max_length}...")
    mapped = dataset.map(
        lambda ex: tokenize_sft_example(ex, tokenizer, max_length),
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    mapped = mapped.filter(lambda ex: ex["_valid"], num_proc=num_workers)
    tokenized = [
        {"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"], "num_actions": ex["num_actions"]}
        for ex in mapped
    ]
    logger.info(f"Kept {len(tokenized)}/{len(dataset)} examples after filtering")

    logger.info("Initializing policy worker...")
    num_gpus = cfg.trainer.placement.policy_num_gpus_per_node
    raw_pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=60)
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
    ray.get(actor_group.async_init_model(MODEL_PATH, freeze_vision_encoder=True))

    dispatch = WorkerDispatch(cfg, policy_actor_group=actor_group)
    dispatch.set_lr("policy", learning_rate)

    steps_per_epoch = (len(tokenized) + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch
    logger.info(
        f"Starting SFT: {num_epochs} epochs, {steps_per_epoch} steps/epoch, "
        f"{total_steps} total steps, batch_size={batch_size}, lr={learning_rate}"
    )

    with open(EVAL_MD_PATH, "w") as f:
        f.write("# SFT Training Samples\n\n")
        f.write(f"**Model:** `{MODEL_PATH}` | **Dataset:** `{DATASET_NAME}`\n\n")

    global_step = 0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        for i in tqdm(range(0, len(tokenized), batch_size), desc=f"Epoch {epoch + 1}"):
            batch_examples = tokenized[i : i + batch_size]
            if len(batch_examples) < 2:
                continue

            batch = collate_sft_batch(batch_examples, tokenizer)
            metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
            grad_norm = dispatch.optim_step("policy")

            train_loss = metrics.get("final_loss", metrics.get("loss", 0.0))
            tracker.log(
                {"train/loss": train_loss, "train/grad_norm": grad_norm, "train/epoch": epoch},
                step=global_step,
            )

            if global_step % log_interval == 0:
                logger.info(f"Step {global_step}: loss={train_loss:.4f}, grad_norm={grad_norm}")

            if global_step % sample_interval == 0:
                ex = batch_examples[0]
                response_ids = ex["input_ids"][-ex["num_actions"]:]
                response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                with open(EVAL_MD_PATH, "a") as f:
                    f.write(f"---\n\n### Step {global_step} | loss={train_loss:.4f}\n\n")
                    f.write(f"```\n{response_text}\n```\n\n")

            if global_step > 0 and global_step % checkpoint_interval == 0:
                ckpt_path = str(checkpoint_dir / f"step_{global_step}")
                logger.info(f"Saving checkpoint at step {global_step} to {ckpt_path}")
                dispatch.save_checkpoint("policy", ckpt_path, tokenizer=tokenizer)

                if max_checkpoints > 0:
                    existing = sorted(
                        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
                        key=lambda p: int(p.name.split("_")[1]),
                    )
                    while len(existing) > max_checkpoints:
                        old = existing.pop(0)
                        logger.info(f"Removing old checkpoint: {old}")
                        shutil.rmtree(old)

            global_step += 1

    final_ckpt_path = str(checkpoint_dir / "final")
    logger.info(f"Saving final checkpoint to {final_ckpt_path}")
    dispatch.save_checkpoint("policy", final_ckpt_path, tokenizer=tokenizer)

    logger.info(f"SFT training complete! Samples written to {EVAL_MD_PATH}")
    tracker.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
