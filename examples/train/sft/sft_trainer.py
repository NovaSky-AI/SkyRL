"""
SFT trainer for alphaXiv page-relevance classification.

Fine-tunes a model to predict whether a PDF page is relevant to a user query,
given the paper's title, abstract, and the page text.

Dataset: alphaXiv/page-labels (binary label: 1=relevant, 0=irrelevant)

Usage:
    # export WANDB_API_KEY=<your_key>
    # export HF_TOKEN=<your_token>
    bash examples/train/sft/run_sft.sh
"""

import json
import os
import random
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

SYSTEM_PROMPT = (
    "You will be given a query, the paper's title and abstract, and the text of one page of a PDF. "
    "Output 1 if the page is relevant to answering the query, or 0 if it is not."
)

EVAL_MD_PATH = Path(__file__).parent / "EVAL.md"


def get_sft_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()

    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.trainer.placement.policy_num_gpus_per_node = int(os.environ.get("NUM_GPUS", "1"))
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.trainer.logger = os.environ.get("LOGGER", "console")
    cfg.trainer.micro_train_batch_size_per_gpu = 2

    validate_cfg(cfg)
    return cfg


def build_chat_messages(example: dict) -> list[dict]:
    user_content = json.dumps(
        {
            "query": example["query"],
            "paperTitle": example["title"],
            "paperAbstract": example["abstract"],
            "pageNum": example["pageNumber"],
            "page": example["page"],
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def tokenize_sft_example(example: dict, tokenizer, max_length: int = 2048) -> dict | None:
    messages = build_chat_messages(example)
    completion = str(example["label"])

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + completion

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_length)
    full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)

    prompt_len = len(prompt_tokens["input_ids"])
    full_len = len(full_tokens["input_ids"])
    num_actions = full_len - prompt_len

    if num_actions <= 0:
        return None

    return {
        "input_ids": full_tokens["input_ids"],
        "attention_mask": full_tokens["attention_mask"],
        "num_actions": num_actions,
    }


def collate_sft_batch(examples: list, tokenizer) -> TrainingInputBatch:
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


def run_validation(dispatch, val_tokenized, tokenizer, batch_size, num_eval_samples=100):
    sample = random.sample(val_tokenized, min(num_eval_samples, len(val_tokenized)))

    total_loss = 0.0
    num_batches = 0

    for i in range(0, len(sample), batch_size):
        batch_examples = sample[i : i + batch_size]
        if not batch_examples:
            continue
        batch = collate_sft_batch(batch_examples, tokenizer)
        metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        loss = metrics.get("final_loss", metrics.get("loss", 0.0))
        total_loss += loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def write_eval_md(eval_results):
    with open(EVAL_MD_PATH, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("**Dataset:** alphaXiv/page-labels (validation split, 100 samples per eval)\n\n")
        f.write("| Step | Val Loss |\n")
        f.write("|------|----------|\n")
        for step, loss in eval_results:
            f.write(f"| {step} | {loss:.4f} |\n")


def main():
    cfg = get_sft_config()
    initialize_ray(cfg)

    logger_backend = os.environ.get("LOGGER", "console")
    project_name = os.environ.get("WANDB_PROJECT", "alphaxiv-page-labels")
    run_name = os.environ.get("WANDB_RUN_NAME", "sft-qwen2.5-0.5b")

    tracker = Tracking(
        project_name=project_name,
        experiment_name=run_name,
        backends=logger_backend,
        config=cfg,
    )

    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    num_steps = int(os.environ.get("NUM_STEPS", "500"))
    eval_interval = int(os.environ.get("EVAL_INTERVAL", "50"))

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset...")
    train_dataset = load_dataset("alphaXiv/page-labels", split="train")
    val_dataset = load_dataset("alphaXiv/page-labels", split="validation")

    logger.info("Tokenizing train dataset...")
    tokenized_train = [tokenize_sft_example(ex, tokenizer, max_length) for ex in train_dataset]
    tokenized_train = [ex for ex in tokenized_train if ex is not None]
    logger.info(f"Train: kept {len(tokenized_train)}/{len(train_dataset)} examples")

    logger.info("Tokenizing validation dataset...")
    tokenized_val = [tokenize_sft_example(ex, tokenizer, max_length) for ex in val_dataset]
    tokenized_val = [ex for ex in tokenized_val if ex is not None]
    logger.info(f"Val: kept {len(tokenized_val)}/{len(val_dataset)} examples")

    logger.info("Initializing policy worker...")
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

    eval_results = []
    logger.info(f"Starting SFT training for {num_steps} steps (eval every {eval_interval})...")

    for step in tqdm(range(num_steps)):
        start_idx = (step * batch_size) % len(tokenized_train)
        batch_examples = tokenized_train[start_idx : start_idx + batch_size]
        if len(batch_examples) < batch_size:
            batch_examples = tokenized_train[:batch_size]

        batch = collate_sft_batch(batch_examples, tokenizer)
        metrics = dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        grad_norm = dispatch.optim_step("policy")

        train_loss = metrics.get("final_loss", metrics.get("loss", 0.0))
        tracker.log({"train/loss": train_loss, "train/grad_norm": grad_norm}, step=step)

        if step % eval_interval == 0 or step == num_steps - 1:
            logger.info(f"Running validation at step {step}...")
            val_loss = run_validation(dispatch, tokenized_val, tokenizer, batch_size)
            tracker.log({"val/loss": val_loss}, step=step)
            eval_results.append((step, val_loss))
            write_eval_md(eval_results)
            logger.info(f"Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    write_eval_md(eval_results)
    logger.info(f"SFT training complete! Results written to {EVAL_MD_PATH}")
    tracker.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
