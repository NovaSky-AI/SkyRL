"""
SFT (Supervised Fine-Tuning) trainer for SkyRL.

Supports both FSDP and Megatron backends via a single ``SFTTrainer`` class.
The backend is selected dynamically based on ``SFTConfig.strategy``.

Usage::

    from skyrl.train.sft_config import SFTConfig, SFTPlacementConfig
    from skyrl.train.sft_trainer import SFTTrainer

    cfg = SFTConfig(strategy="megatron")
    trainer = SFTTrainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.shutdown()

Or as a CLI entrypoint::

    python -m skyrl.train.sft_trainer strategy=megatron model.path=Qwen/Qwen3-0.6B
"""

import os
import random
import sys

import ray
import torch
from datasets import load_dataset
from loguru import logger
from ray.util.placement_group import placement_group
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.train.sft_config import SFTConfig, build_skyrl_config_for_sft
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.tracking import Tracking
from skyrl.train.utils.trainer_utils import GLOBAL_STEP_PREFIX, extract_step_from_path
from skyrl.train.utils.utils import ResolvedPlacementGroup, Timer, initialize_ray

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


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


def tokenize_chat_example(
    example: dict,
    tokenizer,
    max_length: int = 512,
    messages_key: str = "messages",
) -> dict | None:
    """Tokenize a chat-format example. Loss on last assistant message only.

    Uses apply_chat_template to tokenize prompt (all messages except the last)
    and full conversation, then computes num_actions from the difference.

    Returns dict with input_ids, attention_mask, num_actions -- same format as
    tokenize_sft_example(), so collate_sft_batch() works unchanged.
    """
    messages = example[messages_key]

    # Validate: last message must be from assistant
    if not messages or messages[-1]["role"] != "assistant":
        return None

    # Tokenize prompt (everything except last assistant message)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=True,
        tokenize=True,
        truncation=True,
        max_length=max_length,
        return_dict=False,
    )

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        truncation=True,
        max_length=max_length,
        return_dict=False,
    )

    num_actions = len(full_ids) - len(prompt_ids)
    if num_actions <= 0:
        return None

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "num_actions": num_actions,
    }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SFTTrainer
# ---------------------------------------------------------------------------


class SFTTrainer:
    """SFT trainer supporting FSDP and Megatron backends.

    Unlike RayPPOTrainer, this does NOT subclass it. SFT's concerns are
    fundamentally different: no generation, no critic, no advantages, no
    KL penalty. Sharing a base class would create confusing dead code paths.

    Usage::

        trainer = SFTTrainer(SFTConfig(strategy="megatron"))
        trainer.setup()
        trainer.train()
        trainer.shutdown()
    """

    def __init__(self, cfg: SFTConfig):
        self.sft_cfg = cfg
        self.cfg = build_skyrl_config_for_sft(cfg)
        self.tokenizer = None
        self.dispatch: WorkerDispatch | None = None
        self.tracker: Tracking | None = None

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup(self):
        """Initialize Ray, tokenizer, workers, dispatch, and tracker."""
        self._init_ray()
        self._init_tokenizer()
        self._init_workers()
        self._init_tracker()

    def _init_ray(self):
        initialize_ray(self.cfg)

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_cfg.model.path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_workers(self):
        """Create PPORayActorGroup and WorkerDispatch.

        Selects the correct PolicyWorker based on strategy.
        """
        if self.sft_cfg.strategy == "megatron":
            from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
                PolicyWorker,
            )
        else:
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker

        num_gpus = self.sft_cfg.placement.num_gpus_per_node
        raw_pg = placement_group(
            [{"GPU": num_gpus, "CPU": num_gpus}] * self.sft_cfg.placement.num_nodes,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
        pg = ResolvedPlacementGroup(raw_pg)

        actor_group = PPORayActorGroup(
            self.cfg.trainer,
            num_nodes=self.sft_cfg.placement.num_nodes,
            num_gpus_per_node=num_gpus,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=self.cfg.trainer.policy.sequence_parallel_size,
        )
        ray.get(
            actor_group.async_init_model(
                self.sft_cfg.model.path,
                num_training_steps=self.sft_cfg.num_steps,
            )
        )
        ray.get(actor_group.async_run_ray_method("pass_through", "_set_pad_token_id", self.tokenizer.pad_token_id))

        self.dispatch = WorkerDispatch(self.cfg, policy_actor_group=actor_group)

    def _init_tracker(self):
        self.tracker = Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #

    def _load_and_tokenize(self, dataset_name: str, dataset_split: str) -> list:
        """Load and tokenize a dataset.

        Auto-detects the dataset format based on column names:
        - If a ``messages_key`` column exists, uses chat-format tokenization.
        - If ``instruction`` and ``output`` columns exist, uses Alpaca-format
          tokenization.

        Args:
            dataset_name: HuggingFace dataset name (e.g. ``"yahma/alpaca-cleaned"``).
            dataset_split: Dataset split (e.g. ``"train[:100]"`` or ``"test"``).

        Returns a list of tokenized examples (dicts with ``input_ids``,
        ``attention_mask``, ``num_actions``).
        """
        logger.info(f"Loading dataset '{dataset_name}' split='{dataset_split}'...")
        dataset = load_dataset(dataset_name, split=dataset_split)

        columns = dataset.column_names
        logger.info("Tokenizing dataset...")

        if self.sft_cfg.messages_key in columns:
            # Chat format
            tokenized = [
                tokenize_chat_example(ex, self.tokenizer, self.sft_cfg.max_length, self.sft_cfg.messages_key)
                for ex in dataset
            ]
        elif "instruction" in columns and "output" in columns:
            # Alpaca format
            tokenized = [tokenize_sft_example(ex, self.tokenizer, self.sft_cfg.max_length) for ex in dataset]
        else:
            raise ValueError(
                f"Unrecognized dataset format. Expected '{self.sft_cfg.messages_key}' column "
                f"(chat format) or 'instruction'+'output' columns (Alpaca format). "
                f"Found columns: {columns}"
            )

        tokenized = [ex for ex in tokenized if ex is not None]
        logger.info(f"Tokenized {len(tokenized)} examples (filtered from {len(dataset)})")
        return tokenized

    def load_dataset(self) -> list:
        """Load and tokenize the training dataset."""
        return self._load_and_tokenize(self.sft_cfg.dataset_name, self.sft_cfg.dataset_split)

    def collate_batch(self, examples: list) -> TrainingInputBatch:
        """Collate examples into a TrainingInputBatch with loss normalization.

        Normalizes the loss_mask so that the sum-reduction in cross_entropy_loss
        produces a properly averaged loss. Megatron sums over tokens within each
        micro-batch and averages across micro-batches, so we scale by
        ``1 / (micro_batch_size * max_sequence_length)`` to get a per-token mean.
        """
        batch = collate_sft_batch(examples, self.tokenizer)
        # Loss normalization
        micro_batch_size = self.sft_cfg.micro_train_batch_size_per_gpu
        max_sequence_length = batch["sequences"].shape[-1]
        batch["loss_mask"] = batch["loss_mask"].float() / (micro_batch_size * max_sequence_length)
        return batch

    # ------------------------------------------------------------------ #
    # Checkpoint resume
    # ------------------------------------------------------------------ #

    def load_checkpoint(self) -> int:
        """Load a checkpoint and return the step number to resume from.

        Behaviour depends on ``sft_cfg.resume_from``:
        - ``""`` (empty): no resume, return 0.
        - ``"latest"``: read ``latest_ckpt_global_step.txt`` from ``ckpt_path``.
        - otherwise: treat as a direct path to a ``global_step_N`` directory.

        Returns:
            The global step to resume from (0 if no checkpoint loaded).
        """
        resume_from = self.sft_cfg.resume_from
        if not resume_from:
            return 0

        if resume_from == "latest":
            if not self.sft_cfg.ckpt_path:
                logger.info("resume_from='latest' but ckpt_path is empty, starting from scratch")
                return 0
            latest_file = os.path.join(self.sft_cfg.ckpt_path, "latest_ckpt_global_step.txt")
            if not io.exists(latest_file):
                logger.info("No latest checkpoint marker found, starting from scratch")
                return 0
            with io.open_file(latest_file, "r") as f:
                ckpt_step = int(f.read().strip())
            checkpoint_path = os.path.join(self.sft_cfg.ckpt_path, f"{GLOBAL_STEP_PREFIX}{ckpt_step}")
        else:
            checkpoint_path = resume_from

        if not io.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        global_step = extract_step_from_path(checkpoint_path)
        if global_step == -1:
            raise ValueError(
                f"Cannot extract step number from checkpoint path: {checkpoint_path}. "
                f"Expected a directory named '{GLOBAL_STEP_PREFIX}<N>'."
            )

        policy_ckpt_dir = os.path.join(checkpoint_path, "policy")
        logger.info(f"Loading checkpoint from {checkpoint_path} (step {global_step})")
        self.dispatch.load_checkpoint(
            "policy",
            policy_ckpt_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        logger.info(f"Successfully resumed from global_step_{global_step}")
        return global_step

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TrainingInputBatch, step: int) -> dict:
        """Execute a single training step: forward_backward + optim_step.

        Args:
            batch: The collated training batch.
            step: Current global step (reserved for future use, e.g. scheduling).

        Returns:
            Dict with ``loss``, ``grad_norm``, and ``timings``.
        """
        timings: dict[str, float] = {}
        with Timer("forward_backward", timings):
            metrics = self.dispatch.forward_backward("policy", batch, loss_fn="cross_entropy")
        with Timer("optim_step", timings):
            grad_norm = self.dispatch.optim_step("policy")

        loss_val = metrics.get("final_loss", metrics.get("loss", float("nan")))
        return {
            "loss": loss_val,
            "grad_norm": grad_norm,
            "timings": timings,
        }

    def train(self):
        """Full training loop: load data, iterate, log, checkpoint."""
        tokenized = self.load_dataset()

        batch_size = self.sft_cfg.batch_size
        num_steps = self.sft_cfg.num_steps

        # Early validation: dataset must have at least batch_size examples
        if len(tokenized) < batch_size:
            raise ValueError(
                f"Dataset has {len(tokenized)} examples after tokenization, but batch_size={batch_size}. "
                f"Reduce batch_size or use more data."
            )

        # Validate batch_size is divisible by data-parallel size
        total_gpus = self.sft_cfg.placement.num_nodes * self.sft_cfg.placement.num_gpus_per_node
        if self.sft_cfg.strategy == "megatron":
            tp = self.sft_cfg.megatron.tensor_model_parallel_size
            pp = self.sft_cfg.megatron.pipeline_model_parallel_size
            dp_size = total_gpus // (tp * pp)
        else:
            # FSDP: all GPUs are data-parallel
            dp_size = total_gpus
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by data-parallel size ({dp_size})")
        per_dp_batch = batch_size // dp_size
        micro_batch = self.sft_cfg.micro_train_batch_size_per_gpu
        if per_dp_batch % micro_batch != 0:
            raise ValueError(
                f"batch_size / dp_size ({per_dp_batch}) must be divisible by "
                f"micro_train_batch_size_per_gpu ({micro_batch})"
            )

        # Resume from checkpoint if configured
        start_step = self.load_checkpoint()

        # Shuffle data before training
        rng = random.Random(self.sft_cfg.seed)
        rng.shuffle(tokenized)

        # Replay epoch shuffles for reproducibility on resume
        start_epoch = (start_step * batch_size) // len(tokenized)
        for _ in range(start_epoch):
            rng.shuffle(tokenized)
        current_epoch = start_epoch

        logger.info(f"Starting SFT training for {num_steps} steps (batch_size={batch_size})...")
        if start_step > 0:
            logger.info(f"Resuming from step {start_step}")

        for step in tqdm(range(start_step, num_steps)):
            all_timings: dict[str, float] = {}

            with Timer("step", all_timings):
                # Check for epoch boundary and reshuffle
                epoch = (step * batch_size) // len(tokenized)
                if epoch > current_epoch:
                    for _ in range(epoch - current_epoch):
                        rng.shuffle(tokenized)
                    current_epoch = epoch

                # Data loading with wrap-around
                with Timer("data_loading", all_timings):
                    start_idx = (step * batch_size) % len(tokenized)
                    end_idx = start_idx + batch_size
                    if end_idx > len(tokenized):
                        batch_examples = tokenized[start_idx:] + tokenized[: end_idx - len(tokenized)]
                    else:
                        batch_examples = tokenized[start_idx:end_idx]
                    batch = self.collate_batch(batch_examples)

                # Training step
                step_result = self.train_step(batch, step)
                all_timings.update(step_result["timings"])

            # Compute throughput using actual (non-padding) tokens
            batch_num_seq = batch["sequences"].shape[0]
            batch_padded_seq_len = batch["sequences"].shape[1]
            actual_num_tokens = batch["attention_mask"].sum().item()
            tokens_per_second = actual_num_tokens / all_timings["step"]

            # Build log dict
            log_dict = {
                "train/loss": step_result["loss"],
                "train/grad_norm": step_result["grad_norm"],
                "train/tokens_per_second": tokens_per_second,
                "train/actual_num_tokens": actual_num_tokens,
                "train/batch_num_seq": batch_num_seq,
                "train/batch_padded_seq_len": batch_padded_seq_len,
            }
            log_dict.update({f"timing/{k}": v for k, v in all_timings.items()})

            # Checkpoint at regular intervals
            if (
                self.sft_cfg.ckpt_path
                and self.sft_cfg.ckpt_interval > 0
                and step > 0
                and step % self.sft_cfg.ckpt_interval == 0
            ):
                with Timer("save_checkpoint", all_timings):
                    self.save_checkpoint(step)
                log_dict["timing/save_checkpoint"] = all_timings["save_checkpoint"]

            self.tracker.log(log_dict, step=step, commit=True)

            if step % 5 == 0:
                logger.info(f"Step {step}: loss={step_result['loss']:.4f}, " f"grad_norm={step_result['grad_norm']}")

        # Save final checkpoint (if checkpointing is enabled)
        if self.sft_cfg.ckpt_path:
            final_step = num_steps - 1
            already_saved = (
                self.sft_cfg.ckpt_interval > 0 and final_step > 0 and final_step % self.sft_cfg.ckpt_interval == 0
            )
            if not already_saved:
                logger.info(f"Saving final checkpoint at step {final_step}")
                self.save_checkpoint(final_step)

        logger.info("SFT training complete!")

    def save_checkpoint(self, step: int):
        """Save a checkpoint at the given step."""
        global_step_folder = os.path.join(self.sft_cfg.ckpt_path, f"{GLOBAL_STEP_PREFIX}{step}")
        policy_save_dir = os.path.join(global_step_folder, "policy")
        io.makedirs(global_step_folder, exist_ok=True)
        logger.info(f"Saving checkpoint at step {step} to {global_step_folder}")
        self.dispatch.save_checkpoint("policy", policy_save_dir, self.tokenizer)
        # Write latest checkpoint marker
        latest_file = os.path.join(self.sft_cfg.ckpt_path, "latest_ckpt_global_step.txt")
        with io.open_file(latest_file, "w") as f:
            f.write(str(step))
        logger.info(f"Checkpoint saved for global_step_{step}")

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def shutdown(self):
        """Finish tracking and shutdown Ray."""
        if self.tracker is not None:
            self.tracker.finish()
        ray.shutdown()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main():
    """CLI entrypoint for SFT training.

    Parses dotlist CLI arguments to build an ``SFTConfig``, then runs the
    full training pipeline (setup, train, shutdown).

    Usage::

        python -m skyrl.train.sft_trainer strategy=megatron model.path=Qwen/Qwen3-0.6B
    """
    cfg = SFTConfig.from_cli_overrides(sys.argv[1:])
    trainer = SFTTrainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.shutdown()


if __name__ == "__main__":
    main()
