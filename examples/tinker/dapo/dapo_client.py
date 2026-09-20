"""DAPO training client for SkyRL's Tinker API server.

Usage:
    # Terminal 1 (GPU node)
    bash examples/tinker/dapo/run_tinker_server.sh

    # Terminal 2
    TINKER_API_KEY=tml-dummy uv run --extra tinker --with datasets --with torch \
        python examples/tinker/dapo/dapo_client.py --lora-rank 128   # or --lora-rank 0 for full FT

Reproduces examples/train/algorithms/dapo/run_dapo_qwen3_30b_a3b_{lora_,}megatron_aime.sh through the
Tinker codepath. The client owns the algorithm: GRPO group-normalized advantages, DAPO soft overlong
punishment and overlong filtering, token-mean-legacy loss scaling, clip-higher epsilons and LR warmup.
The server (see run_tinker_server.sh) owns execution: Megatron parallelism, dual-clip loss type,
weight decay and off-policy correction.

Off-policy correction: unlike the native scripts, this recipe does NOT use TIS. It uses geometric
sequence masking (see docs/content/docs/algorithms/off_policy_correction.mdx). For the mask to see the
real train/inference mismatch the client runs a forward pass to obtain the training policy's logprobs
at sampling time (as the native trainer does) and sends the vLLM sampling logprobs separately as
`rollout_logprobs` (a SkyRL extension of the Tinker datum).

Smoke testing: the DAPO_* environment variables below override batch sizes and lengths so the loop
can be exercised on a small model / few GPUs before the full recipe.
"""

from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
import tinker
from tinker import types
from typing import Any, Iterable, Sequence
import random
import torch
import datasets
from dataclasses import dataclass
from collections import defaultdict
import json
import time

from skyrl_gym.envs.aime import utils as aime_utils

from skyrl.backends.skyrl_train.utils.ppo_utils import (
    apply_loss_reduction_to_advantages_minibatch,
)


logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read an integer override from `DAPO_<name>`; used to shrink the recipe for smoke tests."""
    return int(os.environ.get(f"DAPO_{name}", default))


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Base"
DEFAULT_DATA_DIR = os.path.expanduser("~/data/dapo")
# Written by examples/train/algorithms/dapo/prepare_dapo_data.sh (the "-cleaned" outputs of
# data_preprocess_dapo_aime.py, which drops the duplicate rows in DAPO-Math-17k).
TRAIN_FILE_NAME = "dapo-math-17k-cleaned.parquet"
VAL_FILE_NAME = "aime-2024-cleaned.parquet"
DEFAULT_CKPT_DIR = os.path.expanduser("~/ckpts/dapo_qwen3_30b_a3b_tinker")
DEFAULT_WANDB_PROJECT = "dapo_aime"
DEFAULT_WANDB_RUN_NAME = "dapo_qwen3_30b_a3b_tinker"
# LoRA rank: 128 for the LoRA recipe, 0 for full fine-tuning.
DEFAULT_LORA_RANK = 128

# Hyperparameters below mirror examples/train/algorithms/dapo/run_dapo_qwen3_30b_a3b_lora_megatron_aime.sh
TRAIN_EPOCHS = _env_int("TRAIN_EPOCHS", 20)
TRAIN_BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 512)
EVAL_BATCH_SIZE = _env_int("EVAL_BATCH_SIZE", 1024)
POLICY_MINI_BATCH_SIZE = _env_int("POLICY_MINI_BATCH_SIZE", 32)  # in prompts
UPDATE_EPOCHS_PER_BATCH = 1
N_SAMPLES_PER_PROMPT = _env_int("N_SAMPLES_PER_PROMPT", 16)
EVAL_N_SAMPLES_PER_PROMPT = _env_int("EVAL_N_SAMPLES_PER_PROMPT", 32)
MAX_PROMPT_LENGTH = _env_int("MAX_PROMPT_LENGTH", 2048)
MAX_GENERATE_LENGTH = _env_int("MAX_GENERATE_LENGTH", 8192)
EVAL_BEFORE_TRAIN = bool(_env_int("EVAL_BEFORE_TRAIN", 1))
CKPT_INTERVAL = _env_int("CKPT_INTERVAL", 10)
EVAL_INTERVAL = _env_int("EVAL_INTERVAL", 5)

# Loss: dual-clip PPO. The clip epsilons are sent per request; `policy_loss_type=dual_clip`
# and `clip_ratio_c=10.0` must be set server-side in backend_config (see run_tinker_server.sh).
POLICY_LOSS = "ppo"
CLIP_RATIO_LOW = 0.2  # eps_clip_low: ratio floor is 1 - 0.2 = 0.8
CLIP_RATIO_HIGH = 0.28  # eps_clip_high: ratio ceiling is 1 + 0.28 = 1.28
LOSS_REDUCTION = "token_mean_legacy"
# Keep aligned with trainer.micro_train_batch_size_per_gpu in run_tinker_server.sh
# (4 for the LoRA recipe, 2 for full fine-tuning).
MICRO_TRAIN_BATCH_SIZE = _env_int("MICRO_TRAIN_BATCH_SIZE", 4)
# Sequences per `forward` request when recomputing old logprobs; the server micro-batches internally.
FORWARD_BATCH_SIZE = POLICY_MINI_BATCH_SIZE * N_SAMPLES_PER_PROMPT

# Optimizer (trainer.policy.optimizer_config in the reference script)
POLICY_LEARNING_RATE = 1.0e-5
NUM_WARMUP_STEPS = _env_int("NUM_WARMUP_STEPS", 160)  # counted in optimizer (mini-batch) steps
# Applied server-side: set trainer.policy.optimizer_config.weight_decay=0.1 in backend_config.
# max_grad_norm=1.0 is the SkyRL optimizer default and is also applied server-side.
WEIGHT_DECAY = 0.1

# Soft overlong punishment / overlong filtering (DAPO)
OVERLONG_BUFFER_LEN = _env_int("OVERLONG_BUFFER_LEN", 1024 * 4)
OVERLONG_BUFFER_PENALTY_FACTOR = 1.0
APPLY_OVERLONG_FILTERING = True

# Off-policy correction. When True, after sampling the client runs a `forward` pass to get the
# training policy's logprobs (PPO ratio denominator, as in the native trainer) and sends the vLLM
# sampling logprobs as `rollout_logprobs` so the server's geometric sequence mask
# (trainer.algorithm.off_policy_correction.sequence_mask_metric="geometric" in backend_config)
# measures the true train/inference mismatch. When False, the sampling logprobs are used as the
# ratio denominator and any server-side off-policy correction is a no-op.
RECOMPUTE_OLD_LOGPROBS = bool(_env_int("RECOMPUTE_OLD_LOGPROBS", 1))

# Sampling
SAMPLING_STOP_STRINGS: list[str] | None = None
SAMPLING_TOP_K = -1
TRAIN_SAMPLING_TEMPERATURE = 1.0
TRAIN_SAMPLING_TOP_P = 1.0
EVAL_SAMPLING_TEMPERATURE = 1.0
EVAL_SAMPLING_TOP_P = 0.7


class WandbLogger:
    def __init__(self, output_dir: str | None, run_config: dict[str, Any] | None = None):
        self._run = None
        self.enabled = False
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logger.warning("WANDB_API_KEY is not set; skipping wandb logging")
            return

        try:
            import wandb
        except ImportError:
            logger.warning("WANDB_API_KEY is set, but wandb is not installed; skipping wandb logging")
            return

        run_kwargs: dict[str, Any] = {
            "project": os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            "config": {
                **(run_config or {}),
                "train_batch_size": TRAIN_BATCH_SIZE,
                "policy_mini_batch_size": POLICY_MINI_BATCH_SIZE,
                "update_epochs_per_batch": UPDATE_EPOCHS_PER_BATCH,
                "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
                "policy_learning_rate": POLICY_LEARNING_RATE,
                "num_warmup_steps": NUM_WARMUP_STEPS,
                "weight_decay": WEIGHT_DECAY,
                "recompute_old_logprobs": RECOMPUTE_OLD_LOGPROBS,
                "clip_ratio_low": CLIP_RATIO_LOW,
                "clip_ratio_high": CLIP_RATIO_HIGH,
                "loss_reduction": LOSS_REDUCTION,
                "max_generate_length": MAX_GENERATE_LENGTH,
            },
        }
        if output_dir:
            run_kwargs["dir"] = expand_path(output_dir)
        if entity := os.environ.get("WANDB_ENTITY"):
            run_kwargs["entity"] = entity
        run_kwargs["name"] = os.environ.get("WANDB_RUN_NAME", DEFAULT_WANDB_RUN_NAME)
        if tags := os.environ.get("WANDB_TAGS"):
            run_kwargs["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]

        logger.info(
            "Initializing wandb: project=%s entity=%s run_name=%s output_dir=%s",
            run_kwargs["project"],
            run_kwargs.get("entity"),
            run_kwargs.get("name"),
            run_kwargs.get("dir"),
        )
        self._wandb = wandb
        self._run = wandb.init(**run_kwargs)
        self.enabled = self._run is not None
        if self.enabled:
            logger.info("wandb initialized successfully")

    def log(self, payload: dict[str, Any]) -> None:
        if self._run is None:
            return

        numeric_payload = {
            key: value
            for key, value in payload.items()
            if isinstance(value, (int, float, bool)) and not isinstance(value, str)
        }
        if numeric_payload:
            self._wandb.log(numeric_payload, step=payload.get("step"))

        for key, value in payload.items():
            if key in numeric_payload:
                continue
            if isinstance(value, str):
                self._run.summary[key] = value

    def finish(self) -> None:
        if self._run is not None:
            self._wandb.finish()


def expand_path(path: str | Path) -> str:
    return os.path.expanduser(str(path))


def prompt_tokens_for_messages(tokenizer, messages: list[dict], max_prompt_length: int) -> list[int] | None:
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False, tokenize=True)
    if len(tokens) > max_prompt_length:
        return None
    return list(tokens)


@dataclass
class ExampleRecord:
    prompt_messages: list[dict]
    prompt_tokens: list[int]
    ground_truth: str
    question: str
    dataset_index: int


@dataclass
class Trajectory:
    prompt_tokens: list[int]
    response_tokens: list[int]
    # Training-policy logprobs at sampling time: the PPO ratio denominator. Initialized to the
    # sampling logprobs and replaced by `compute_old_logprobs` when RECOMPUTE_OLD_LOGPROBS is set.
    old_logprobs: list[float]
    # vLLM sampling logprobs, sent as `rollout_logprobs` for off-policy correction.
    rollout_logprobs: list[float]
    # Per-response-token loss weights. All ones, or all zeros when the response was truncated
    # and overlong filtering is on (DAPO's "Overlong Filtering").
    loss_mask: list[float]
    stop_reason: str
    advantages: list[float]
    reward: float
    question: str
    ground_truth: str
    response_text: str
    prompt_group: int


def load_split(path: str, tokenizer, max_prompt_length: int) -> list[ExampleRecord]:
    dataset = datasets.load_dataset("parquet", data_files=expand_path(path), keep_in_memory=True)["train"]
    records: list[ExampleRecord] = []
    filtered = 0
    for idx, row in enumerate(dataset):
        prompt_tokens = prompt_tokens_for_messages(tokenizer, row["prompt"], max_prompt_length=max_prompt_length)
        if prompt_tokens is None:
            filtered += 1
            continue
        records.append(
            ExampleRecord(
                prompt_messages=row["prompt"],
                prompt_tokens=prompt_tokens,
                # Key names follow the DAPO-Math-17k / AIME-2024 parquet schema, i.e. what
                # skyrl_gym.envs.aime.AIMEEnv reads from env_extras in the native recipe.
                ground_truth=str(row["reward_model"]["ground_truth"]).strip(),
                question=(row.get("extra_info") or {}).get("raw_problem", ""),
                dataset_index=idx,
            )
        )
    logger.info("Loaded %s records from %s (filtered %s long prompts)", len(records), path, filtered)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Base model; must match the server's --base-model")
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank (128 for the LoRA recipe, 0 for full fine-tuning)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    return parser.parse_args()


def build_split_paths(data_dir: str) -> tuple[str, str]:
    """Locate the splits written by examples/train/algorithms/dapo/prepare_dapo_data.sh.

    The filenames match TRAIN_FILE / TEST_FILE in the reference scripts
    (run_dapo_qwen3_30b_a3b_{lora_,}megatron_aime.sh), i.e. the de-duplicated outputs
    of data_preprocess_dapo_aime.py rather than the raw downloads.
    """
    train_path = os.path.join(expand_path(data_dir), TRAIN_FILE_NAME)
    val_path = os.path.join(expand_path(data_dir), VAL_FILE_NAME)
    return train_path, val_path


def policy_loss_config() -> dict | None:
    if POLICY_LOSS == "ppo":
        return {
            "clip_low_threshold": 1.0 - CLIP_RATIO_LOW,
            "clip_high_threshold": 1.0 + CLIP_RATIO_HIGH,
        }
    return None


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def grouped_minibatches(
    trajectories: Sequence[Trajectory],
    prompt_mini_batch_size: int,
) -> Iterable[list[Trajectory]]:
    grouped = defaultdict(list)
    for trajectory in trajectories:
        grouped[trajectory.prompt_group].append(trajectory)

    prompt_groups = list(grouped.keys())
    random.shuffle(prompt_groups)

    for group_batch in chunked(prompt_groups, prompt_mini_batch_size):
        minibatch = []
        for prompt_group in group_batch:
            minibatch.extend(grouped[prompt_group])
        if minibatch:
            yield minibatch


def overlong_filter_loss_mask(response_tokens: Sequence[int], stop_reason: str) -> list[float]:
    """DAPO Overlong Filtering: zero every token's loss weight when the response was truncated.

    Mirrors `skyrl.train.generators.utils.apply_overlong_filtering`, which keys off the engine's
    stop reason ("stop" = finished normally, "length" = hit max_tokens) rather than an EOS id.
    """
    if APPLY_OVERLONG_FILTERING and stop_reason != "stop":
        return [0.0] * len(response_tokens)
    return [1.0] * len(response_tokens)


def apply_soft_overlong_punishment(trajectories: Sequence[Trajectory]) -> dict[str, float]:
    """DAPO Soft Overlong Punishment: penalize responses that run into the last
    `OVERLONG_BUFFER_LEN` tokens of the generation budget.

    Mirrors `DAPOTrainer.postprocess_generator_output` in examples/train/algorithms/dapo/main_dapo.py:
    within the buffer the penalty grows linearly from 0 to `OVERLONG_BUFFER_PENALTY_FACTOR`; a response
    longer than the budget gets reward 0 (its loss is already masked by overlong filtering).
    Must run before `compute_advantages` so the GRPO group statistics see the penalized rewards.

    Returns:
        Metrics: fraction of responses penalized / truncated and the mean reward after the penalty.
    """
    if not trajectories:
        return {}
    max_exceed_length = MAX_GENERATE_LENGTH - OVERLONG_BUFFER_LEN
    num_penalized = 0
    num_truncated = 0
    for trajectory in trajectories:
        response_length = len(trajectory.response_tokens)
        if max_exceed_length < response_length <= MAX_GENERATE_LENGTH:
            exceed_length = response_length - max_exceed_length
            trajectory.reward -= exceed_length / OVERLONG_BUFFER_LEN * OVERLONG_BUFFER_PENALTY_FACTOR
            num_penalized += 1
        elif response_length > MAX_GENERATE_LENGTH:
            trajectory.reward = 0.0
        if trajectory.stop_reason != "stop":
            num_truncated += 1
    return {
        "overlong_penalized_ratio": num_penalized / len(trajectories),
        "truncated_ratio": num_truncated / len(trajectories),
        "avg_reward_after_penalty": sum(t.reward for t in trajectories) / len(trajectories),
    }


def compute_advantages(
    trajectories: Sequence[Trajectory],
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
) -> None:
    if not trajectories:
        return

    id2score = defaultdict(list)
    id2mean = dict()
    id2std = dict()

    for _, trajectory in enumerate(trajectories):
        id2score[trajectory.prompt_group].append(trajectory.reward)

    for prompt_id, trajectory_list in id2score.items():
        if len(trajectory_list) == 1:
            id2mean[prompt_id] = torch.tensor(0.0)
            id2std[prompt_id] = torch.tensor(1.0)
        elif len(trajectory_list) > 1:
            id2mean[prompt_id] = torch.mean(torch.tensor(id2score[prompt_id]))
            id2std[prompt_id] = torch.std(torch.tensor(id2score[prompt_id]))
        else:
            raise ValueError(f"No score in prompt id: {prompt_id}")
    for trajectory in trajectories:
        if grpo_norm_by_std:
            advantage = (trajectory.reward - id2mean[trajectory.prompt_group]) / (
                id2std[trajectory.prompt_group] + epsilon
            )
        else:
            advantage = trajectory.reward - id2mean[trajectory.prompt_group]
        trajectory.advantages = advantage.repeat(len(trajectory.response_tokens)).tolist()


def policy_learning_rate(optim_step: int) -> float:
    """Constant LR with linear warmup, counted in optimizer (mini-batch) steps.

    Mirrors SkyRL's `constant_with_warmup` scheduler (Megatron's OptimizerParamScheduler with
    init_lr=0): the server-side scheduler is disabled for Tinker, so the client ramps the LR.
    """
    if NUM_WARMUP_STEPS <= 0:
        return POLICY_LEARNING_RATE
    return POLICY_LEARNING_RATE * min(1.0, (optim_step + 1) / NUM_WARMUP_STEPS)


def adam_params(learning_rate: float) -> types.AdamParams:
    # The SkyRL-Train Tinker backend applies only `learning_rate` at optim_step (via set_lr).
    # betas/eps/weight_decay/max_grad_norm are fixed at optimizer creation from the server's
    # trainer.policy.optimizer_config, so weight_decay=0.1 must be set in backend_config.
    return types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1.0e-8,
    )


def average_metrics(metrics_list: Sequence[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def normalize_policy_minibatch_advantage(
    minibatch: Sequence[Trajectory],
) -> list[list[float]]:
    max_len = max(len(t.response_tokens) for t in minibatch)
    advantages = torch.zeros((len(minibatch), max_len), dtype=torch.float32)
    loss_mask = torch.zeros((len(minibatch), max_len), dtype=torch.float32)

    for row, trajectory in enumerate(minibatch):
        length = len(trajectory.response_tokens)
        advantages[row, :length] = torch.tensor(trajectory.advantages, dtype=torch.float32)
        loss_mask[row, :length] = torch.tensor(trajectory.loss_mask, dtype=torch.float32)

    normalized = apply_loss_reduction_to_advantages_minibatch(
        advantages=advantages,
        loss_mask=loss_mask,
        loss_reduction=LOSS_REDUCTION,
        micro_batch_size=MICRO_TRAIN_BATCH_SIZE,
        max_seq_len=MAX_PROMPT_LENGTH + MAX_GENERATE_LENGTH,
    )

    return [normalized[row, : len(trajectory.response_tokens)].tolist() for row, trajectory in enumerate(minibatch)]


def tensor_data_int(values: list[int]) -> types.TensorData:
    return types.TensorData.from_torch(torch.tensor(values, dtype=torch.int64))


def tensor_data_float(values: list[float]) -> types.TensorData:
    return types.TensorData.from_torch(torch.tensor(values, dtype=torch.float32))


def rollout_model_input(prompt_tokens: list[int], response_tokens: list[int]) -> types.ModelInput:
    prefix = prompt_tokens + response_tokens[:-1]
    return types.ModelInput.from_ints(prefix)


def build_policy_train_datum(
    prompt_tokens: list[int],
    response_tokens: list[int],
    old_logprobs: list[float],
    advantages: list[float],
    weights: list[float],
    rollout_logprobs: list[float] | None = None,
) -> types.Datum:
    loss_fn_inputs = {
        "target_tokens": tensor_data_int(response_tokens),
        "weights": tensor_data_float(weights),
        "logprobs": tensor_data_float(old_logprobs),
        "advantages": tensor_data_float(advantages),
    }
    if rollout_logprobs is not None:
        # SkyRL extension: lets the server apply off-policy correction against the rollout policy
        # while `logprobs` remains the PPO ratio denominator.
        loss_fn_inputs["rollout_logprobs"] = tensor_data_float(rollout_logprobs)
    return types.Datum(
        model_input=rollout_model_input(prompt_tokens, response_tokens),
        loss_fn_inputs=loss_fn_inputs,
    )


def build_forward_datum(prompt_tokens: list[int], response_tokens: list[int]) -> types.Datum:
    return types.Datum(
        model_input=rollout_model_input(prompt_tokens, response_tokens),
        loss_fn_inputs={
            "target_tokens": tensor_data_int(response_tokens),
            "weights": tensor_data_float([1.0] * len(response_tokens)),
        },
    )


def extract_logprobs(output: Any) -> list[float]:
    logprobs = output["logprobs"]
    if hasattr(logprobs, "data"):
        return [float(v) for v in logprobs.data]
    if isinstance(logprobs, dict) and "data" in logprobs:
        return [float(v) for v in logprobs["data"]]
    if isinstance(logprobs, list):
        return [float(v) for v in logprobs]
    raise TypeError(f"Unsupported forward output format: {type(logprobs)!r}")


def compute_old_logprobs(policy_client: tinker.TrainingClient, trajectories: Sequence[Trajectory]) -> None:
    """Fill `old_logprobs` with the training policy's logprobs before any update this step.

    Mirrors the native trainer's forward pass for `action_log_probs`. Must run before `train_policy`
    (while the training weights still equal the sampling weights).
    """
    for chunk in chunked(list(trajectories), FORWARD_BATCH_SIZE):
        data = [build_forward_datum(t.prompt_tokens, t.response_tokens) for t in chunk]
        result = policy_client.forward(data, "cross_entropy").result()
        for trajectory, output in zip(chunk, result.loss_fn_outputs, strict=True):
            logprobs = extract_logprobs(output)
            if len(logprobs) != len(trajectory.response_tokens):
                raise ValueError(
                    f"forward returned {len(logprobs)} logprobs for {len(trajectory.response_tokens)} response tokens"
                )
            trajectory.old_logprobs = logprobs


def train_policy(
    policy_client: tinker.TrainingClient,
    trajectories: Sequence[Trajectory],
    optim_step: int,
) -> tuple[dict[str, float], int]:
    """Run one DAPO update over `trajectories`.

    Args:
        optim_step: Number of optimizer steps taken so far (drives LR warmup).

    Returns:
        Averaged per-minibatch metrics and the updated optimizer step count.
    """
    all_metrics = []
    loss_fn_config = policy_loss_config()

    for _ in range(UPDATE_EPOCHS_PER_BATCH):
        for minibatch in grouped_minibatches(trajectories, POLICY_MINI_BATCH_SIZE):
            optimizer = adam_params(policy_learning_rate(optim_step))
            normalized_advantages = normalize_policy_minibatch_advantage(minibatch)
            data = [
                build_policy_train_datum(
                    t.prompt_tokens,
                    t.response_tokens,
                    t.old_logprobs,
                    advantages,
                    weights=t.loss_mask,
                    rollout_logprobs=t.rollout_logprobs if RECOMPUTE_OLD_LOGPROBS else None,
                )
                for t, advantages in zip(minibatch, normalized_advantages, strict=True)
            ]
            forward_result = policy_client.forward_backward(data, POLICY_LOSS, loss_fn_config).result()
            optim_result = policy_client.optim_step(optimizer).result()
            metrics = dict(forward_result.metrics)
            metrics.update(optim_result.metrics or {})
            all_metrics.append(metrics)
            optim_step += 1

    return average_metrics(all_metrics), optim_step


def compute_aime_reward(response_text: str, ground_truth: str) -> float:
    """Score a response with the same verifier as SkyRL's native `aime` env.

    The verifier reads the last `Answer: ...` line (optionally boxed) and compares it to the
    ground truth after normalization. Returns 1.0 if correct and -1.0 otherwise, matching
    `skyrl_gym.envs.aime.utils.compute_score`.
    """
    return float(aime_utils.compute_score(response_text, ground_truth)["score"])


def collect_rollouts(
    policy_client: tinker.TrainingClient,
    batch: Sequence[ExampleRecord],
    tokenizer,
    args: argparse.Namespace,
    global_step: int,
    *,
    eval_mode: bool,
) -> tuple[list[Trajectory], dict[str, float]]:
    sampling_client = policy_client.save_weights_and_get_sampling_client()
    temperature = EVAL_SAMPLING_TEMPERATURE if eval_mode else TRAIN_SAMPLING_TEMPERATURE
    top_p = EVAL_SAMPLING_TOP_P if eval_mode else TRAIN_SAMPLING_TOP_P
    n_samples = EVAL_N_SAMPLES_PER_PROMPT if eval_mode else N_SAMPLES_PER_PROMPT
    trajectories: list[Trajectory] = []
    prompt_rewards: list[list[float]] = []
    pending_samples: list[tuple[ExampleRecord, object]] = []

    for batch_offset, record in enumerate(batch):
        params = types.SamplingParams(
            max_tokens=MAX_GENERATE_LENGTH,
            seed=args.seed + global_step * 10_000 + batch_offset,
            temperature=temperature,
            stop_strings=SAMPLING_STOP_STRINGS,
            top_p=top_p,
            top_k=SAMPLING_TOP_K,
        )
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(record.prompt_tokens),
            num_samples=n_samples,
            sampling_params=params,
        )
        pending_samples.append((record, future))

    for prompt_group, (record, future) in enumerate(pending_samples):
        result = future.result()
        rewards_for_prompt: list[float] = []
        for sequence in result.sequences:
            response_tokens = list(sequence.tokens)
            if not response_tokens:
                continue
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            reward = compute_aime_reward(response_text, record.ground_truth)
            rollout_logprobs = list(sequence.logprobs or [0.0] * len(response_tokens))
            stop_reason = str(sequence.stop_reason)
            trajectories.append(
                Trajectory(
                    prompt_tokens=record.prompt_tokens,
                    response_tokens=response_tokens,
                    old_logprobs=rollout_logprobs,
                    rollout_logprobs=rollout_logprobs,
                    loss_mask=overlong_filter_loss_mask(response_tokens, stop_reason),
                    stop_reason=stop_reason,
                    advantages=[],
                    reward=reward,
                    question=record.question,
                    ground_truth=record.ground_truth,
                    response_text=response_text,
                    prompt_group=prompt_group,
                )
            )
            rewards_for_prompt.append(reward)
        prompt_rewards.append(rewards_for_prompt)

    return trajectories, summarize_reward_metrics(prompt_rewards, n_samples_per_prompt=n_samples)


def summarize_reward_metrics(
    prompt_rewards: Sequence[Sequence[float]],
    *,
    n_samples_per_prompt: int,
) -> dict[str, float]:
    flat_rewards = [reward for rewards in prompt_rewards for reward in rewards]
    pass_at_n = 0.0
    if prompt_rewards:
        pass_at_n = sum(1 for rewards in prompt_rewards if any(r > 0.0 for r in rewards)) / len(prompt_rewards)

    avg_reward = float(sum(flat_rewards) / len(flat_rewards)) if flat_rewards else 0.0
    mean_positive_reward = (
        float(sum(max(reward, 0.0) for reward in flat_rewards) / len(flat_rewards)) if flat_rewards else 0.0
    )

    return {
        "avg_reward": avg_reward,
        "avg_raw_reward": avg_reward,
        "pass_at_n": pass_at_n,
        f"avg_pass_at_{n_samples_per_prompt}": pass_at_n,
        "mean_positive_reward": mean_positive_reward,
        "num_prompts": float(len(prompt_rewards)),
        "num_trajectories": float(len(flat_rewards)),
    }


def evaluate_policy(
    policy_client: tinker.TrainingClient,
    eval_records: Sequence[ExampleRecord],
    tokenizer,
    args: argparse.Namespace,
    global_step: int,
) -> dict[str, float]:
    total_reward = 0.0
    total_positive_reward = 0.0
    total_passes = 0.0
    total_prompts = 0.0
    total_trajectories = 0.0
    eval_steps = 0

    for batch in chunked(list(eval_records), EVAL_BATCH_SIZE):
        _, metrics = collect_rollouts(
            policy_client,
            batch,
            tokenizer,
            args,
            global_step=global_step + eval_steps,
            eval_mode=True,
        )
        total_reward += metrics["avg_raw_reward"] * metrics["num_trajectories"]
        total_positive_reward += metrics["mean_positive_reward"] * metrics["num_trajectories"]
        total_passes += metrics["pass_at_n"] * metrics["num_prompts"]
        total_prompts += metrics["num_prompts"]
        total_trajectories += metrics["num_trajectories"]
        eval_steps += 1
        if args.max_eval_steps is not None and eval_steps >= args.max_eval_steps:
            break

    # avg_score is the mean over all EVAL_N_SAMPLES_PER_PROMPT samples (same as native `eval/all/avg_score`);
    # pass_at_n is the fraction of prompts with at least one correct sample.
    avg_score = total_reward / total_trajectories if total_trajectories else 0.0
    mean_positive_reward = total_positive_reward / total_trajectories if total_trajectories else 0.0
    pass_at_n = total_passes / total_prompts if total_prompts else 0.0
    return {
        "eval/all/avg_score": avg_score,
        f"eval/all/pass_at_{EVAL_N_SAMPLES_PER_PROMPT}": pass_at_n,
        "eval/all/mean_positive_reward": mean_positive_reward,
        "eval/num_steps": float(eval_steps),
    }


def metrics_path(output_dir: str | None) -> Path | None:
    if not output_dir:
        return None
    out_dir = Path(expand_path(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "metrics.jsonl"


def append_metrics(output_dir: str | None, payload: dict) -> None:
    path = metrics_path(output_dir)
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def save_checkpoint(
    policy_client: tinker.TrainingClient,
    step: int,
) -> str:
    tag = f"step_{step:06d}"
    return policy_client.save_state(f"policy_{tag}").result().path


def run_training(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    wandb_logger = WandbLogger(args.output_dir, run_config={"base_model": args.model, "lora_rank": args.lora_rank})
    logger.info(
        "wandb status: enabled=%s project=%s run_name=%s entity=%s",
        wandb_logger.enabled,
        os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        os.environ.get("WANDB_RUN_NAME", DEFAULT_WANDB_RUN_NAME),
        os.environ.get("WANDB_ENTITY"),
    )

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    # rank=0 selects full-parameter fine-tuning on the SkyRL server.
    policy_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = policy_client.get_tokenizer()
    train_path, val_path = build_split_paths(args.data_dir)
    train_records = load_split(train_path, tokenizer, max_prompt_length=MAX_PROMPT_LENGTH)
    eval_records = load_split(val_path, tokenizer, max_prompt_length=MAX_PROMPT_LENGTH)

    logger.info(
        "Starting DAPO Tinker training: train_examples=%s, eval_examples=%s, model=%s, lora_rank=%s, "
        "policy_loss=%s, recompute_old_logprobs=%s",
        len(train_records),
        len(eval_records),
        args.model,
        args.lora_rank,
        POLICY_LOSS,
        RECOMPUTE_OLD_LOGPROBS,
    )

    global_step = 0
    train_steps = 0
    optim_step = 0

    try:
        if EVAL_BEFORE_TRAIN:
            eval_metrics = evaluate_policy(policy_client, eval_records, tokenizer, args, global_step=global_step)
            logger.info("Initial eval: %s", eval_metrics)
            payload = {"step": global_step, **eval_metrics}
            append_metrics(args.output_dir, payload)
            wandb_logger.log(payload)

        for epoch in range(TRAIN_EPOCHS):
            epoch_rng = random.Random(args.seed + epoch)
            epoch_records = list(train_records)
            epoch_rng.shuffle(epoch_records)

            for batch in chunked(epoch_records, TRAIN_BATCH_SIZE):
                step_start = time.time()
                trajectories, rollout_metrics = collect_rollouts(
                    policy_client,
                    batch,
                    tokenizer,
                    args,
                    global_step=global_step,
                    eval_mode=False,
                )
                if not trajectories:
                    logger.warning("Skipping empty rollout batch at step %s", global_step)
                    continue

                if RECOMPUTE_OLD_LOGPROBS:
                    compute_old_logprobs(policy_client, trajectories)
                overlong_metrics = apply_soft_overlong_punishment(trajectories)
                compute_advantages(trajectories)
                policy_metrics, optim_step = train_policy(policy_client, trajectories, optim_step)

                global_step += 1
                train_steps += 1
                elapsed = time.time() - step_start

                log_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "time/step_seconds": elapsed,
                    "policy/optim_step": optim_step,
                    "rollout/avg_reward": rollout_metrics["avg_reward"],
                    f"rollout/pass_at_{N_SAMPLES_PER_PROMPT}": rollout_metrics["pass_at_n"],
                    "rollout/num_trajectories": rollout_metrics["num_trajectories"],
                    "reward/avg_raw_reward": rollout_metrics["avg_raw_reward"],
                    f"reward/avg_pass_at_{N_SAMPLES_PER_PROMPT}": rollout_metrics[
                        f"avg_pass_at_{N_SAMPLES_PER_PROMPT}"
                    ],
                    "reward/mean_positive_reward": rollout_metrics["mean_positive_reward"],
                }
                log_payload.update({f"reward/{k}": v for k, v in overlong_metrics.items()})
                log_payload.update({f"policy/{k}": v for k, v in policy_metrics.items()})

                logger.info("Train step %s: %s", global_step, log_payload)
                append_metrics(args.output_dir, log_payload)
                wandb_logger.log(log_payload)

                if CKPT_INTERVAL > 0 and global_step % CKPT_INTERVAL == 0:
                    policy_ckpt_path = save_checkpoint(policy_client, global_step)
                    logger.info("Saved policy checkpoint at step %s: %s", global_step, policy_ckpt_path)
                    payload = {"step": global_step, "policy_path": policy_ckpt_path}
                    append_metrics(args.output_dir, payload)
                    wandb_logger.log(payload)

                if EVAL_INTERVAL > 0 and global_step % EVAL_INTERVAL == 0:
                    eval_metrics = evaluate_policy(
                        policy_client, eval_records, tokenizer, args, global_step=global_step
                    )
                    logger.info("Eval step %s: %s", global_step, eval_metrics)
                    payload = {"step": global_step, **eval_metrics}
                    append_metrics(args.output_dir, payload)
                    wandb_logger.log(payload)

                if args.max_train_steps is not None and train_steps >= args.max_train_steps:
                    logger.info("Reached max_train_steps=%s, stopping early", args.max_train_steps)
                    return
    finally:
        service_client.holder.close()
        wandb_logger.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger.info(
        "base_url=%s data_dir=%s output_dir=%s model=%s",
        args.base_url,
        args.data_dir,
        args.output_dir,
        args.model,
    )
    run_training(args)


if __name__ == "__main__":
    main()
