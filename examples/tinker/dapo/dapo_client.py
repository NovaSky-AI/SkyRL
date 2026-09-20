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
import re
import datasets
from dataclasses import dataclass
from collections import defaultdict
import json
import time

from skyrl.backends.skyrl_train.utils.ppo_utils import (
    apply_loss_reduction_to_advantages_minibatch,
)


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATA_DIR = os.path.expanduser("~/data/gsm8k")
DEFAULT_CKPT_DIR = os.path.expanduser("~/ckpts/gsm8k_1.5B_ckpt_ppo")
DEFAULT_WANDB_PROJECT = "gsm8k"
DEFAULT_WANDB_RUN_NAME = "gsm8k_tinker_ppo"
DEFAULT_LORA_RANK = 0
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024
POLICY_MINI_BATCH_SIZE = 256
CRITIC_MINI_BATCH_SIZE = 256
UPDATE_EPOCHS_PER_BATCH = 1
N_SAMPLES_PER_PROMPT = 5
POLICY_LEARNING_RATE = 1.0e-6
CRITIC_LEARNING_RATE = 5.0e-6
MAX_PROMPT_LENGTH = 512
POLICY_LOSS = "ppo"
EVAL_BEFORE_TRAIN = True
CLIP_RATIO_LOW = 0.2   # eps_clip_low: ratio floor is 1 - 0.2 = 0.8
CLIP_RATIO_HIGH = 0.28 # eps_clip_high: ratio ceiling is 1 + 0.28 = 1.28
LOSS_REDUCTION = "token_mean"
MICRO_TRAIN_BATCH_SIZE = 64
MAX_GENERATE_LENGTH = 1024
SAMPLING_STOP_STRINGS: list[str] | None = None
SAMPLING_TOP_P = 1.0
SAMPLING_TOP_K = -1
TRAIN_SAMPLING_TEMPERATURE = 1.0
EVAL_SAMPLING_TEMPERATURE = 0.0
EVAL_N_SAMPLES_PER_PROMPT = 1
STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.,]+)")
FLEXIBLE_ANSWER_RE = re.compile(r"(\-?[0-9\.,]+)")
CKPT_INTERVAL = 10
EVAL_INTERVAL = 5


class WandbLogger:
    def __init__(self, output_dir: str | None):
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
                "base_model": DEFAULT_MODEL_NAME,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "policy_mini_batch_size": POLICY_MINI_BATCH_SIZE,
                "critic_mini_batch_size": CRITIC_MINI_BATCH_SIZE,
                "update_epochs_per_batch": UPDATE_EPOCHS_PER_BATCH,
                "n_samples_per_prompt": N_SAMPLES_PER_PROMPT,
                "policy_learning_rate": POLICY_LEARNING_RATE,
                "critic_learning_rate": CRITIC_LEARNING_RATE,
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
    old_logprobs: list[float]
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
                ground_truth=str(row["reward_spec"]["ground_truth"]).strip(),
                question=row.get("extra_info", {}).get("question", ""),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    return parser.parse_args()


def build_split_paths(data_dir: str) -> tuple[str, str]:
    train_path = os.path.join(expand_path(data_dir), "train.parquet")
    val_path = os.path.join(expand_path(data_dir), "validation.parquet")
    return train_path, val_path


def policy_loss_config() -> dict | None:
    if POLICY_LOSS == 'ppo':
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
            advantage = (trajectory.reward - id2mean[trajectory.prompt_group]) / (id2std[trajectory.prompt_group] + epsilon)
        else:
            advantage = trajectory.reward - id2mean[trajectory.prompt_group]
        trajectory.advantages = advantage.repeat(len(trajectory.response_tokens)).tolist()


def adam_params(learning_rate: float) -> types.AdamParams:
    # The Tinker backend currently applies only the learning rate at optim step.
    # These fixed Adam fields are passed solely to satisfy the request schema.
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
        loss_mask[row, :length] = 1.0

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
) -> types.Datum:
    weights = [1.0] * len(response_tokens)
    return types.Datum(
        model_input=rollout_model_input(prompt_tokens, response_tokens),
        loss_fn_inputs={
            "target_tokens": tensor_data_int(response_tokens),
            "weights": tensor_data_float(weights),
            "logprobs": tensor_data_float(old_logprobs),
            "advantages": tensor_data_float(advantages),
        },
    )


def train_policy(
    policy_client: tinker.TrainingClient,
    trajectories: Sequence[Trajectory],
) -> dict[str, float]:
    all_metrics = []
    optimizer = adam_params(POLICY_LEARNING_RATE)
    loss_fn_config = policy_loss_config()

    for _ in range(UPDATE_EPOCHS_PER_BATCH):
        for minibatch in grouped_minibatches(trajectories, POLICY_MINI_BATCH_SIZE):
            normalized_advantages = normalize_policy_minibatch_advantage(minibatch)
            data = [
                build_policy_train_datum(
                    t.prompt_tokens,
                    t.response_tokens,
                    t.old_logprobs,
                    advantages,
                )
                for t, advantages in zip(minibatch, normalized_advantages, strict=True)
            ]
            forward_result = policy_client.forward_backward(data, POLICY_LOSS, loss_fn_config).result()
            optim_result = policy_client.optim_step(optimizer).result()
            metrics = dict(forward_result.metrics)
            metrics.update(optim_result.metrics or {})
            all_metrics.append(metrics)

    return average_metrics(all_metrics)


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    if method == "strict":
        match = STRICT_ANSWER_RE.search(solution_str)
        if match is None:
            return None
        return match.group(1).replace(",", "").replace("$", "")

    answer = FLEXIBLE_ANSWER_RE.findall(solution_str)
    for candidate in reversed(answer):
        if candidate not in {"", "."}:
            return candidate.replace(",", "").replace("$", "")
    return None


def compute_gsm8k_reward(response_text: str, ground_truth: str) -> float:
    answer = extract_solution(response_text, method="strict")
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth else 0.0


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
            top_p=SAMPLING_TOP_P,
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
            reward = compute_gsm8k_reward(response_text, record.ground_truth)
            old_logprobs = list(sequence.logprobs or [0.0] * len(response_tokens))
            trajectories.append(
                Trajectory(
                    prompt_tokens=record.prompt_tokens,
                    response_tokens=response_tokens,
                    old_logprobs=old_logprobs,
                    values=[],
                    advantages=[],
                    returns=[],
                    token_rewards=[],
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

    avg_reward = total_reward / total_trajectories if total_trajectories else 0.0
    mean_positive_reward = total_positive_reward / total_trajectories if total_trajectories else 0.0
    pass_at_1 = total_passes / total_prompts if total_prompts else 0.0
    return {
        "eval/avg_reward": avg_reward,
        "eval/pass_at_1": pass_at_1,
        "eval/num_steps": float(eval_steps),
        "eval/all/avg_score": avg_reward,
        f"eval/all/pass_at_{EVAL_N_SAMPLES_PER_PROMPT}": pass_at_1,
        "eval/all/mean_positive_reward": mean_positive_reward,
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
    wandb_logger = WandbLogger(args.output_dir)
    logger.info(
        "wandb status: enabled=%s project=%s run_name=%s entity=%s",
        wandb_logger.enabled,
        os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
        os.environ.get("WANDB_RUN_NAME", DEFAULT_WANDB_RUN_NAME),
        os.environ.get("WANDB_ENTITY"),
    )

    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    policy_client = service_client.create_lora_training_client(
        base_model=DEFAULT_MODEL_NAME,
        rank=DEFAULT_LORA_RANK,
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
        "Starting DAPO Tinker training: train_examples=%s, eval_examples=%s, model=%s, policy_loss=%s",
        len(train_records),
        len(eval_records),
        DEFAULT_MODEL_NAME,
        POLICY_LOSS,
    )

    global_step = 0
    train_steps = 0

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

                compute_advantages(trajectories)
                policy_metrics = train_policy(policy_client, trajectories)

                global_step += 1
                train_steps += 1
                elapsed = time.time() - step_start

                log_payload = {
                    "step": global_step,
                    "epoch": epoch,
                    "time/step_seconds": elapsed,
                    "rollout/avg_reward": rollout_metrics["avg_reward"],
                    f"rollout/pass_at_{N_SAMPLES_PER_PROMPT}": rollout_metrics["pass_at_n"],
                    "rollout/num_trajectories": rollout_metrics["num_trajectories"],
                    "reward/avg_raw_reward": rollout_metrics["avg_raw_reward"],
                    f"reward/avg_pass_at_{N_SAMPLES_PER_PROMPT}": rollout_metrics[
                        f"avg_pass_at_{N_SAMPLES_PER_PROMPT}"
                    ],
                    "reward/mean_positive_reward": rollout_metrics["mean_positive_reward"],
                }
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    logger.info(
        "base_url=%s data_dir=%s output_dir=%s model=%s",
        args.base_url,
        args.data_dir,
        args.output_dir,
        DEFAULT_MODEL_NAME,
    )
    run_training(args)


if __name__ == "__main__":
    main()