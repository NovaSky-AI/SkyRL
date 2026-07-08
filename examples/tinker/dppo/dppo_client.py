"""
DPPO (Divergence PPO, https://arxiv.org/abs/2602.04879) RL training against
SkyRL's Tinker API server with the Megatron backend.

Usage:
    # Terminal 1 (server; see the script for the backend config)
    bash examples/tinker/dppo/run_tinker_server_megatron.sh

    # Terminal 2 (client)
    TINKER_API_KEY=tml-dummy uv run --isolated --extra tinker \
        --with torch --with ./skyrl-gym \
        python examples/tinker/dppo/dppo_client.py

This is the Tinker-client counterpart of
examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora.sh with the
policy loss swapped from dual-clip PPO to DPPO. Client-side pieces mirror that
recipe: DAPO math-17k train data, AIME-2024 eval data, the `aime` env reward,
soft overlong punishment, overlong filtering, GRPO advantages, token-mean loss
scaling, and LR 1e-5 with linear warmup. Ratio clipping parameters are
replaced by DPPO's divergence thresholds (delta_low/delta_high, sent per
request via loss_fn_config); the divergence variant (binary_tv/binary_kl) is
server-side config (`trainer.algorithm.dppo.dppo_type`).

The sampling logprobs returned by the server are passed as `logprobs` in the
loss inputs; the backend uses them as the behavior policy for both the DPPO
importance ratio and the divergence mask.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import datasets
import tinker
import torch
from tinker import types

from skyrl_gym.envs.aime.utils import compute_score

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_DATA_DIR = os.path.expanduser("~/data/dapo")


@dataclass
class ExampleRecord:
    prompt_tokens: list[int]
    ground_truth: str
    dataset_index: int


@dataclass
class Trajectory:
    prompt_tokens: list[int]
    response_tokens: list[int]
    sampling_logprobs: list[float]
    stop_reason: str
    reward: float = 0.0
    advantage: float = 0.0
    loss_weight: float = 1.0
    prompt_group: int = -1
    advantages_scaled: list[float] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("SKYRL_SERVICE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", default=None, help="Defaults to <data-dir>/dapo-math-17k-cleaned.parquet")
    parser.add_argument("--val-file", default=None, help="Defaults to <data-dir>/aime-2024-cleaned.parquet")
    parser.add_argument("--output-dir", default=os.path.expanduser("~/ckpts/dppo_tinker_qwen3_6_35b_a3b"))
    parser.add_argument("--seed", type=int, default=42)

    # Run-size parameters (defaults mirror the DAPO repro script; shrink for smoke tests)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=128, help="Prompts per training step")
    parser.add_argument("--policy-mini-batch-size", type=int, default=32, help="Prompts per optimizer step")
    parser.add_argument("--n-samples-per-prompt", type=int, default=16)
    parser.add_argument("--update-epochs-per-batch", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-generate-length", type=int, default=8192)

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    # Optimizer (LR applied per optim_step; scheduler is client-driven)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-warmup-steps", type=int, default=40)

    # DPPO loss parameters (0.2 recommended for binary_tv; 0.05 for binary_kl)
    parser.add_argument("--dppo-delta-low", type=float, default=0.2)
    parser.add_argument("--dppo-delta-high", type=float, default=0.2)

    # DAPO reward shaping (overlong buffer scales with max generate length)
    parser.add_argument("--overlong-buffer-len", type=int, default=4096)
    parser.add_argument("--overlong-buffer-penalty-factor", type=float, default=1.0)
    parser.add_argument("--apply-overlong-filtering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grpo-norm-by-std", action=argparse.BooleanOptionalAction, default=True)

    # Eval (disabled when --eval-interval 0)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-n-samples-per-prompt", type=int, default=32)
    parser.add_argument("--eval-top-p", type=float, default=0.7)
    parser.add_argument("--eval-max-prompts", type=int, default=None)

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--ckpt-interval", type=int, default=5)
    return parser.parse_args()


def append_metrics(output_dir: str | None, payload: dict) -> None:
    if not output_dir:
        return
    out = Path(os.path.expanduser(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def chunked(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_split(path: str, tokenizer, max_prompt_length: int) -> list[ExampleRecord]:
    dataset = datasets.load_dataset("parquet", data_files=os.path.expanduser(path), keep_in_memory=True)["train"]
    records: list[ExampleRecord] = []
    filtered = 0
    for idx, row in enumerate(dataset):
        messages = list(row["prompt"])
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
        if len(tokens) > max_prompt_length:
            filtered += 1
            continue
        records.append(
            ExampleRecord(
                prompt_tokens=list(tokens),
                ground_truth=str(row["reward_model"]["ground_truth"]).strip(),
                dataset_index=idx,
            )
        )
    logger.info("Loaded %s records from %s (filtered %s long prompts)", len(records), path, filtered)
    return records


def compute_reward(
    response_text: str,
    response_length: int,
    ground_truth: str,
    *,
    truncated: bool,
    max_generate_length: int,
    overlong_buffer_len: int,
    overlong_buffer_penalty_factor: float,
) -> float:
    """AIME env reward (+1/-1) with DAPO soft overlong punishment.

    Mirrors examples/train/algorithms/dapo/main_dapo.py: responses inside the
    overlong buffer get a linear penalty; truncated responses get 0 reward
    (their loss is additionally masked out by overlong filtering).
    """
    if truncated:
        return 0.0
    reward = float(compute_score(response_text, ground_truth)["score"])
    max_exceed_length = max_generate_length - overlong_buffer_len
    if response_length > max_exceed_length:
        exceed_length = response_length - max_exceed_length
        reward -= exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor
    return reward


def compute_grpo_advantages(trajectories: Sequence[Trajectory], norm_by_std: bool, epsilon: float = 1e-6) -> None:
    """Group-relative advantages (mirrors skyrl's compute_grpo_outcome_advantage)."""
    groups: dict[int, list[Trajectory]] = {}
    for t in trajectories:
        groups.setdefault(t.prompt_group, []).append(t)
    for members in groups.values():
        rewards = torch.tensor([t.reward for t in members], dtype=torch.float32)
        if len(members) == 1:
            mean, std = torch.tensor(0.0), torch.tensor(1.0)
        else:
            mean, std = rewards.mean(), rewards.std()
        for t, r in zip(members, rewards):
            t.advantage = float((r - mean) / (std + epsilon)) if norm_by_std else float(r - mean)


def scale_advantages_token_mean(minibatch: Sequence[Trajectory]) -> None:
    """token_mean loss reduction: pre-scale advantages by 1/(total valid tokens).

    The dppo loss on the server is sum-reduced over `advantage * ratio * mask`
    weighted by the loss mask, so dividing every token's advantage by the
    minibatch's total valid-token count yields a token-mean loss (mirrors
    apply_loss_reduction_to_advantages_minibatch's "token_mean" branch).
    """
    total_tokens = sum(len(t.response_tokens) * (1.0 if t.loss_weight > 0 else 0.0) for t in minibatch)
    total_tokens = max(total_tokens, 1.0)
    for t in minibatch:
        t.advantages_scaled = [t.advantage / total_tokens] * len(t.response_tokens)


def build_dppo_datum(t: Trajectory) -> types.Datum:
    # Tinker pre-shifts: model_input covers prompt + response[:-1]; targets are
    # the response tokens; weights carry the (possibly filtered) loss mask.
    prefix = t.prompt_tokens + t.response_tokens[:-1]
    return types.Datum(
        model_input=types.ModelInput.from_ints(prefix),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor(t.response_tokens, dtype=torch.int64)),
            "weights": types.TensorData.from_torch(
                torch.full((len(t.response_tokens),), t.loss_weight, dtype=torch.float32)
            ),
            "logprobs": types.TensorData.from_torch(torch.tensor(t.sampling_logprobs, dtype=torch.float32)),
            "advantages": types.TensorData.from_torch(torch.tensor(t.advantages_scaled, dtype=torch.float32)),
        },
    )


def grouped_minibatches(
    trajectories: Sequence[Trajectory], prompt_mini_batch_size: int, rng: random.Random
) -> Iterable[list[Trajectory]]:
    grouped: dict[int, list[Trajectory]] = {}
    for t in trajectories:
        grouped.setdefault(t.prompt_group, []).append(t)
    prompt_groups = list(grouped)
    rng.shuffle(prompt_groups)
    for group_batch in chunked(prompt_groups, prompt_mini_batch_size):
        minibatch = [t for g in group_batch for t in grouped[g]]
        if minibatch:
            yield minibatch


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
    n_samples = args.eval_n_samples_per_prompt if eval_mode else args.n_samples_per_prompt
    top_p = args.eval_top_p if eval_mode else args.top_p

    pending = []
    for offset, record in enumerate(batch):
        params = types.SamplingParams(
            max_tokens=args.max_generate_length,
            seed=args.seed + global_step * 10_000 + offset,
            temperature=args.temperature,
            top_p=top_p,
        )
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(record.prompt_tokens),
            num_samples=n_samples,
            sampling_params=params,
        )
        pending.append((record, future))

    trajectories: list[Trajectory] = []
    prompt_rewards: list[list[float]] = []
    num_truncated = 0
    for prompt_group, (record, future) in enumerate(pending):
        result = future.result()
        rewards_for_prompt = []
        for sequence in result.sequences:
            response_tokens = list(sequence.tokens)
            if not response_tokens:
                continue
            truncated = sequence.stop_reason == "length"
            num_truncated += int(truncated)
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            reward = compute_reward(
                response_text,
                len(response_tokens),
                record.ground_truth,
                truncated=truncated,
                max_generate_length=args.max_generate_length,
                overlong_buffer_len=args.overlong_buffer_len,
                overlong_buffer_penalty_factor=args.overlong_buffer_penalty_factor,
            )
            trajectories.append(
                Trajectory(
                    prompt_tokens=record.prompt_tokens,
                    response_tokens=response_tokens,
                    sampling_logprobs=list(sequence.logprobs or [0.0] * len(response_tokens)),
                    stop_reason=sequence.stop_reason,
                    reward=reward,
                    loss_weight=0.0 if (truncated and args.apply_overlong_filtering) else 1.0,
                    prompt_group=prompt_group,
                )
            )
            rewards_for_prompt.append(reward)
        prompt_rewards.append(rewards_for_prompt)

    num_correct = sum(1 for t in trajectories if t.reward > 0.0)
    metrics = {
        "avg_reward": sum(t.reward for t in trajectories) / max(len(trajectories), 1),
        "accuracy": num_correct / max(len(trajectories), 1),
        "pass_at_n": (
            sum(1 for rewards in prompt_rewards if any(r > 0.0 for r in rewards)) / max(len(prompt_rewards), 1)
        ),
        "num_trajectories": float(len(trajectories)),
        "num_truncated": float(num_truncated),
        "avg_response_length": (sum(len(t.response_tokens) for t in trajectories) / max(len(trajectories), 1)),
    }
    return trajectories, metrics


def adam_params(learning_rate: float) -> types.AdamParams:
    # Only the LR is applied dynamically by the SkyRL backend; the other Adam
    # fields are fixed at optimizer creation and passed to satisfy the schema.
    return types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.999, eps=1.0e-8)


def train_policy_dppo(
    policy_client: tinker.TrainingClient,
    trajectories: Sequence[Trajectory],
    args: argparse.Namespace,
    learning_rate: float,
    rng: random.Random,
) -> dict[str, float]:
    loss_fn_config = {"delta_low": args.dppo_delta_low, "delta_high": args.dppo_delta_high}
    all_metrics: list[dict[str, float]] = []
    for _ in range(args.update_epochs_per_batch):
        for minibatch in grouped_minibatches(trajectories, args.policy_mini_batch_size, rng):
            scale_advantages_token_mean(minibatch)
            data = [build_dppo_datum(t) for t in minibatch]
            fwd_bwd_result = policy_client.forward_backward(data, "dppo", loss_fn_config).result()
            optim_result = policy_client.optim_step(adam_params(learning_rate)).result()
            metrics = dict(fwd_bwd_result.metrics)
            metrics.update(optim_result.metrics or {})
            all_metrics.append(metrics)
    return average_metrics(all_metrics)


def average_metrics(metrics_list: Sequence[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def evaluate_policy(
    policy_client: tinker.TrainingClient,
    eval_records: Sequence[ExampleRecord],
    tokenizer,
    args: argparse.Namespace,
    global_step: int,
) -> dict[str, float]:
    records = list(eval_records)
    if args.eval_max_prompts is not None:
        records = records[: args.eval_max_prompts]
    _, metrics = collect_rollouts(
        policy_client,
        records,
        tokenizer,
        args,
        global_step=global_step,
        eval_mode=True,
    )
    return {f"eval/{k}": v for k, v in metrics.items()}


def run_training(args: argparse.Namespace, service_client: tinker.ServiceClient) -> None:
    random.seed(args.seed)
    rng = random.Random(args.seed)

    train_file = args.train_file or os.path.join(args.data_dir, "dapo-math-17k-cleaned.parquet")
    val_file = args.val_file or os.path.join(args.data_dir, "aime-2024-cleaned.parquet")

    policy_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
        seed=args.seed,
    )
    tokenizer = policy_client.get_tokenizer()

    train_records = load_split(train_file, tokenizer, args.max_prompt_length)
    eval_records = load_split(val_file, tokenizer, args.max_prompt_length)

    logger.info(
        "Starting DPPO Tinker training: model=%s train_examples=%s eval_examples=%s "
        "delta_low=%s delta_high=%s batch=%s n_samples=%s",
        args.model_name,
        len(train_records),
        len(eval_records),
        args.dppo_delta_low,
        args.dppo_delta_high,
        args.train_batch_size,
        args.n_samples_per_prompt,
    )

    global_step = 0
    for epoch in range(args.epochs):
        epoch_records = list(train_records)
        random.Random(args.seed + epoch).shuffle(epoch_records)

        for batch in chunked(epoch_records, args.train_batch_size):
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

            compute_grpo_advantages(trajectories, norm_by_std=args.grpo_norm_by_std)

            if args.num_warmup_steps > 0:
                learning_rate = args.lr * min(1.0, (global_step + 1) / args.num_warmup_steps)
            else:
                learning_rate = args.lr
            policy_metrics = train_policy_dppo(policy_client, trajectories, args, learning_rate, rng)

            global_step += 1
            payload = {
                "step": global_step,
                "epoch": epoch,
                "time/step_seconds": time.time() - step_start,
                "lr": learning_rate,
                **{f"rollout/{k}": v for k, v in rollout_metrics.items()},
                **{f"policy/{k}": v for k, v in policy_metrics.items()},
            }
            logger.info("Train step %s: %s", global_step, json.dumps(payload, sort_keys=True))
            append_metrics(args.output_dir, payload)

            if args.ckpt_interval > 0 and global_step % args.ckpt_interval == 0:
                path = policy_client.save_state(f"step_{global_step:06d}").result().path
                logger.info("Saved checkpoint at step %s: %s", global_step, path)

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                eval_metrics = evaluate_policy(policy_client, eval_records, tokenizer, args, global_step)
                logger.info("Eval step %s: %s", global_step, json.dumps(eval_metrics, sort_keys=True))
                append_metrics(args.output_dir, {"step": global_step, **eval_metrics})

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                logger.info("Reached max_train_steps=%s, stopping", args.max_train_steps)
                return


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    try:
        run_training(args, service_client)
    finally:
        service_client.holder.close()


if __name__ == "__main__":
    main()
