"""GRPO-style GSM8K AMD Tinker client using the public Tinker SDK.

The client prepares or loads GSM8K-style parquet data, samples groups of
responses, computes group-relative advantages from exact-match rewards, and
trains the policy with the public Tinker ``ppo`` loss.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import datasets
import torch
import tinker
from tinker import types


DEFAULT_BASE_URL = "http://localhost:9000"
DEFAULT_API_KEY = "tml-dummy"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT_DIR = "/tmp/skyrl-tinker-grpo"
DEFAULT_DATA_DIR = "/tmp/skyrl-tinker-grpo/gsm8k"

LORA_RANK = 32
LEARNING_RATE = 3.0e-5
POLICY_LOSS = "ppo"
GSM8K_SOURCE = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
STRICT_ANSWER_RE = re.compile(r"####\s*(-?[0-9\.,]+)")
FALLBACK_NUMBER_RE = re.compile(r"(-?[0-9\.,]+)")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GSM8KRecord:
    prompt_messages: list[dict[str, str]]
    prompt_tokens: list[int]
    question: str
    answer: str
    dataset_index: int


@dataclass
class Trajectory:
    prompt_tokens: list[int]
    response_tokens: list[int]
    old_logprobs: list[float]
    reward: float
    advantage: float
    prompt_group: int
    question: str
    answer: str
    response_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--base-model", default=os.environ.get("TINKER_GRPO_BASE_MODEL", DEFAULT_MODEL_NAME))
    parser.add_argument("--data-dir", default=os.environ.get("GSM8K_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--max-val-examples", type=int, default=128)
    parser.add_argument("--reprepare-data", action="store_true")
    parser.add_argument("--no-auto-prepare-data", action="store_true")
    return parser.parse_args()


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def write_jsonl(output_dir: str, payload: Mapping[str, Any]) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"accept": "application/json", "content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def unload_training_model(base_url: str, training_client: tinker.TrainingClient | None) -> None:
    if training_client is None:
        return
    model_id = getattr(training_client, "model_id", None)
    if not model_id:
        return
    try:
        response = post_json(base_url, "/api/v1/unload_model", {"model_id": model_id}, timeout=10.0)
        request_id = response["request_id"]
        post_json(base_url, "/api/v1/retrieve_future", {"request_id": request_id}, timeout=300.0)
        logger.info("Unloaded model %s", model_id)
    except (HTTPError, URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Failed to unload model %s: %s", model_id, exc)


def extract_solution(solution_str: str) -> str:
    match = STRICT_ANSWER_RE.search(solution_str)
    if match is None:
        raise ValueError(f"Could not extract GSM8K answer from: {solution_str!r}")
    return match.group(1).replace(",", "")


def process_gsm8k_row(example: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    question_raw = example["question"]
    answer_raw = example["answer"]
    return {
        "data_source": GSM8K_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": f"{question_raw} {GSM8K_INSTRUCTION}",
            }
        ],
        "env_class": "gsm8k",
        "reward_spec": {
            "method": "rule",
            "ground_truth": extract_solution(answer_raw),
        },
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer_raw,
            "question": question_raw,
        },
    }


def write_split(dataset: datasets.Dataset, path: Path, split: str, max_examples: int, seed: int) -> None:
    dataset = dataset.shuffle(seed=seed)
    if max_examples > 0:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    processed = dataset.map(lambda example, idx: process_gsm8k_row(example, split, idx), with_indices=True)
    processed.to_parquet(str(path))
    logger.info("Prepared %s GSM8K records at %s", len(processed), path)


def ensure_gsm8k_data(args: argparse.Namespace) -> tuple[Path, Path]:
    data_dir = Path(os.path.expanduser(args.data_dir))
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "validation.parquet"
    if train_path.exists() and val_path.exists() and not args.reprepare_data:
        return train_path, val_path

    if args.no_auto_prepare_data:
        raise FileNotFoundError(
            f"Missing {train_path} or {val_path}. Run without --no-auto-prepare-data to create a small GSM8K subset."
        )

    logger.info("Preparing GSM8K parquet data in %s", data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.load_dataset(GSM8K_SOURCE, GSM8K_CONFIG)
    write_split(dataset["train"], train_path, "train", args.max_train_examples, seed=args.seed)
    write_split(dataset["test"], val_path, "test", args.max_val_examples, seed=args.seed + 1)
    return train_path, val_path


def prompt_tokens_for_messages(
    tokenizer: Any, messages: list[dict[str, str]], max_prompt_length: int
) -> list[int] | None:
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False, tokenize=True)
    token_list = list(tokens["input_ids"] if isinstance(tokens, Mapping) else tokens)
    if len(token_list) > max_prompt_length:
        return None
    return token_list


def load_split(path: Path, tokenizer: Any, max_prompt_length: int, limit: int | None = None) -> list[GSM8KRecord]:
    dataset = datasets.load_dataset("parquet", data_files=str(path), keep_in_memory=True)["train"]
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    filtered = 0
    for idx, row in enumerate(dataset):
        prompt_tokens = prompt_tokens_for_messages(tokenizer, row["prompt"], max_prompt_length=max_prompt_length)
        if prompt_tokens is None:
            filtered += 1
            continue
        records.append(
            GSM8KRecord(
                prompt_messages=row["prompt"],
                prompt_tokens=prompt_tokens,
                question=row.get("extra_info", {}).get("question", ""),
                answer=str(row["reward_spec"]["ground_truth"]).strip(),
                dataset_index=idx,
            )
        )
    logger.info("Loaded %s records from %s (filtered %s long prompts)", len(records), path, filtered)
    return records


def sample_train_records(records: Sequence[GSM8KRecord], num_prompts: int, seed: int, step: int) -> list[GSM8KRecord]:
    if num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if len(records) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts, but only loaded {len(records)} records. "
            "Increase --max-train-examples or lower --num-prompts."
        )
    rng = random.Random(seed + step * 1_000_003)
    return rng.sample(list(records), num_prompts)


def extract_answer(text: str) -> str | None:
    match = STRICT_ANSWER_RE.search(text)
    if match is not None:
        return match.group(1).replace(",", "")
    matches = FALLBACK_NUMBER_RE.findall(text)
    for candidate in reversed(matches):
        candidate = candidate.replace(",", "")
        if candidate not in {"", "."}:
            return candidate
    return None


def reward_response(text: str, answer: str) -> float:
    return 1.0 if extract_answer(text) == answer else 0.0


def tensor_i64(values: list[int]) -> types.TensorData:
    return types.TensorData.from_torch(torch.tensor(values, dtype=torch.int64))


def tensor_f32(values: list[float]) -> types.TensorData:
    return types.TensorData.from_torch(torch.tensor(values, dtype=torch.float32))


def model_input_for_rollout(prompt_tokens: list[int], response_tokens: list[int]) -> types.ModelInput:
    return types.ModelInput.from_ints(prompt_tokens + response_tokens[:-1])


def build_policy_datum(trajectory: Trajectory) -> types.Datum:
    token_count = len(trajectory.response_tokens)
    return types.Datum(
        model_input=model_input_for_rollout(trajectory.prompt_tokens, trajectory.response_tokens),
        loss_fn_inputs={
            "target_tokens": tensor_i64(trajectory.response_tokens),
            "weights": tensor_f32([1.0] * token_count),
            "logprobs": tensor_f32(trajectory.old_logprobs),
            "advantages": tensor_f32([trajectory.advantage] * token_count),
        },
    )


def normalise_logprobs(logprobs: Sequence[float] | None, token_count: int) -> list[float]:
    values = [float(value or 0.0) for value in (logprobs or [])]
    if len(values) < token_count:
        values.extend([0.0] * (token_count - len(values)))
    return values[:token_count]


def assign_group_advantages(trajectories: list[Trajectory]) -> int:
    zero_variance_groups = 0
    by_group: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        by_group.setdefault(trajectory.prompt_group, []).append(trajectory)

    for group in by_group.values():
        rewards = torch.tensor([trajectory.reward for trajectory in group], dtype=torch.float32)
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        if float(std) <= 1.0e-8:
            zero_variance_groups += 1
            advantages = rewards - mean
        else:
            advantages = (rewards - mean) / (std + 1.0e-8)
        for trajectory, advantage in zip(group, advantages.tolist(), strict=True):
            trajectory.advantage = float(advantage)
    return zero_variance_groups


def collect_rollouts(
    sampling_client: Any,
    tokenizer: Any,
    records: Sequence[GSM8KRecord],
    args: argparse.Namespace,
    step: int,
) -> tuple[list[Trajectory], dict[str, float]]:
    pending = []
    for index, record in enumerate(records):
        params = types.SamplingParams(
            max_tokens=args.max_tokens,
            seed=args.seed + step * 10_000 + index,
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
        )
        future = sampling_client.sample(
            prompt=types.ModelInput.from_ints(record.prompt_tokens),
            num_samples=args.group_size,
            sampling_params=params,
        )
        pending.append((index, record, future))

    trajectories: list[Trajectory] = []
    rewards_by_group: list[list[float]] = []
    for prompt_group, record, future in pending:
        result = future.result()
        rewards_for_group = []
        for sequence in result.sequences:
            response_tokens = list(sequence.tokens)
            if not response_tokens:
                continue
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            reward = reward_response(response_text, record.answer)
            rewards_for_group.append(reward)
            trajectories.append(
                Trajectory(
                    prompt_tokens=list(record.prompt_tokens),
                    response_tokens=response_tokens,
                    old_logprobs=normalise_logprobs(sequence.logprobs, len(response_tokens)),
                    reward=reward,
                    advantage=0.0,
                    prompt_group=prompt_group,
                    question=record.question,
                    answer=record.answer,
                    response_text=response_text,
                )
            )
        rewards_by_group.append(rewards_for_group)

    zero_variance_groups = assign_group_advantages(trajectories)
    flat_rewards = [reward for group in rewards_by_group for reward in group]
    pass_at_group = (
        sum(1 for group in rewards_by_group if any(reward > 0.0 for reward in group)) / len(rewards_by_group)
        if rewards_by_group
        else 0.0
    )
    metrics = {
        "num_prompts": float(len(records)),
        "num_trajectories": float(len(trajectories)),
        "avg_reward": float(sum(flat_rewards) / len(flat_rewards)) if flat_rewards else 0.0,
        "pass_at_group": float(pass_at_group),
        "zero_variance_groups": float(zero_variance_groups),
    }
    return trajectories, metrics


def average_metrics(metrics_list: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in metrics_list:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def train_policy(
    training_client: tinker.TrainingClient,
    trajectories: Sequence[Trajectory],
    learning_rate: float,
) -> dict[str, float]:
    all_metrics = []
    optimizer = types.AdamParams(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        eps=1.0e-8,
    )
    for minibatch in chunked(list(trajectories), 32):
        data = [build_policy_datum(trajectory) for trajectory in minibatch]
        forward_result = training_client.forward_backward(data, POLICY_LOSS).result()
        optim_result = training_client.optim_step(optimizer).result()
        metrics = dict(forward_result.metrics)
        metrics.update(optim_result.metrics or {})
        all_metrics.append(metrics)
    return average_metrics(all_metrics)


def sample_preview(sampling_client: Any, tokenizer: Any, record: GSM8KRecord, args: argparse.Namespace) -> str:
    params = types.SamplingParams(
        max_tokens=args.max_tokens,
        seed=args.seed + 999_999,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )
    result = sampling_client.sample(
        prompt=types.ModelInput.from_ints(record.prompt_tokens),
        num_samples=1,
        sampling_params=params,
    ).result()
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    random.seed(args.seed)
    base_url = args.base_url.rstrip("/")

    logger.info("base_url=%s model=%s data_dir=%s", args.base_url, args.base_model, args.data_dir)
    train_path, val_path = ensure_gsm8k_data(args)

    service_client = None
    training_client = None
    try:
        service_client = tinker.ServiceClient(base_url=base_url, api_key=args.api_key)
        training_client = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=LORA_RANK,
            seed=args.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata={"example": "skyrl-amd-tinker-grpo-gsm8k"},
        )
        tokenizer = training_client.get_tokenizer()
        train_records = load_split(train_path, tokenizer, args.max_prompt_length, limit=args.max_train_examples)
        val_records = load_split(val_path, tokenizer, args.max_prompt_length, limit=args.max_val_examples)
        if not train_records:
            raise RuntimeError("No train records were loaded")

        for step in range(1, args.max_train_steps + 1):
            step_start = time.time()
            step_records = sample_train_records(train_records, args.num_prompts, args.seed, step)
            sampling_client = training_client.save_weights_and_get_sampling_client()
            trajectories, rollout_metrics = collect_rollouts(
                sampling_client,
                tokenizer,
                step_records,
                args,
                step,
            )
            if not trajectories:
                raise RuntimeError("No trajectories were produced")

            train_metrics = train_policy(training_client, trajectories, LEARNING_RATE)
            elapsed = time.time() - step_start
            payload = {
                "step": step,
                "time/step_seconds": elapsed,
                **{f"rollout/{key}": value for key, value in rollout_metrics.items()},
                **{f"train/{key}": value for key, value in train_metrics.items()},
            }
            write_jsonl(args.output_dir, payload)
            logger.info("Train step %s: %s", step, payload)

        sampling_client = training_client.save_weights_and_get_sampling_client()
        preview_record = val_records[0] if val_records else train_records[0]
        preview = sample_preview(sampling_client, tokenizer, preview_record, args)
        logger.info("Final sample question: %s", preview_record.question)
        logger.info("Final sample expected answer: %s", preview_record.answer)
        logger.info("Final sample response: %r", preview)
        logger.info("PASS")
    finally:
        unload_training_model(base_url, training_client)
        if service_client is not None:
            service_client.holder.close()


if __name__ == "__main__":
    main()
