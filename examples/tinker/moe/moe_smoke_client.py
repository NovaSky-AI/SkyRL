"""LoRA GRPO run on GSM8K against SkyRL's Tinker server=

Usage (server must already be running -- see modal_moe_tinker.py):

    TINKER_API_KEY=tml-dummy uv run --extra tinker --with datasets \
        python examples/tinker/moe/moe_smoke_client.py \
        --base-url http://localhost:8000 \
        --base-model Qwen/Qwen3-30B-A3B-Base \
        --num-steps 50 --metrics-path ./moe_gsm8k_metrics.jsonl

For router replay, pass --enable-router-replay to echo moe routing matrices back into 
forward_backward datums -- passes through client 
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import time

import datasets
import numpy as np
import tinker
from tinker import types

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
STRICT_ANSWER_RE = re.compile(r"#### (\-?[0-9\.,]+)")

LOGPROB_DIFF_KEYS = (
    "policy/rollout_train_logprobs_abs_diff_mean:mean",
    "policy/rollout_train_logprobs_abs_diff_std:mean",
    "policy/rollout_train_logprobs_abs_diff_max:max",
)


def tensor_data_int(values: list[int]) -> types.TensorData:
    # dtype is required by the tinker SDK (TensorData.__init__ raises without it)
    return types.TensorData(data=[int(v) for v in values], dtype="int64")


def tensor_data_float(values: list[float]) -> types.TensorData:
    return types.TensorData(data=[float(v) for v in values], dtype="float32")


# ---------------------------------------------------------------------------
# Router replay (R3) support
# ---------------------------------------------------------------------------


def install_routing_matrix_capture() -> None:
    # Patch tinker SDK to pass through routing matrix from server 
    from tinker.lib import api_future_impl
    from tinker.types.sample_response import SampleResponse

    original = api_future_impl.deserialize_json_response

    def patched(result_dict, model_cls):
        result = original(result_dict, model_cls)
        if model_cls is SampleResponse and isinstance(result_dict, dict):
            for seq_obj, seq_dict in zip(result.sequences, result_dict.get("sequences", [])):
                # SampledSequence is a frozen dataclass; attach out-of-band.
                object.__setattr__(seq_obj, "routing_matrix", seq_dict.get("routing_matrix"))
        return result

    api_future_impl.deserialize_json_response = patched


def decode_routing_matrix(b64: str) -> np.ndarray:
    # Decode vllm .npy: payload -> (num_forrwarded_tokens, layers, topk)
    return np.load(io.BytesIO(base64.b64decode(b64)))


# mirror examples/train/gsm8k/gsm8k_dataset.py and examples/tinker/ppo/ppo_client.py 


def extract_ground_truth(solution: str) -> str:
    return solution.split("#### ")[1].replace(",", "").strip()


def compute_reward(response_text: str, ground_truth: str) -> float:
    match = STRICT_ANSWER_RE.search(response_text)
    if match is None:
        return 0.0
    return 1.0 if match.group(1).replace(",", "").strip() == ground_truth else 0.0


def load_gsm8k(tokenizer, max_prompt_length: int, seed: int) -> list[dict]:
    """Load GSM8K train split straight from HF, chat-templated prompts."""
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    records = []
    for row in ds:
        messages = [{"role": "user", "content": f"{row['question']} {INSTRUCTION}"}]
        # return_dict=False: transformers 5.x otherwise returns a BatchEncoding dict
        tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        if len(tokens) > max_prompt_length:
            continue
        records.append(
            {"prompt_tokens": list(tokens), "ground_truth": extract_ground_truth(row["answer"])}
        )
    random.Random(seed).shuffle(records)
    print(f"Loaded {len(records)} GSM8K train examples")
    return records


# ---------------------------------------------------------------------------
# GRPO datum construction
# ---------------------------------------------------------------------------


def build_datum(
    prompt_tokens: list[int],
    response_tokens: list[int],
    sampling_logprobs: list[float],
    advantage: float,
    routing_matrix: np.ndarray | None = None,
) -> types.Datum:

    prefix = prompt_tokens + response_tokens[:-1]
    n = len(response_tokens)
    loss_fn_inputs = {
        "target_tokens": tensor_data_int(response_tokens),
        "weights": tensor_data_float([1.0] * n),
        "logprobs": tensor_data_float(sampling_logprobs),
        "advantages": tensor_data_float([advantage] * n),
    }
    if routing_matrix is not None:
        # vllm covers all forwarded tokens, should be that (prompt + generated - 1) == len(prefix)
        if routing_matrix.shape[0] != len(prefix):
            raise ValueError(
                f"routing_matrix covers {routing_matrix.shape[0]} tokens, expected {len(prefix)} (model_input)"
            )
        loss_fn_inputs["routing_matrix"] = types.TensorData.from_numpy(routing_matrix.astype(np.int64))
    return types.Datum(
        model_input=types.ModelInput.from_ints(prefix),
        loss_fn_inputs=loss_fn_inputs,
    )


def grpo_advantages(rewards: list[float], norm_by_std: bool) -> list[float]:
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if not norm_by_std:
        return centered
    var = sum(c * c for c in centered) / len(centered)
    std = var**0.5
    return [c / (std + 1e-6) for c in centered]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--base-model", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument("--api-key", default=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, help="prompts per step")
    parser.add_argument("--group-size", type=int, default=8, help="samples per prompt (GRPO group)")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-generate-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--loss-fn", default="importance_sampling", choices=["importance_sampling", "ppo"])
    parser.add_argument("--no-norm-by-std", action="store_true", help="GRPO without std normalization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-path", default="./moe_gsm8k_metrics.jsonl")
    parser.add_argument(
        "--enable-router-replay",
        action="store_true",
        help="Echo MoE routing matrices from sample responses into forward_backward datums. "
        "Requires the server to run with enable_return_routed_experts + moe_enable_routing_replay.",
    )
    args = parser.parse_args()

    if args.enable_router_replay:
        install_routing_matrix_capture()

    print(f"Connecting to {args.base_url} (model={args.base_model}, lora_rank={args.lora_rank})")
    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
    )
    tokenizer = training_client.get_tokenizer()
    records = load_gsm8k(tokenizer, args.max_prompt_length, args.seed)

    # The backend applies only the lr; the other Adam fields satisfy the schema.
    adam = types.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.999, eps=1.0e-8)
    loss_fn_config = (
        {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2} if args.loss_fn == "ppo" else None
    )

    metrics_file = open(args.metrics_path, "a", buffering=1)
    print(f"Writing per-step metrics to {args.metrics_path}")


    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "skyrl-tinker-moe"),
                name=os.environ.get("WANDB_RUN_NAME", f"moe-gsm8k-lora-r{args.lora_rank}"),
                config=vars(args),
            )
            print(f"wandb logging enabled: {wandb_run.url}")
        except Exception as e:
            print(f"wandb logging disabled ({e})")
    else:
        print("wandb logging disabled (WANDB_API_KEY not set)")

    for step in range(args.num_steps):
        step_start = time.time()
        batch = [records[(step * args.batch_size + i) % len(records)] for i in range(args.batch_size)]


        sampling_client = training_client.save_weights_and_get_sampling_client()
        futures = []
        for offset, rec in enumerate(batch):
            params = types.SamplingParams(
                max_tokens=args.max_generate_length,
                temperature=args.temperature,
                seed=args.seed + step * 10_000 + offset,
            )
            futures.append(
                sampling_client.sample(
                    prompt=types.ModelInput.from_ints(rec["prompt_tokens"]),
                    num_samples=args.group_size,
                    sampling_params=params,
                )
            )


        data: list[types.Datum] = []
        all_rewards: list[float] = []
        skipped = 0
        missing_routing = 0
        for rec, future in zip(batch, futures):
            result = future.result()
            group = []
            for seq in result.sequences:
                tokens = list(seq.tokens)
                logprobs = list(seq.logprobs or [])
                if not tokens or len(logprobs) != len(tokens):
                    skipped += 1
                    continue
                routing_matrix = None
                if args.enable_router_replay:
                    raw = getattr(seq, "routing_matrix", None)
                    if raw is None:
 
                        missing_routing += 1
                        skipped += 1
                        continue
                    routing_matrix = decode_routing_matrix(raw)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                group.append((tokens, logprobs, routing_matrix, compute_reward(text, rec["ground_truth"])))
            if len(group) < 2:
                skipped += len(group)
                continue
            advantages = grpo_advantages([g[3] for g in group], norm_by_std=not args.no_norm_by_std)
            for (tokens, logprobs, routing_matrix, reward), adv in zip(group, advantages):
                all_rewards.append(reward)
                data.append(build_datum(rec["prompt_tokens"], tokens, logprobs, adv, routing_matrix))

        if args.enable_router_replay and missing_routing:
            print(
                f"step {step}: WARNING {missing_routing} sequences lacked routing_matrix -- "
                "is the server running with generator.inference_engine.enable_return_routed_experts=true?"
            )

        if not data:
            print(f"step {step}: no usable samples (skipped={skipped}); skipping update")
            continue


        fb = training_client.forward_backward(data, args.loss_fn, loss_fn_config).result()
        training_client.optim_step(adam).result()


        server_metrics = dict(getattr(fb, "metrics", {}) or {})
        row = {
            "step": step,
            "reward_mean": sum(all_rewards) / len(all_rewards),
            "num_datums": len(data),
            "skipped": skipped,
            "loss": server_metrics.get("total_loss:sum"),
            "step_time_s": round(time.time() - step_start, 1),
        }
        for key in LOGPROB_DIFF_KEYS:
            short = key.split("/")[-1].split(":")[0]  # e.g. rollout_train_logprobs_abs_diff_mean
            row[short] = server_metrics.get(key)
        metrics_file.write(json.dumps(row) + "\n")
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items() if v is not None}, step=step)

        diff_mean = row.get("rollout_train_logprobs_abs_diff_mean")
        diff_max = row.get("rollout_train_logprobs_abs_diff_max")
        print(
            f"step {step:3d} | reward {row['reward_mean']:.3f} | datums {len(data):3d} | "
            f"logprob_diff mean={diff_mean if diff_mean is not None else 'n/a'} "
            f"max={diff_max if diff_max is not None else 'n/a'} | {row['step_time_s']}s"
        )
        if step == 0 and diff_mean is None:
            print(
                "WARNING: server metrics did not include "
                "policy/rollout_train_logprobs_abs_diff_* — check that datums carry "
                "sampling logprobs and the server is the skyrl_train backend."
            )

    metrics_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    print("run complete")


if __name__ == "__main__":
    main()
