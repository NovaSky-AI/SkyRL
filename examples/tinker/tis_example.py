"""RL training loop with importance sampling using the Tinker API.

Demonstrates GRPO-style RL on GSM8K math problems, using prompt_logprobs to get
full-sequence logprobs under the sampling policy. This replaces the typical
zero-padding of prompt logprobs with real values from the model.

Based on the tinker-cookbook rl_loop recipe:
https://github.com/thinking-machines-lab/tinker-cookbook

Usage:
    # Terminal 1: Start Tinker server
    uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
        --base-model "Qwen/Qwen3-0.6B" --backend fsdp

    # Terminal 2: Run this example
    TINKER_API_KEY=tml-dummy uv run --with tinker --with datasets --with torch \
        python examples/tinker/tis_example.py \
        --base-url http://localhost:8000 \
        --model-name "Qwen/Qwen3-0.6B"

    # Or run on Modal (from examples/train_integrations/modal/):
    MODAL_GPU=L4:1 modal run main.py --command "bash -c '
        uv run --extra tinker --extra fsdp -m skyrl.tinker.api \\
            --base-model Qwen/Qwen3-0.6B --backend fsdp &
        sleep 60 &&
        TINKER_API_KEY=tml-dummy python examples/tinker/tis_example.py \\
            --base-url http://localhost:8000 \\
            --model-name Qwen/Qwen3-0.6B \\
            --wandb-project skyrl-tis-example \\
            --num-steps 10
    '"
"""

import argparse
import re
import time

import tinker
import torch
from datasets import load_dataset
from tinker import types
from tinker.types.tensor_data import TensorData

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str | None:
    """Extract the numeric answer after #### in model output."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def score_completion(completion_text: str, ground_truth: str) -> float:
    """Return 1.0 if the extracted answer matches the ground truth, else 0.0."""
    predicted = extract_answer(completion_text)
    if predicted is None:
        return 0.0
    expected = ground_truth.strip().replace(",", "")
    return 1.0 if predicted == expected else 0.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="RL training with importance sampling via Tinker API")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--api-key", type=str, default="tml-dummy")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of prompts per step")
    parser.add_argument("--num-samples", type=int, default=4, help="Samples per prompt (GRPO group size)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--wandb-project", type=str, default=None, help="Wandb project name (None = disabled)")
    args = parser.parse_args()

    # --- Optional wandb setup ---
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, config=vars(args))

    # --- Connect to Tinker server ---
    service_client = tinker.ServiceClient(base_url=args.base_url, api_key=args.api_key)

    training_client = service_client.create_lora_training_client(
        base_model=args.model_name,
        rank=args.lora_rank,
    )
    tokenizer = training_client.get_tokenizer()

    # --- Load GSM8K ---
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    prompts_and_answers = [
        (row["question"], row["answer"].split("####")[-1].strip())
        for row in dataset
    ]

    adam_params = types.AdamParams(learning_rate=args.learning_rate)
    sampling_params = types.SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompt_idx = 0

    for step in range(args.num_steps):
        step_start = time.time()
        metrics: dict[str, float] = {}

        # --- Sync weights to inference engine ---
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step:04d}"
        )

        # --- Build prompts for this batch ---
        batch_questions = []
        batch_answers = []
        for _ in range(args.batch_size):
            question, answer = prompts_and_answers[prompt_idx % len(prompts_and_answers)]
            batch_questions.append(question)
            batch_answers.append(answer)
            prompt_idx += 1

        # Tokenize and submit sampling requests (with prompt_logprobs)
        prompts_P: list[types.ModelInput] = []
        futures_P = []
        for question in batch_questions:
            prompt_ids = tokenizer.encode(question, add_special_tokens=True)
            prompt_input = types.ModelInput.from_ints(tokens=prompt_ids)
            prompts_P.append(prompt_input)

            future = sampling_client.sample(
                prompt=prompt_input,
                sampling_params=sampling_params,
                num_samples=args.num_samples,
                include_prompt_logprobs=True,
            )
            futures_P.append(future)

        # --- Collect results, score, compute advantages, build datums ---
        datums_D: list[types.Datum] = []
        rewards_P: list[float] = []
        all_prompt_logprobs_flat: list[float] = []  # for metrics

        for future, prompt, answer in zip(futures_P, prompts_P, batch_answers):
            sample_result = future.result()
            ob_len = prompt.length - 1

            # --- Verify prompt_logprobs plumbing ---
            prompt_lp = sample_result.prompt_logprobs
            assert prompt_lp is not None, (
                "prompt_logprobs is None — the backend did not return prompt logprobs. "
                "Ensure the server is running with --backend fsdp or jax."
            )
            assert len(prompt_lp) == ob_len, (
                f"prompt_logprobs length mismatch: got {len(prompt_lp)}, expected {ob_len}"
            )
            all_prompt_logprobs_flat.extend(prompt_lp)

            # Score each sample in the group
            rewards_G: list[float] = []
            sampled_tokens_G: list[list[int]] = []
            logprobs_G: list[list[float]] = []

            for seq in sample_result.sequences:
                sampled_tokens_G.append(seq.tokens)
                logprobs_G.append(seq.logprobs)

                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                rewards_G.append(score_completion(text, answer))

            # GRPO: advantage = reward - mean(group)
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            # Skip if all advantages are zero (no learning signal)
            if all(a == 0.0 for a in advantages_G):
                continue

            # Build one Datum per sample
            for sampled_tokens, response_logprobs, advantage in zip(
                sampled_tokens_G, logprobs_G, advantages_G
            ):
                model_input = prompt.append(
                    types.EncodedTextChunk(tokens=sampled_tokens[:-1])
                )
                target_tokens = [0] * ob_len + sampled_tokens

                # KEY DIFFERENCE from rl_loop: use real prompt_logprobs instead of zeros.
                # rl_loop does: padded_logprobs = [0.0] * ob_len + logprobs
                # We do:        padded_logprobs = prompt_lp + logprobs
                padded_logprobs = list(prompt_lp) + list(response_logprobs)

                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                assert model_input.length == len(target_tokens) == len(padded_logprobs) == len(padded_advantages), (
                    f"Length mismatch: model_input={model_input.length}, targets={len(target_tokens)}, "
                    f"logprobs={len(padded_logprobs)}, advantages={len(padded_advantages)}"
                )

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        metrics["time/sample"] = time.time() - step_start

        # --- Prompt logprobs summary ---
        if all_prompt_logprobs_flat:
            metrics["prompt_logprobs/mean"] = sum(all_prompt_logprobs_flat) / len(all_prompt_logprobs_flat)
            metrics["prompt_logprobs/min"] = min(all_prompt_logprobs_flat)
            metrics["prompt_logprobs/max"] = max(all_prompt_logprobs_flat)

        if step == 0 and all_prompt_logprobs_flat:
            print(
                f"  prompt_logprobs OK — "
                f"mean={metrics['prompt_logprobs/mean']:.2f}, "
                f"min={metrics['prompt_logprobs/min']:.2f}, "
                f"max={metrics['prompt_logprobs/max']:.2f}"
            )

        # --- Train with importance sampling ---
        train_start = time.time()
        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        if fwd_bwd_result.metrics:
            metrics.update(fwd_bwd_result.metrics)
        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        metrics["time/train"] = time.time() - train_start
        metrics["time/total"] = time.time() - step_start
        metrics["reward/mean"] = sum(rewards_P) / len(rewards_P) if rewards_P else 0.0

        print(
            f"Step {step:3d} | "
            f"reward={metrics['reward/mean']:.3f} | "
            f"grad_norm={metrics.get('skyrl.ai/grad_norm', 0):.4f} | "
            f"time={metrics['time/total']:.1f}s"
        )

        if use_wandb:
            wandb.log(metrics, step=step)

    print("Training complete.")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
