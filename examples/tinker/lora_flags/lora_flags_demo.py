"""Train one LoRA configuration against a running Tinker server and log the loss.

Run it once per configuration (FSDP serves a single adapter per server
lifetime, so each configuration needs its own server). ``run_demo.sh`` does the
full three-way sweep; this script is the inner loop.

Usage:
    python lora_flags_demo.py --train-attn --no-train-mlp --label attn_only
"""

from __future__ import annotations

import argparse
import json
import os

import tinker
from tinker import types as tinker_types

# A fixed, tiny supervised set. Deliberately not a dataset download: the point
# is a deterministic loss curve that differs only by the LoRA configuration.
EXAMPLES = [
    ("Question: What is 2 + 3?\nAnswer:", " 5"),
    ("Question: What is 7 - 4?\nAnswer:", " 3"),
    ("Question: What is 6 * 3?\nAnswer:", " 18"),
    ("Question: What is 20 / 4?\nAnswer:", " 5"),
    ("Question: What is 9 + 8?\nAnswer:", " 17"),
    ("Question: What is 15 - 6?\nAnswer:", " 9"),
    ("Question: What is 4 * 7?\nAnswer:", " 28"),
    ("Question: What is 36 / 6?\nAnswer:", " 6"),
]


def make_datum(tokenizer, prompt: str, completion: str) -> tinker_types.Datum:
    """Build a Datum whose loss covers the completion tokens only."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    target_tokens = all_tokens[1:] + [tokenizer.eos_token_id]
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights[1:] + [1.0]},
    )


def _mean_loss(out) -> float:
    """Mean cross-entropy over the batch's scored tokens.

    The batch is fixed across steps and configurations, so the denominator is
    constant and the curves are directly comparable.
    """
    values = [v for o in out.loss_fn_outputs for v in o["elementwise_loss"].data]
    return float(sum(values)) / float(len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://0.0.0.0:8000/")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--label", required=True, help="Name for this configuration in the output.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="lora_flags_results.json")
    parser.add_argument("--train-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-mlp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    service = tinker.ServiceClient(base_url=args.base_url, api_key=os.environ.get("TINKER_API_KEY", "tml-dummy"))
    client = service.create_lora_training_client(
        base_model=args.base_model,
        rank=args.rank,
        seed=args.seed,
        train_attn=args.train_attn,
        train_mlp=args.train_mlp,
        train_unembed=False,
    )
    tokenizer = client.get_tokenizer()
    batch = [make_datum(tokenizer, prompt, completion) for prompt, completion in EXAMPLES]

    print(f"[{args.label}] train_attn={args.train_attn} train_mlp={args.train_mlp} rank={args.rank}")
    losses = []
    for step in range(args.steps):
        fwd_bwd = client.forward_backward(batch, "cross_entropy").result()
        client.optim_step(tinker_types.AdamParams(learning_rate=args.lr)).result()
        losses.append(_mean_loss(fwd_bwd))
        loss = losses[-1]
        print(f"[{args.label}] step {step:3d}  loss {loss:.4f}", flush=True)

    record = {
        "label": args.label,
        "train_attn": args.train_attn,
        "train_mlp": args.train_mlp,
        "rank": args.rank,
        "seed": args.seed,
        "lr": args.lr,
        "losses": losses,
        "first_loss": losses[0],
        "final_loss": losses[-1],
    }
    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    results = [r for r in results if r["label"] != args.label] + [record]
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{args.label}] first={losses[0]:.4f} final={losses[-1]:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
