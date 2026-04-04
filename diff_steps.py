#!/usr/bin/env python3
"""Compare prompt_token_ids between two steps of a trajectory.

Usage:
    python diff_steps.py <trajectory_num> <step_a> <step_b>

Example:
    python diff_steps.py 1 1 2
    python diff_steps.py 1 1 2 --model alphaXiv/rlm-sft-Qwen3.5-9B-v1
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

DEFAULT_MODEL = "alphaXiv/rlm-sft-Qwen3.5-9B-v1"


def load_step(traj: int, step: int) -> dict:
    path = Path("trajectories") / f"trajectory_{traj}" / f"step_{step}.json"
    if not path.exists():
        sys.exit(f"File not found: {path}")
    return json.loads(path.read_text())


def colorize(text: str, color: str) -> str:
    codes = {
        "green": "\033[32m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{codes.get(color, '')}{text}{codes['reset']}"


def main():
    parser = argparse.ArgumentParser(description="Diff prompt_token_ids between two trajectory steps")
    parser.add_argument("trajectory", type=int, help="Trajectory number")
    parser.add_argument("step_a", type=int, help="Earlier step number")
    parser.add_argument("step_b", type=int, help="Later step number")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path for tokenizer")
    parser.add_argument("--context", type=int, default=80, help="Shared-prefix tokens to show at the boundary")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    a = load_step(args.trajectory, args.step_a)
    b = load_step(args.trajectory, args.step_b)

    ids_a = a["prompt_token_ids"]
    ids_b = b["prompt_token_ids"]

    # Find shared prefix length
    prefix_len = 0
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            break
        prefix_len = i + 1

    is_prefix = prefix_len == len(ids_a)
    tail_a = ids_a[prefix_len:]
    tail_b = ids_b[prefix_len:]
    shared = ids_b[:prefix_len]

    print(f"Step {args.step_a} prompt length: {len(ids_a)} tokens")
    print(f"Step {args.step_b} prompt length: {len(ids_b)} tokens")
    print(f"Shared prefix:        {prefix_len} tokens")
    print(f"Step {args.step_a} is exact prefix of step {args.step_b}: "
          f"{colorize(str(is_prefix), 'green' if is_prefix else 'red')}")
    print(f"Tokens only in step {args.step_a} (after prefix): {len(tail_a)}")
    print(f"Tokens only in step {args.step_b} (after prefix): {len(tail_b)}")
    print()

    # --- Shared prefix (show tail end for context) ---
    boundary_start = max(0, prefix_len - args.context)
    boundary_ids = shared[boundary_start:]
    print(f"{'='*80}")
    print(f"SHARED PREFIX — {prefix_len} tokens total (showing last {len(boundary_ids)})")
    print(f"{'='*80}")
    print(colorize(tokenizer.decode(boundary_ids), "dim"))
    print()

    # --- Tail of step A (only if it diverges, i.e. not a clean prefix) ---
    if tail_a:
        print(f"{'='*80}")
        print(colorize(f"STEP {args.step_a} ONLY — {len(tail_a)} tokens after shared prefix", "red"))
        print(f"{'='*80}")
        print(colorize(tokenizer.decode(tail_a), "red"))
        print()

    # --- Tail of step B (new content added between turns) ---
    print(f"{'='*80}")
    print(colorize(f"STEP {args.step_b} ONLY — {len(tail_b)} tokens after shared prefix", "green"))
    print(f"{'='*80}")
    print(colorize(tokenizer.decode(tail_b), "green"))


if __name__ == "__main__":
    main()
