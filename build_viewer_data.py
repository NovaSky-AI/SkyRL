#!/usr/bin/env python3.13
"""
Build viewer_data.json for the trajectory comparison viewer.

Reads:
  - baseline_global_step_5/global_step_5.jsonl
  - exp_global_step_5/global_step_5.jsonl
  - research/judge/training_traj_eval.csv

Writes:
  - viewer_data.json
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
BASELINE_JSONL = ROOT / "baseline_global_step_5" / "global_step_5.jsonl"
EXP_JSONL = ROOT / "exp_global_step_5" / "global_step_5.jsonl"
JUDGE_CSV = ROOT / "research" / "judge" / "training_traj_eval.csv"
OUTPUT = ROOT / "viewer_data.json"

META = {
    "baseline_run": "zd3sk2db",
    "exp_run": "gjfocn7r",
    "step": 5,
}

# ---------------------------------------------------------------------------
# Image path translation
# ---------------------------------------------------------------------------
_IMG_FILENAME_RE = re.compile(r"(traj_\d+_img_\d+\.jpg)$")


def translate_image_path(remote_path: str, split: str) -> str:
    """Convert remote image path to local relative path for the HTTP server."""
    m = _IMG_FILENAME_RE.search(remote_path)
    if not m:
        return remote_path
    filename = m.group(1)
    if split == "baseline":
        return f"baseline_global_step_5/global_step_5_images/{filename}"
    else:
        return f"exp_global_step_5/global_step_5_images/{filename}"


# ---------------------------------------------------------------------------
# Load judge CSV  →  {(split, traj_idx): row_dict}
# ---------------------------------------------------------------------------
def load_judge_csv(path: Path) -> dict:
    index: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"].strip()
            traj_idx = int(row["traj_idx"])
            index[(split, traj_idx)] = row
    return index


def maybe_float(value: str) -> float | None:
    """Parse a float string, returning None for empty/invalid values."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Load JSONL rows
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Build rollout records
# ---------------------------------------------------------------------------
def build_rollouts(
    split: str,
    rows: list[dict],
    judge_index: dict,
    id_offset: int,
) -> list[dict]:
    rollouts = []
    for traj_idx, row in enumerate(rows):
        env_key = row.get("env_key", "unknown")
        reward = float(row.get("reward", 0.0))

        # Translate image paths
        raw_paths = row.get("image_paths") or []
        images = [translate_image_path(p, split) for p in raw_paths]

        # Look up judge data
        judge_key = (split, traj_idx)
        judge_row = judge_index.get(judge_key)
        judge_score = None
        judge_rationale = None
        if judge_row:
            judge_score = maybe_float(judge_row.get("judge_score", ""))
            rationale = judge_row.get("rationale", "").strip()
            judge_rationale = rationale if rationale else None

        rollout = {
            "id": id_offset + traj_idx,
            "split": split,
            "traj_idx": traj_idx,
            "env_key": env_key,
            "reward": reward,
            "passed": reward > 0,
            "turns": int(row.get("turns", 0)),
            "tokens": int(row.get("tokens", 0)),
            "stop_reason": row.get("stop_reason", ""),
            "judge_score": judge_score,
            "judge_rationale": judge_rationale,
            "images": images,
            "text": row.get("text", ""),
        }
        rollouts.append(rollout)
    return rollouts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading judge CSV …")
    judge_index = load_judge_csv(JUDGE_CSV)
    print(f"  {len(judge_index)} judge entries loaded")

    print("Loading baseline JSONL …")
    baseline_rows = load_jsonl(BASELINE_JSONL)
    print(f"  {len(baseline_rows)} rows")

    print("Loading exp JSONL …")
    exp_rows = load_jsonl(EXP_JSONL)
    print(f"  {len(exp_rows)} rows")

    print("Building rollout records …")
    baseline_rollouts = build_rollouts("baseline", baseline_rows, judge_index, id_offset=0)
    exp_rollouts = build_rollouts("exp", exp_rows, judge_index, id_offset=len(baseline_rows))

    all_rollouts = baseline_rollouts + exp_rollouts

    # Collect all env_keys, excluding "unknown", sorted alphabetically
    envs = sorted(
        {r["env_key"] for r in all_rollouts if r["env_key"] != "unknown"}
    )

    output = {
        "meta": META,
        "envs": envs,
        "rollouts": all_rollouts,
    }

    print(f"Writing {OUTPUT} …")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))
    print("Done.")

    # Summary
    b_pass = sum(1 for r in baseline_rollouts if r["passed"])
    e_pass = sum(1 for r in exp_rollouts if r["passed"])
    b_judge = sum(1 for r in baseline_rollouts if r["judge_score"] is not None)
    e_judge = sum(1 for r in exp_rollouts if r["judge_score"] is not None)

    print()
    print("=== Summary ===")
    print(f"  baseline : {len(baseline_rollouts)} rollouts, {b_pass} passing, {b_judge} with judge scores")
    print(f"  exp      : {len(exp_rollouts)} rollouts, {e_pass} passing, {e_judge} with judge scores")
    print(f"  envs     : {len(envs)} ({', '.join(envs)})")
    file_size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"  output   : {OUTPUT} ({file_size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
