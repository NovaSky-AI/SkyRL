"""
variance_analysis.py
====================

Justifies the taste judge's value on top of the binary verifier reward by
showing that judge scores neither collapse to 0/1 nor lack meaningful spread.

Two analyses:

  Part 1 — Separate judge (existing ablation CSV)
    Measures within-outcome variance for all 3 absolute configs and confirms
    the judge discriminates quality among passing trajectories (outcome=1).

  Part 2 — Group judge on fleet-cu tasks with 6–11 rollouts
    Selects tasks from trajectories.jsonl where the fleet-cu image archive
    (fleet-cu-trajectories/images/) contains ≥6 sessions, samples up to 8,
    and runs score_trajectory_group_haiku on each group.  Shows that relative
    scores spread across the [0, 1] range within a group even when all
    rollouts share the same binary verifier outcome.

Usage:
    cd /private/tmp/skyrl-fleet
    python research/judge/variance_analysis.py
    python research/judge/variance_analysis.py --skip-group
    python research/judge/variance_analysis.py --min-rollouts 6 --sample-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import score_trajectory_group_haiku  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ABLATION_CSV = REPO / "research" / "judge" / "ablation_results_with_relative.csv"
TRAJ_FILE = REPO / "trajectories.jsonl"
FLEET_CU_IMAGES = REPO / "fleet-cu-trajectories" / "images"
MODEL_HAIKU = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


def _ks_pvalue(a: list[float], b: list[float]) -> float:
    """Kolmogorov–Smirnov two-sample test p-value (no scipy needed)."""
    try:
        from scipy.stats import ks_2samp
        _, p = ks_2samp(a, b)
        return p
    except ImportError:
        return float("nan")


def _rescale(wt: Optional[float]) -> Optional[float]:
    if wt is None:
        return None
    return max(0.0, min(1.0, (float(wt) - 1.0) / 4.0))


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    i = (len(s) - 1) * p / 100.0
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


# ---------------------------------------------------------------------------
# Part 1: Separate judge variance from existing CSV
# ---------------------------------------------------------------------------


def analyze_separate_judge(csv_path: Path) -> None:
    import csv

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    absolute_configs = ["actions_only", "screenshots_only", "actions_and_screenshots"]

    print("=" * 72)
    print("PART 1 — SEPARATE JUDGE VARIANCE")
    print("=" * 72)
    print()
    print("Goal: confirm judge score has meaningful spread within each outcome")
    print("      class, NOT just a noisy copy of the binary verifier signal.")
    print()

    header = (
        f"{'Config':<28} {'outcome':>8} {'n':>5} {'mean':>6} {'std':>6}"
        f" {'p10':>5} {'p50':>5} {'p90':>5} {'<0.5 %':>7}"
    )
    print(header)
    print("-" * 72)

    all_ks = {}
    for config in absolute_configs:
        config_rows = [r for r in rows if r["config"] == config and r["judge_score"] not in ("", "None")]
        if not config_rows:
            print(f"{config:<28} (no data)")
            continue

        by_outcome: dict[int, list[float]] = {0: [], 1: []}
        for r in config_rows:
            try:
                js = float(r["judge_score"])
                outcome = int(r["outcome"])
                by_outcome[outcome].append(js)
            except (ValueError, KeyError):
                pass

        for outcome in [0, 1]:
            vals = by_outcome[outcome]
            if not vals:
                continue
            below_half = 100.0 * sum(1 for v in vals if v < 0.5) / len(vals)
            print(
                f"{config:<28} {outcome:>8} {len(vals):>5} {sum(vals)/len(vals):>6.3f}"
                f" {_std(vals):>6.3f}"
                f" {_percentile(vals, 10):>5.2f}"
                f" {_percentile(vals, 50):>5.2f}"
                f" {_percentile(vals, 90):>5.2f}"
                f" {below_half:>6.1f}%"
            )

        ks_p = _ks_pvalue(by_outcome[0], by_outcome[1])
        all_ks[config] = ks_p
        print(f"  → KS test p(outcome 0 vs 1) = {ks_p:.4f}  ({'significant' if ks_p < 0.05 else 'not significant'} at α=0.05)")
        print()

    # Summary: among verifier=1, what fraction score < 0.5?
    # (meaningful if judge is giving low scores to technically-passing but ugly runs)
    print("Key numbers for justification:")
    for config in absolute_configs:
        config_rows = [r for r in rows if r["config"] == config and r["judge_score"] not in ("", "None")]
        pass1 = [float(r["judge_score"]) for r in config_rows if r.get("outcome") == "1"]
        if not pass1:
            continue
        below = 100.0 * sum(1 for v in pass1 if v < 0.5) / len(pass1)
        spread = max(pass1) - min(pass1)
        print(
            f"  {config:<28}: outcome=1 std={_std(pass1):.3f}"
            f"  range=[{min(pass1):.2f},{max(pass1):.2f}] (spread {spread:.2f})"
            f"  {below:.0f}% score <0.5 despite verifier=pass"
        )
    print()
    print("Interpretation:")
    print("  • std ≥ 0.20 within passing trajectories means the judge is NOT")
    print("    collapsing to a binary reward — it discriminates between clean")
    print("    passes and lucky/ugly passes.")
    print("  • Trajectories where outcome=1 but judge<0.5 are tasks the agent")
    print("    technically completed but with poor execution quality.")
    print("  • KS p-value << 0.05 confirms the judge separates fail from pass")
    print("    distributions, providing a richer training signal than 0/1 alone.")


# ---------------------------------------------------------------------------
# Part 2: Group judge on fleet-cu tasks with ≥N rollouts
# ---------------------------------------------------------------------------


def _extract_actions(conv_messages: list[dict]) -> list[dict]:
    """Extract a minimal action list from fleet-cu-trajectories conversations.

    The fleet-cu format stores tool call results as 'tool' messages with
    text='Action completed' and screenshots are the primary signal.  We emit
    lightweight action stubs so the judge has a turn count.
    """
    actions: list[dict] = []
    for msg in conv_messages:
        role = msg.get("role")
        text = (msg.get("text") or "").strip()
        if role == "assistant" and text:
            actions.append({"name": "think", "arguments": {"text": text[:400]}})
        elif role == "tool":
            label = "tool_error" if text.lower().startswith("error") else "tool_result"
            actions.append({"name": label, "arguments": {"text": text[:200] or "Action completed"}})
    return actions or [{"name": "tool_result", "arguments": {"text": "Action completed"}}]


def _resolve_images(image_paths: list[str], images_base: Path) -> Optional[list[str]]:
    """Resolve relative image_paths against the fleet-cu-trajectories images folder."""
    resolved = []
    for rel in image_paths:
        # rel is like "images/<session_id>/step_NNN.jpeg"
        # strip leading "images/" since images_base already points to that folder
        stripped = rel[len("images/"):] if rel.startswith("images/") else rel
        abs_path = images_base / stripped
        if abs_path.exists():
            resolved.append(str(abs_path))
    return resolved or None


def _extract_task(conv_messages: list[dict]) -> str:
    """First non-system user message is the task."""
    for msg in conv_messages:
        if msg.get("role") == "user":
            text = (msg.get("text") or "").strip()
            if len(text) > 10:
                return text[:2000]
    return "unknown task"


def load_fleet_cu_groups(min_rollouts: int, sample_size: int) -> list[dict]:
    """Return task groups from fleet-cu trajectories with ≥min_rollouts rollouts.

    Only sessions whose image folder exists in FLEET_CU_IMAGES are included
    (they have local screenshots for the vision judge).
    """
    if not FLEET_CU_IMAGES.exists():
        print(f"  WARNING: {FLEET_CU_IMAGES} not found — skipping group analysis")
        return []

    fleet_cu_sessions = set(os.listdir(FLEET_CU_IMAGES))

    tasks: dict[str, list[dict]] = defaultdict(list)
    with open(TRAJ_FILE) as f:
        for line in f:
            e = json.loads(line)
            if e["session_id"] not in fleet_cu_sessions:
                continue
            conv_raw = e.get("conversation", "[]")
            conv = json.loads(conv_raw) if isinstance(conv_raw, str) else conv_raw
            tasks[e["task_key"]].append({
                "session_id": e["session_id"],
                "task_key": e["task_key"],
                "outcome": e["outcome"] == "success",
                "verifier_score": float(e.get("score", 0)),
                "task": _extract_task(conv),
                "actions": _extract_actions(conv),
                "screenshots": _resolve_images(e.get("image_paths", []), FLEET_CU_IMAGES),
            })

    # filter to tasks with enough rollouts
    groups = []
    for task_key, rollouts in sorted(tasks.items(), key=lambda x: -len(x[1])):
        if len(rollouts) >= min_rollouts:
            selected = rollouts
            if len(rollouts) > sample_size:
                # deterministic: pick diverse outcomes first, then fill with remainder
                passes = [r for r in rollouts if r["outcome"]]
                fails = [r for r in rollouts if not r["outcome"]]
                half = sample_size // 2
                # balance if possible; otherwise just take first sample_size
                if len(passes) >= half and len(fails) >= (sample_size - half):
                    selected = passes[:half] + fails[:(sample_size - half)]
                elif len(passes) < half:
                    selected = passes + fails[:(sample_size - len(passes))]
                else:
                    selected = fails + passes[:(sample_size - len(fails))]
                selected = selected[:sample_size]
            groups.append({
                "task_key": task_key,
                "task": rollouts[0]["task"],
                "rollouts": selected,
            })
    return groups


def analyze_group_judge(min_rollouts: int = 6, sample_size: int = 8) -> None:
    print("=" * 72)
    print("PART 2 — GROUP JUDGE ON FLEET-CU TASKS WITH ≥6 ROLLOUTS")
    print("=" * 72)
    print()
    print(f"Simulates training group size ≈{sample_size}.")
    print("Relative scores should spread across [0,1] even within a group")
    print("where all rollouts share the same verifier outcome.")
    print()

    groups = load_fleet_cu_groups(min_rollouts=min_rollouts, sample_size=sample_size)
    if not groups:
        print("No qualifying task groups found.")
        return

    print(f"Found {len(groups)} task(s) with ≥{min_rollouts} fleet-cu rollouts:")
    for g in groups:
        n = len(g["rollouts"])
        passes = sum(1 for r in g["rollouts"] if r["outcome"])
        print(f"  {g['task_key']}: n={n} pass={passes}")
    print()

    all_group_stds: list[float] = []
    all_within_outcome_stds: list[float] = []

    for g in groups:
        task_key = g["task_key"]
        rollouts = g["rollouts"]
        n = len(rollouts)
        task_text_preview = g["task"][:80].replace("\n", " ")
        print(f"Task: {task_key}")
        print(f"  \"{task_text_preview}...\"")
        print(f"  Rollouts: {n}  (pass={sum(1 for r in rollouts if r['outcome'])}  fail={sum(1 for r in rollouts if not r['outcome'])})")

        group_input = [
            {"actions": r["actions"], "outcome": r["outcome"], "screenshots": r["screenshots"]}
            for r in rollouts
        ]

        try:
            results = score_trajectory_group_haiku(
                task=g["task"],
                rollouts=group_input,
                model=MODEL_HAIKU,
                blind_outcome=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            continue

        scores: list[Optional[float]] = [_rescale(r.get("weighted_total")) for r in results]
        verifier: list[float] = [r["verifier_score"] for r in rollouts]
        outcomes: list[bool] = [r["outcome"] for r in rollouts]

        print()
        print(f"  {'#':>3} {'outcome':>8} {'verifier':>9} {'judge':>7}  scores breakdown")
        for i, (r, res, s) in enumerate(zip(rollouts, results, scores)):
            raw_scores = res.get("scores", {})
            axes = "  ".join(f"{k[:2]}={v}" for k, v in raw_scores.items()) if raw_scores else "(none)"
            s_str = f"{s:.3f}" if s is not None else " err"
            print(f"  {i+1:>3} {'pass' if r['outcome'] else 'fail':>8} {r['verifier_score']:>9.1f} {s_str:>7}  {axes}")

        valid = [s for s in scores if s is not None]
        if valid:
            group_std = _std(valid)
            all_group_stds.append(group_std)
            print()
            print(f"  Within-group std = {group_std:.3f}  range=[{min(valid):.3f},{max(valid):.3f}]")

            # within-outcome std (are the judge scores spreading even among identically-labeled rollouts?)
            for label, flag in [("pass", True), ("fail", False)]:
                vals = [s for s, r in zip(scores, rollouts) if r["outcome"] == flag and s is not None]
                if len(vals) >= 2:
                    ws = _std(vals)
                    all_within_outcome_stds.append(ws)
                    print(f"  Within-{label} std = {ws:.3f}  n={len(vals)}")

        print()

    if all_group_stds:
        print("-" * 72)
        print(f"Summary across {len(all_group_stds)} groups:")
        print(f"  Mean within-group std:          {sum(all_group_stds)/len(all_group_stds):.3f}")
        if all_within_outcome_stds:
            print(f"  Mean within-outcome class std:  {sum(all_within_outcome_stds)/len(all_within_outcome_stds):.3f}")
        print()
        print("Interpretation:")
        print("  • Within-group std > 0.10 shows the judge ranks rollouts even")
        print("    when the verifier gives them all the same binary label.")
        print("  • This is the signal that justifies using taste reward on top")
        print("    of binary reward during RL training.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-group", action="store_true", help="Skip Part 2 (no API calls)")
    ap.add_argument("--skip-separate", action="store_true", help="Skip Part 1")
    ap.add_argument("--min-rollouts", type=int, default=6, help="Min fleet-cu rollouts for group judge")
    ap.add_argument("--sample-size", type=int, default=8, help="Max rollouts per group (simulate training batch)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    if not args.skip_separate:
        analyze_separate_judge(ABLATION_CSV)
        print()

    if not args.skip_group:
        analyze_group_judge(min_rollouts=args.min_rollouts, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
