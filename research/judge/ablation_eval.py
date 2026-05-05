"""
ablation_eval.py
================

Offline judge-efficacy evaluation across the ablation matrix defined in
fleet-taste-*.sh scripts, using the 200 Claude trajectories in
fleet-cu-claude-trajectories/claude_trajectories.jsonl.

Measures Spearman correlation and AUC-ROC between judge scores and verifier
scores to determine which ablations (screenshots, relative scoring, Sonnet)
add meaningful signal beyond actions-only Haiku.

Usage:
    cd /private/tmp/skyrl-fleet
    python research/judge/ablation_eval.py
    python research/judge/ablation_eval.py --workers 12 --out results.csv

Requires: ANTHROPIC_API_KEY, scipy, scikit-learn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Make judge importable from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge import (  # noqa: E402
    score_trajectory_haiku,
    score_trajectory_group,
    score_trajectory_group_haiku,
    AXES,
)

TRAJ_FILE = Path(__file__).resolve().parents[2] / "fleet-cu-claude-trajectories" / "claude_trajectories.jsonl"
IMAGES_BASE = Path(__file__).resolve().parents[2] / "fleet-cu-claude-trajectories" / "images"

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _extract_actions(conv: list[dict]) -> list[dict]:
    """Convert conversation messages to a flat action list the judge understands.

    The Claude-trajectory format has no structured tool_use blocks; instead
    the agent narrates in assistant text. We emit:
      {"name": "think", "arguments": {"text": <narration>}}  for assistant text
      {"name": "tool_result", "arguments": {"text": <result>}}  for tool messages
    Tool-error messages are tagged so the recovery axis has something to score.
    """
    actions: list[dict] = []
    for msg in conv:
        role = msg.get("role")
        content = msg.get("content")
        if role == "assistant" and isinstance(content, str) and content.strip():
            actions.append({"name": "think", "arguments": {"text": content.strip()[:600]}})
        elif role == "tool" and isinstance(content, str) and content.strip():
            label = "tool_error" if content.lower().startswith("error") else "tool_result"
            actions.append({"name": label, "arguments": {"text": content.strip()[:300]}})
    return actions


def _extract_reasoning_traces(conv: list[dict]) -> list[str]:
    """Extract per-step reasoning traces from the agent's assistant narration.

    For Claude models there are no separate <think> blocks; the assistant
    messages ARE the reasoning.  We return them as a flat list of strings
    (one per assistant turn) so the judge sees them via the REASONING_TRACES
    section of the system prompt rather than the ACTIONS section.

    This produces a cleaner signal for intent_clarity / coherence because:
    - Tool results (noise for intent axes) are excluded.
    - The judge is explicitly instructed to use REASONING_TRACES for those axes.
    """
    traces: list[str] = []
    for msg in conv:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                traces.append(content.strip()[:600])
    return traces


def _extract_task(conv: list[dict], task_key: str) -> str:
    """First user message is the task description."""
    for msg in conv:
        if msg.get("role") == "user":
            c = msg.get("content")
            if isinstance(c, str) and len(c) > 10:
                return c[:2000]
    return task_key


def _resolve_screenshot_paths(traj: dict) -> Optional[list[str]]:
    """Return absolute paths for the trajectory's screenshots, or None if none exist locally."""
    paths = []
    for rel in traj.get("image_paths", []):
        abs_path = IMAGES_BASE.parent / rel  # rel is like "images/<sid>/pos_NNN.jpeg"
        if abs_path.exists():
            paths.append(str(abs_path))
    return paths if paths else None


def load_trajectories() -> list[dict]:
    trajs = []
    with open(TRAJ_FILE) as f:
        for line in f:
            raw = json.loads(line)
            conv = raw.get("conversation", [])
            shots = _resolve_screenshot_paths(raw)
            trajs.append({
                "session_id": raw["session_id"],
                "task_key": raw["task_key"],
                "task": _extract_task(conv, raw["task_key"]),
                "actions": _extract_actions(conv),
                "reasoning_traces": _extract_reasoning_traces(conv),
                "outcome": raw["score"] >= 0.5,
                "verifier_score": float(raw["score"]),
                "screenshots": shots,
                "has_images": shots is not None,
            })
    return trajs


# ---------------------------------------------------------------------------
# Ablation configurations
# Each entry: (label, requires_images, fn)
# fn(traj) -> Optional[float]  (rescaled to [0,1] here for consistency)
# ---------------------------------------------------------------------------


def _rescale(wt: Optional[float]) -> Optional[float]:
    if wt is None:
        return None
    return max(0.0, min(1.0, (float(wt) - 1.0) / 4.0))


def run_actions_only(traj: dict) -> Optional[dict]:
    """haiku_vision, actions only, no screenshots."""
    res = score_trajectory_haiku(
        task=traj["task"],
        actions=traj["actions"],
        outcome=traj["outcome"],
        screenshots=None,
        model=MODEL_HAIKU,
        blind_outcome=True,
        blind_actions=False,
    )
    return None if res.get("error") else res


def run_screenshots_only(traj: dict) -> Optional[dict]:
    """haiku_vision, blind_actions=True, screenshots required."""
    if not traj["has_images"]:
        return None
    res = score_trajectory_haiku(
        task=traj["task"],
        actions=traj["actions"],
        outcome=traj["outcome"],
        screenshots=traj["screenshots"],
        model=MODEL_HAIKU,
        blind_outcome=True,
        blind_actions=True,
    )
    return None if res.get("error") else res


def run_actions_and_screenshots(traj: dict) -> Optional[dict]:
    """haiku_vision, actions + screenshots (full info ceiling)."""
    if not traj["has_images"]:
        return None
    res = score_trajectory_haiku(
        task=traj["task"],
        actions=traj["actions"],
        outcome=traj["outcome"],
        screenshots=traj["screenshots"],
        model=MODEL_HAIKU,
        blind_outcome=True,
        blind_actions=False,
    )
    return None if res.get("error") else res


def run_screenshots_with_reasoning(traj: dict) -> Optional[dict]:
    """haiku_vision, screenshots + Claude narration passed as reasoning_traces.

    Blind to actions (tool results excluded); the agent's stated intent is
    surfaced via the REASONING_TRACES section of the judge prompt rather than
    the ACTIONS section.  Tests whether framing Claude's narration as explicit
    reasoning (vs mixed action/tool-result noise) improves intent_clarity and
    coherence discrimination.
    """
    if not traj["has_images"]:
        return None
    res = score_trajectory_haiku(
        task=traj["task"],
        actions=traj["actions"],
        outcome=traj["outcome"],
        screenshots=traj["screenshots"],
        model=MODEL_HAIKU,
        blind_outcome=True,
        blind_actions=True,
        reasoning_traces=traj["reasoning_traces"],
    )
    return None if res.get("error") else res


# Maps label -> (requires_images, fn)
CONFIGS: dict[str, tuple[bool, object]] = {
    "actions_only":              (False, run_actions_only),
    "screenshots_only":          (True,  run_screenshots_only),
    "actions_and_screenshots":   (True,  run_actions_and_screenshots),
    "screenshots_with_reasoning":(True,  run_screenshots_with_reasoning),
}


# ---------------------------------------------------------------------------
# Relative judge evaluation (separate: grouped by task)
# ---------------------------------------------------------------------------


def run_relative_haiku_group(task: str, rollouts: list[dict]) -> list[Optional[float]]:
    """Group-relative Haiku judge (actions + screenshots). Returns rescaled weighted_totals."""
    group = [
        {"actions": r["actions"], "outcome": r["outcome"], "screenshots": r.get("screenshots")}
        for r in rollouts
    ]
    res_list = score_trajectory_group_haiku(
        task=task, rollouts=group, model=MODEL_HAIKU, blind_outcome=True,
    )
    return [_rescale(r.get("weighted_total")) if not r.get("error") else None for r in res_list]


def run_relative_sonnet_group(task: str, rollouts: list[dict]) -> list[Optional[float]]:
    """Group-relative Sonnet judge (actions only, text keeps context manageable). Returns rescaled weighted_totals."""
    group = [{"actions": r["actions"], "outcome": r["outcome"]} for r in rollouts]
    res_list = score_trajectory_group(
        task=task, rollouts=group, model=MODEL_SONNET, blind_outcome=True,
    )
    return [_rescale(r.get("weighted_total")) if not r.get("error") else None for r in res_list]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def spearman(x: list[float], y: list[float]) -> float:
    from scipy.stats import spearmanr
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 5:
        return float("nan")
    xs, ys = zip(*pairs)
    r, _ = spearmanr(xs, ys)
    return round(float(r), 4)


def auc_roc(judge_scores: list[Optional[float]], binary_labels: list[bool]) -> float:
    """Trapezoidal AUC-ROC, no sklearn required."""
    pairs = [(s, int(l)) for s, l in zip(judge_scores, binary_labels) if s is not None]
    if len(pairs) < 5:
        return float("nan")
    if len({l for _, l in pairs}) < 2:
        return float("nan")
    pairs_sorted = sorted(pairs, key=lambda x: -x[0])
    n_pos = sum(l for _, l in pairs)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    for _, label in pairs_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        prev_tpr = prev_tp / n_pos
        prev_fpr = prev_fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tp, prev_fp = tp, fp
    return round(auc, 4)


def score_variance(scores: list[Optional[float]]) -> float:
    vals = [s for s in scores if s is not None]
    if len(vals) < 2:
        return float("nan")
    mean = sum(vals) / len(vals)
    return round(sum((v - mean) ** 2 for v in vals) / len(vals), 4)


def none_rate(scores: list[Optional[float]]) -> float:
    return round(sum(1 for s in scores if s is None) / max(len(scores), 1), 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def score_traj_worker(args: tuple) -> tuple:
    label, fn, traj = args
    try:
        result = fn(traj)
    except Exception as e:
        result = None
    return label, traj["session_id"], traj["verifier_score"], traj["outcome"], result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="ablation_results.csv")
    ap.add_argument("--skip-relative", action="store_true",
                    help="Skip relative judge evaluation (expensive)")
    args = ap.parse_args()

    print("Loading trajectories...")
    trajs = load_trajectories()
    total = len(trajs)
    with_images = sum(1 for t in trajs if t["has_images"])
    print(f"  {total} trajectories, {with_images} with local images")
    print()

    # --- Per-trajectory scoring ---
    # Build work items: (label, fn, traj)
    work: list[tuple] = []
    for label, (req_images, fn) in CONFIGS.items():
        subset = trajs if not req_images else [t for t in trajs if t["has_images"]]
        for traj in subset:
            work.append((label, fn, traj))

    print(f"Scoring {len(work)} (label, traj) pairs with {args.workers} workers...")

    # Results: label -> {session_id -> (verifier_score, outcome, result_dict)}
    results: dict[str, dict[str, tuple]] = defaultdict(dict)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(score_traj_worker, item) for item in work]
        done = 0
        for fut in as_completed(futures):
            label, sid, vscore, outcome, result = fut.result()
            results[label][sid] = (vscore, outcome, result)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(work)} done", flush=True)

    print()

    # --- Print summary table ---
    print(f"{'Config':<28} {'n':>5} {'Spearman':>10} {'AUC-ROC':>10} {'Variance':>10} {'None%':>7}")
    print("-" * 70)

    def _jscores_from_results(data: dict) -> list[Optional[float]]:
        return [_rescale(r.get("weighted_total")) if r is not None else None
                for _, _, r in data.values()]

    for label in CONFIGS:
        data = results[label]
        sids = list(data)
        vscores = [data[s][0] for s in sids]
        outcomes = [data[s][1] for s in sids]
        jscores = _jscores_from_results(data)
        n = len([j for j in jscores if j is not None])
        rho = spearman(jscores, vscores)
        auc = auc_roc(jscores, outcomes)
        var = score_variance(jscores)
        nr = none_rate(jscores)
        print(f"{label:<28} {n:>5} {rho:>10.4f} {auc:>10.4f} {var:>10.4f} {nr:>7.1%}")

    # --- Relative judge ---
    if not args.skip_relative:
        print()
        print("Running relative judge on multi-rollout task groups...")
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trajs:
            groups[t["task_key"]].append(t)
        multi_groups = {k: v for k, v in groups.items() if len(v) >= 2}
        print(f"  {len(multi_groups)} tasks with >=2 rollouts")

        rel_configs = [
            ("relative_haiku", run_relative_haiku_group),
            ("relative_sonnet", run_relative_sonnet_group),
        ]
        for rel_label, rel_fn in rel_configs:
            all_jscores: list[Optional[float]] = []
            all_vscores: list[float] = []
            all_outcomes: list[bool] = []
            for task_key, rollouts in multi_groups.items():
                task = rollouts[0]["task"]
                try:
                    group_scores = rel_fn(task, rollouts)
                except Exception as e:
                    print(f"  ERROR {rel_label} {task_key[:30]}: {e}")
                    group_scores = [None] * len(rollouts)
                for r, s in zip(rollouts, group_scores):
                    all_jscores.append(s)
                    all_vscores.append(r["verifier_score"])
                    all_outcomes.append(r["outcome"])
                    # Save per-session relative result for CSV
                    results[rel_label][r["session_id"]] = (r["verifier_score"], r["outcome"], s)

            n = len([j for j in all_jscores if j is not None])
            rho = spearman(all_jscores, all_vscores)
            auc = auc_roc(all_jscores, all_outcomes)
            var = score_variance(all_jscores)
            nr = none_rate(all_jscores)
            print(f"{rel_label:<28} {n:>5} {rho:>10.4f} {auc:>10.4f} {var:>10.4f} {nr:>7.1%}")

    # --- Save raw results (with per-axis scores) ---
    import csv
    from judge import AXES as _AXES
    out_path = Path(args.out)
    rows = []
    fieldnames = ["config", "session_id", "verifier_score", "outcome", "judge_score"] + list(_AXES) + ["rationale"]
    for label, data in results.items():
        for sid, (vscore, outcome, result) in data.items():
            # result may be a full dict (absolute) or a bare rescaled float (relative)
            if isinstance(result, dict):
                judge_score = _rescale(result.get("weighted_total"))
                scores = result.get("scores", {})
                rationale = (result.get("rationale") or "")[:300]
            elif isinstance(result, float):
                judge_score = result  # already rescaled
                scores = {}
                rationale = ""
            else:
                judge_score = None
                scores = {}
                rationale = ""
            row: dict = {
                "config": label,
                "session_id": sid,
                "verifier_score": vscore,
                "outcome": int(outcome),
                "judge_score": judge_score,
            }
            for ax in _AXES:
                row[ax] = scores.get(ax)
            row["rationale"] = rationale
            rows.append(row)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
