"""
training_traj_eval.py
=====================

Judge calibration evals on training trajectories from baseline vs experiment
(screenshots-only taste ablation at global_step_5).

Evals implemented:
  1. Rank correlation within groups — Spearman between judge scores and verifier
     rewards, split by group type (all-fail vs mixed).
  2. Score stability — re-run judge on 20 trajectories with cache bypassed;
     measure per-trajectory variance.
  3. False positive quality — judge score distribution for passing trajectories
     (reward > 0) in baseline vs exp.
  4. Within-group advantage variance — std dev of judge scores per group;
     in mixed groups, check whether the highest-judge rollout is a passer.

Usage:
    cd /private/tmp/skyrl-fleet
    python research/judge/training_traj_eval.py
    python research/judge/training_traj_eval.py --workers 12 --stability-n 20

Requires: ANTHROPIC_API_KEY, scipy
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import csv
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge as _judge_module
from judge import score_trajectory_haiku, AXES, _cache_key, _cache_path

BASELINE_DIR = Path(__file__).resolve().parents[2] / "baseline_global_step_5"
EXP_DIR = Path(__file__).resolve().parents[2] / "exp_global_step_5"
OUT_DIR = Path(__file__).resolve().parent

MODEL = "claude-haiku-4-5-20251001"
PASS_THRESHOLD = 0.0  # reward > this => "passed" (any verifier credit)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _remap_image_paths(raw_paths: list[str], local_images_dir: Path) -> list[str]:
    """Remap remote /home/gcpuser/... paths to local images directory (absolute paths)."""
    remapped = []
    for p in raw_paths:
        fname = Path(p).name
        local = local_images_dir.resolve() / fname
        if local.exists():
            remapped.append(str(local))
    return remapped


def _extract_task(prompt: str) -> str:
    """Extract task text from Qwen chat-format prompt."""
    m = re.search(
        r"<\|im_start\|>user\n(.*?)(?:<\|vision_start\|>|<\|im_end\|>)",
        prompt, re.DOTALL,
    )
    if m:
        return m.group(1).strip()[:3000]
    # Fallback: look for ## Task section
    m2 = re.search(r"## Task\n(.*?)(?:\n##|\Z)", prompt, re.DOTALL)
    if m2:
        return m2.group(1).strip()[:3000]
    return prompt[:500]


def _text_to_actions(text: str) -> list[dict]:
    """Wrap trajectory text as a single structured action for the judge."""
    return [{"name": "think", "arguments": {"text": text[:4000]}}]


def load_trajectories(traj_dir: Path, split_label: str) -> list[dict]:
    # File is named by the step suffix, not the full dir name
    # e.g. baseline_global_step_5/global_step_5.jsonl
    step_suffix = "_".join(traj_dir.name.split("_")[-3:])  # global_step_5
    jsonl = traj_dir / f"{step_suffix}.jsonl"
    images_dir = traj_dir / f"{step_suffix}_images"
    records = []
    with open(jsonl) as f:
        for i, line in enumerate(f):
            raw = json.loads(line)
            task = _extract_task(raw["prompt"])
            shots = _remap_image_paths(raw.get("image_paths", []), images_dir)
            reward = float(raw.get("reward", 0.0))
            records.append({
                "split": split_label,
                "traj_idx": i,
                "env_key": raw.get("env_key", "unknown"),
                "reward": reward,
                "passed": reward > PASS_THRESHOLD,
                "task": task,
                "actions": _text_to_actions(raw.get("text", "")),
                "screenshots": shots if shots else None,
            })
    return records


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_one(traj: dict, bypass_cache: bool = False) -> Optional[dict]:
    """Score a single trajectory with the screenshots-only Haiku judge."""
    res = score_trajectory_haiku(
        task=traj["task"],
        actions=traj["actions"],
        outcome=traj["passed"],
        screenshots=traj["screenshots"],
        model=MODEL,
        blind_outcome=True,
        blind_actions=True,  # screenshots-only, matching ablation config
    )
    if bypass_cache:
        # For stability: delete cache entry so next call hits the API again
        key = _cache_key(
            traj["task"], traj["actions"], traj["passed"], MODEL,
            blind_outcome=True, blind_actions=True,
            screenshots_provided=bool(traj["screenshots"]),
        )
        cp = _cache_path(key)
        if cp.exists():
            cp.unlink()
    return None if res.get("error") else res


def _rescale(wt: Optional[float]) -> Optional[float]:
    if wt is None:
        return None
    return max(0.0, min(1.0, (float(wt) - 1.0) / 4.0))


def _score_worker(args: tuple) -> tuple:
    idx, traj, bypass_cache = args
    try:
        res = _score_one(traj, bypass_cache=bypass_cache)
        js = _rescale(res.get("weighted_total")) if res else None
        ax_scores = res.get("scores", {}) if res else {}
        rationale = (res.get("rationale") or "")[:300] if res else ""
    except Exception as e:
        js, ax_scores, rationale = None, {}, f"error: {e}"
    return idx, js, ax_scores, rationale


def score_all(trajs: list[dict], workers: int, label: str = "", bypass_cache: bool = False) -> list[Optional[float]]:
    scores: dict[int, Optional[float]] = {}
    axis_scores: dict[int, dict] = {}
    rationales: dict[int, str] = {}
    work = [(i, t, bypass_cache) for i, t in enumerate(trajs)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_score_worker, item) for item in work]
        done = 0
        for fut in as_completed(futures):
            i, js, axs, rat = fut.result()
            scores[i] = js
            axis_scores[i] = axs
            rationales[i] = rat
            done += 1
            if done % 20 == 0 or done == len(work):
                print(f"  [{label}] {done}/{len(work)} scored", flush=True)
    for t, i in zip(trajs, range(len(trajs))):
        t["judge_score"] = scores[i]
        t["axis_scores"] = axis_scores[i]
        t["rationale"] = rationales[i]
    return [scores[i] for i in range(len(trajs))]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(x: list, y: list) -> float:
    from scipy.stats import spearmanr
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    r, _ = spearmanr(xs, ys)
    return round(float(r), 4)


def variance(vals: list) -> float:
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return float("nan")
    mu = sum(v) / len(v)
    return round(sum((x - mu) ** 2 for x in v) / len(v), 5)


def std_dev(vals: list) -> float:
    var = variance(vals)
    if var != var:  # nan
        return float("nan")
    return round(var ** 0.5, 4)


# ---------------------------------------------------------------------------
# Eval 1: Rank correlation within groups
# ---------------------------------------------------------------------------

def eval_rank_correlation(trajs: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trajs:
        if t["env_key"] != "unknown":
            groups[t["env_key"]].append(t)

    all_corrs, mixed_corrs, allfail_corrs = [], [], []
    group_details = []

    for env_key, group in groups.items():
        if len(group) < 3:
            continue
        js = [t["judge_score"] for t in group]
        rs = [t["reward"] for t in group]
        if all(j is None for j in js):
            continue

        rewards_set = set(rs)
        is_all_fail = rewards_set == {0.0}
        is_all_pass = all(r > PASS_THRESHOLD for r in rs)
        is_mixed = not is_all_fail and not is_all_pass

        rho = spearman(js, rs)
        group_details.append({
            "env_key": env_key,
            "n": len(group),
            "type": "all_fail" if is_all_fail else ("all_pass" if is_all_pass else "mixed"),
            "spearman": rho,
            "judge_std": std_dev(js),
            "reward_std": std_dev(rs),
        })
        if not is_all_fail:  # skip pure-fail groups for correlation (no reward variance)
            all_corrs.append(rho)
        if is_mixed:
            mixed_corrs.append(rho)
        if is_all_fail:
            allfail_corrs.append(rho)

    def _mean(lst):
        valid = [x for x in lst if x == x]  # filter nan
        return round(sum(valid) / len(valid), 4) if valid else float("nan")

    return {
        "group_details": group_details,
        "mean_spearman_all_nonzero": _mean(all_corrs),
        "mean_spearman_mixed": _mean(mixed_corrs),
        "mean_spearman_allfail": _mean(allfail_corrs),
        "n_mixed_groups": len(mixed_corrs),
        "n_allfail_groups": len(allfail_corrs),
    }


# ---------------------------------------------------------------------------
# Eval 2: Score stability
# ---------------------------------------------------------------------------

def eval_stability(trajs: list[dict], n: int, workers: int) -> dict:
    """Re-score n trajectories after cache-busting to measure LLM variance."""
    sample = random.sample([t for t in trajs if t.get("judge_score") is not None], min(n, sum(1 for t in trajs if t.get("judge_score") is not None)))
    print(f"  Stability: re-scoring {len(sample)} trajectories (cache-busted)...")

    run1 = [t["judge_score"] for t in sample]

    # Bust cache and re-score
    for t in sample:
        key = _cache_key(
            t["task"], t["actions"], t["passed"], MODEL,
            blind_outcome=True, blind_actions=True,
            screenshots_provided=bool(t["screenshots"]),
        )
        cp = _cache_path(key)
        if cp.exists():
            cp.unlink()

    run2_scores: dict[int, Optional[float]] = {}
    work = [(i, t, False) for i, t in enumerate(sample)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_score_worker, item) for item in work]
        for fut in as_completed(futures):
            i, js, _, _ = fut.result()
            run2_scores[i] = js
    run2 = [run2_scores[i] for i in range(len(sample))]

    per_traj_var = []
    for s1, s2 in zip(run1, run2):
        if s1 is not None and s2 is not None:
            per_traj_var.append((s1 - s2) ** 2)

    mean_sq_diff = round(sum(per_traj_var) / len(per_traj_var), 5) if per_traj_var else float("nan")
    rmsd = round(mean_sq_diff ** 0.5, 4) if per_traj_var else float("nan")
    rho = spearman(run1, run2)

    pairs = list(zip(run1, run2))
    return {
        "n": len(per_traj_var),
        "rmsd": rmsd,
        "spearman_run1_run2": rho,
        "mean_sq_diff": mean_sq_diff,
        "pairs_sample": [(round(a, 3), round(b, 3)) for a, b in pairs[:10] if a is not None and b is not None],
    }


# ---------------------------------------------------------------------------
# Eval 3: False positive quality distribution
# ---------------------------------------------------------------------------

def eval_fp_quality(baseline: list[dict], exp: list[dict]) -> dict:
    """Compare judge score distributions for passing trajectories."""
    b_pass = [t for t in baseline if t["passed"] and t.get("judge_score") is not None]
    e_pass = [t for t in exp if t["passed"] and t.get("judge_score") is not None]
    b_fail = [t for t in baseline if not t["passed"] and t.get("judge_score") is not None]
    e_fail = [t for t in exp if not t["passed"] and t.get("judge_score") is not None]

    def _stats(lst):
        scores = [t["judge_score"] for t in lst]
        if not scores:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "n": len(scores),
            "mean": round(sum(scores) / len(scores), 4),
            "std": std_dev(scores),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    # FP = passing trajectory with low judge score (judge says it's bad)
    FP_THRESHOLD = 0.4  # judge score < this for a passing traj => suspicious
    b_fp = [t for t in b_pass if t["judge_score"] < FP_THRESHOLD]
    e_fp = [t for t in e_pass if t["judge_score"] < FP_THRESHOLD]

    return {
        "baseline_pass": _stats(b_pass),
        "exp_pass": _stats(e_pass),
        "baseline_fail": _stats(b_fail),
        "exp_fail": _stats(e_fail),
        "baseline_fp_rate": round(len(b_fp) / max(len(b_pass), 1), 3),
        "exp_fp_rate": round(len(e_fp) / max(len(e_pass), 1), 3),
        "fp_threshold": FP_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Eval 4: Within-group advantage variance + taste/binary alignment
# ---------------------------------------------------------------------------

def eval_within_group_variance(baseline: list[dict], exp: list[dict]) -> dict:
    def _analyze(trajs: list[dict], label: str) -> dict:
        groups: dict[str, list[dict]] = defaultdict(list)
        for t in trajs:
            if t["env_key"] != "unknown":
                groups[t["env_key"]].append(t)

        group_stds, mixed_alignments = [], []
        details = []

        for env_key, group in groups.items():
            js = [t["judge_score"] for t in group if t["judge_score"] is not None]
            rs = [t["reward"] for t in group]
            if len(js) < 2:
                continue

            g_std = std_dev([t["judge_score"] for t in group])
            group_stds.append(g_std)

            # Mixed groups: check if highest-judge rollout is a passer
            passers = [t for t in group if t["passed"]]
            is_mixed = passers and len(passers) < len(group)

            taste_top_is_passer = None
            if is_mixed:
                scored = [(t["judge_score"], t["passed"]) for t in group if t["judge_score"] is not None]
                if scored:
                    top_score, top_passed = max(scored, key=lambda x: x[0])
                    taste_top_is_passer = top_passed
                    mixed_alignments.append(int(top_passed))

            details.append({
                "env_key": env_key,
                "n": len(group),
                "judge_std": g_std,
                "n_passers": len(passers),
                "taste_top_is_passer": taste_top_is_passer,
            })

        mean_std = round(sum(group_stds) / len(group_stds), 4) if group_stds else float("nan")
        taste_alignment = round(sum(mixed_alignments) / len(mixed_alignments), 3) if mixed_alignments else float("nan")

        return {
            "label": label,
            "n_groups": len(group_stds),
            "mean_within_group_judge_std": mean_std,
            "n_mixed_groups": len(mixed_alignments),
            "taste_top_is_passer_rate": taste_alignment,
            "details": details,
        }

    return {
        "baseline": _analyze(baseline, "baseline"),
        "exp": _analyze(exp, "exp"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stability-n", type=int, default=20,
                    help="Number of trajectories to re-score for stability eval")
    ap.add_argument("--skip-stability", action="store_true")
    ap.add_argument("--out-csv", type=str, default="training_traj_eval.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading trajectories...")
    baseline = load_trajectories(BASELINE_DIR, "baseline")
    exp = load_trajectories(EXP_DIR, "exp")
    print(f"  Baseline: {len(baseline)} trajs, {sum(t['passed'] for t in baseline)} passed")
    print(f"  Exp:      {len(exp)} trajs, {sum(t['passed'] for t in exp)} passed")
    print(f"  Baseline with screenshots: {sum(1 for t in baseline if t['screenshots'])}")
    print(f"  Exp with screenshots:      {sum(1 for t in exp if t['screenshots'])}")
    print()

    all_trajs = baseline + exp

    print("Scoring all trajectories (judge: screenshots-only Haiku, blind_outcome=True)...")
    score_all(all_trajs, workers=args.workers, label="all")
    scored = sum(1 for t in all_trajs if t.get("judge_score") is not None)
    print(f"  Scored {scored}/{len(all_trajs)} successfully")
    print()

    # ---- Eval 1: Rank correlation ----
    print("=" * 60)
    print("EVAL 1: Rank correlation within groups")
    print("=" * 60)
    b_corr = eval_rank_correlation(baseline)
    e_corr = eval_rank_correlation(exp)

    for label, result in [("BASELINE", b_corr), ("EXP", e_corr)]:
        print(f"\n  {label}:")
        print(f"    Mixed groups:   n={result['n_mixed_groups']}, mean Spearman = {result['mean_spearman_mixed']:.4f}")
        print(f"    All-fail groups: n={result['n_allfail_groups']}, mean Spearman = {result['mean_spearman_allfail']:.4f}")
        print(f"    All non-zero:   mean Spearman = {result['mean_spearman_all_nonzero']:.4f}")
        print()
        print(f"    {'env_key':<16} {'n':>4} {'type':<10} {'Spearman':>10} {'judge_std':>10} {'reward_std':>10}")
        print(f"    " + "-" * 60)
        for g in sorted(b_corr["group_details"] if label == "BASELINE" else e_corr["group_details"],
                         key=lambda x: x["spearman"] if x["spearman"] == x["spearman"] else -9):
            print(f"    {g['env_key']:<16} {g['n']:>4} {g['type']:<10} {g['spearman']:>10.4f} {g['judge_std']:>10.4f} {g['reward_std']:>10.4f}")

    # ---- Eval 3: FP quality (before stability since no cache busting) ----
    print()
    print("=" * 60)
    print("EVAL 3: False positive quality distribution")
    print("=" * 60)
    fp_result = eval_fp_quality(baseline, exp)

    for cat, key in [("PASSING trajs", "pass"), ("FAILING trajs", "fail")]:
        b = fp_result[f"baseline_{key}"]
        e = fp_result[f"exp_{key}"]
        print(f"\n  {cat}:")
        print(f"    {'':12} {'n':>5} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        print(f"    {'baseline':<12} {b['n']:>5} {b['mean']:>8.4f} {b['std']:>8.4f} {b['min']:>8.4f} {b['max']:>8.4f}")
        print(f"    {'exp':<12} {e['n']:>5} {e['mean']:>8.4f} {e['std']:>8.4f} {e['min']:>8.4f} {e['max']:>8.4f}")

    print(f"\n  FP rate (passing traj with judge_score < {fp_result['fp_threshold']}):")
    print(f"    Baseline: {fp_result['baseline_fp_rate']:.1%} ({int(fp_result['baseline_fp_rate'] * fp_result['baseline_pass']['n'])} / {fp_result['baseline_pass']['n']})")
    print(f"    Exp:      {fp_result['exp_fp_rate']:.1%} ({int(fp_result['exp_fp_rate'] * fp_result['exp_pass']['n'])} / {fp_result['exp_pass']['n']})")

    # ---- Eval 4: Within-group advantage variance ----
    print()
    print("=" * 60)
    print("EVAL 4: Within-group advantage variance")
    print("=" * 60)
    wg_result = eval_within_group_variance(baseline, exp)

    for split_key in ["baseline", "exp"]:
        r = wg_result[split_key]
        print(f"\n  {split_key.upper()}:")
        print(f"    Groups analyzed:         {r['n_groups']}")
        print(f"    Mean within-group std:   {r['mean_within_group_judge_std']:.4f}")
        print(f"    Mixed groups:            {r['n_mixed_groups']}")
        print(f"    Taste top = passer rate: {r['taste_top_is_passer_rate']:.1%}" if r['taste_top_is_passer_rate'] == r['taste_top_is_passer_rate'] else "    Taste top = passer rate: n/a")
        print()
        print(f"    {'env_key':<16} {'n':>4} {'judge_std':>10} {'n_pass':>7} {'top_is_pass':>12}")
        print(f"    " + "-" * 54)
        for g in sorted(r["details"], key=lambda x: -(x["judge_std"] if x["judge_std"] == x["judge_std"] else 0)):
            tip = "yes" if g["taste_top_is_passer"] is True else ("no" if g["taste_top_is_passer"] is False else "-")
            print(f"    {g['env_key']:<16} {g['n']:>4} {g['judge_std']:>10.4f} {g['n_passers']:>7} {tip:>12}")

    # ---- Eval 2: Score stability ----
    if not args.skip_stability:
        print()
        print("=" * 60)
        print("EVAL 2: Score stability (cache-busted re-run)")
        print("=" * 60)
        stab = eval_stability(baseline + exp, n=args.stability_n, workers=args.workers)
        print(f"\n  n trajectories re-scored:  {stab['n']}")
        print(f"  RMSD (run1 vs run2):       {stab['rmsd']:.4f}")
        print(f"  Spearman(run1, run2):      {stab['spearman_run1_run2']:.4f}")
        print(f"  Mean squared diff:         {stab['mean_sq_diff']:.5f}")
        print(f"  Sample pairs (run1, run2): {stab['pairs_sample']}")

    # ---- Save CSV ----
    out_path = OUT_DIR / args.out_csv
    fieldnames = ["split", "traj_idx", "env_key", "reward", "passed", "judge_score"] + list(AXES) + ["rationale"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for t in all_trajs:
            row = {
                "split": t["split"],
                "traj_idx": t["traj_idx"],
                "env_key": t["env_key"],
                "reward": t["reward"],
                "passed": int(t["passed"]),
                "judge_score": t.get("judge_score"),
            }
            for ax in AXES:
                row[ax] = t.get("axis_scores", {}).get(ax)
            row["rationale"] = t.get("rationale", "")
            w.writerow(row)
    print(f"\nPer-trajectory results saved to {out_path}")


if __name__ == "__main__":
    main()
