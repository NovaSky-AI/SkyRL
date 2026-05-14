"""Trend/health check for an SFT CI run, sourced from wandb.

Replaces the bash stdout-parsing block in ``sft_tulu3_megatron.sh``:
pulls the run's logged ``train/loss`` history from wandb and asserts on it.

Checks performed (any failure exits non-zero):
  * Run exists in the given project (matched by display name; most recent wins).
  * At least ``--min_steps`` ``train/loss`` rows are logged
    (defaults to ``2 * window``, i.e. enough for non-overlapping windows).
  * No NaN/inf in the logged loss history.
  * ``mean(last N losses) < mean(first N losses)`` where N is ``--window``.
  * Optionally: the run's final ``_step`` >= ``--expected_steps`` (skipped if
    ``--expected_steps`` is not provided).

The first 4 checks are CI-critical; the last is opt-in because some callers
don't know the exact step count up front.
"""

import argparse
import math
import sys

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", type=str, required=True, help="wandb run display name")
    parser.add_argument("--project_name", type=str, required=True, help="wandb project name")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="wandb entity. If omitted, project_name is passed as-is "
        "(matching the convention used by get_summary.py).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="train/loss",
        help="History metric to pull (default: train/loss).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Window size N for the first-vs-last mean comparison (default: 5).",
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=None,
        help="Minimum number of logged loss rows required. Defaults to 2 * window.",
    )
    parser.add_argument(
        "--expected_steps",
        type=int,
        default=None,
        help="If set, assert the run's final _step is >= this value (completion check).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    min_steps = args.min_steps if args.min_steps is not None else 2 * args.window
    project_path = f"{args.entity}/{args.project_name}" if args.entity else args.project_name

    api = wandb.Api()
    runs = api.runs(project_path, filters={"display_name": args.run_name}, order="-created_at")
    matched_run = next(iter(runs), None)
    if matched_run is None:
        print(f"FAIL: run '{args.run_name}' not found in project '{project_path}'", file=sys.stderr)
        return 1
    print(f"Matched run: id={matched_run.id} state={matched_run.state} url={matched_run.url}")

    # Pull the full loss history. scan_history streams every logged row (vs.
    # the sampled 500-point default from .history()).
    rows = list(matched_run.scan_history(keys=[args.metric]))
    losses = [row[args.metric] for row in rows if args.metric in row]
    print(f"Pulled {len(losses)} '{args.metric}' rows from wandb history.")

    # ---- Completion check (optional) ----
    if args.expected_steps is not None:
        final_step = matched_run.summary_metrics.get("_step")
        if final_step is None or final_step < args.expected_steps:
            print(
                f"FAIL: run final _step={final_step} < expected_steps={args.expected_steps}",
                file=sys.stderr,
            )
            return 1
        print(f"PASS: run completed (final _step={final_step} >= {args.expected_steps}).")

    # ---- Minimum-rows check ----
    if len(losses) < min_steps:
        print(
            f"FAIL: only {len(losses)} '{args.metric}' rows, need at least {min_steps} "
            f"(2 * window={args.window}) for windowed trend check",
            file=sys.stderr,
        )
        return 1

    # ---- NaN/inf check ----
    bad = [(i, v) for i, v in enumerate(losses) if not math.isfinite(v)]
    if bad:
        print(f"FAIL: non-finite '{args.metric}' values detected: {bad[:5]}", file=sys.stderr)
        return 1
    print(f"PASS: no NaN/inf in '{args.metric}' history.")

    # ---- Windowed trend check ----
    n = args.window
    first_mean = sum(losses[:n]) / n
    last_mean = sum(losses[-n:]) / n
    print(f"Mean of first {n} losses: {first_mean:.6f}; Mean of last {n} losses: {last_mean:.6f}")
    if not (last_mean < first_mean):
        print(
            f"FAIL: mean of last {n} losses ({last_mean:.6f}) is not < " f"mean of first {n} ({first_mean:.6f})",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: loss trend check (mean last {n} < mean first {n}).")

    print("All SFT CI assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
