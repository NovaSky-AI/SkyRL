#!/usr/bin/env python3
"""
wandb_run_monitor.py — Kill faulty WandB training runs.

Fault conditions detected:
  GHOST  Run has been alive for --grace-minutes but no reward/timing metrics
         have appeared — training loop is silent (hung in setup, vLLM crash,
         rollout deadlock, etc.).  Only system/* metrics visible.
  STALL  A training step was logged at least once, but the step counter has
         not advanced for --stall-minutes — training is frozen mid-run.
  DEAD   WandB itself reports the run as crashed/failed.

On fault the script:
  1. Sends a wandb.alert() (triggers email if WandB notifications are configured).
  2. Annotates run.summary with fault_reason / fault_time.
  3. Runs `sky down <name> -y` if --sky-cluster is supplied.
  4. Exits with code 1.

Usage:
  # Find run by full path (entity/project/run_id):
  python scripts/wandb_run_monitor.py --run-path fleet-ai/fleet-browser-use-grpo/abc123

  # Find run by project + display name (waits up to --find-timeout-min for it to appear):
  python scripts/wandb_run_monitor.py \\
    --project fleet-browser-use-grpo \\
    --run-name fleet_qwen35_browser_use_add-judge \\
    [--entity fleet-ai] \\
    [--grace-minutes 60] \\
    [--stall-minutes 90] \\
    [--poll-interval 120] \\
    [--sky-cluster taste-add]
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone

import wandb

# Keys we expect to see once the training loop has completed at least one step.
# WandB automatically logs system/* even if training hangs, so absence of
# these after grace period is the "ghost run" signal.
TRAINING_METRIC_PREFIXES = ("reward/", "timing/", "time/", "throughput/", "advantage/")


def _has_training_metrics(summary_keys: set[str]) -> bool:
    return any(k.startswith(p) for k in summary_keys for p in TRAINING_METRIC_PREFIXES)


def _get_step(summary: dict) -> int | None:
    for key in ("trainer/global_step", "global_step", "step"):
        v = summary.get(key)
        if v is not None:
            return int(v)
    return None


def _fault(run, reason: str, sky_cluster: str | None) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"\n[MONITOR] FAULT DETECTED — {reason}", flush=True)
    print(f"[MONITOR] Run: {run.url}", flush=True)

    # wandb.alert() requires an active wandb.init() run; initialize briefly just to send.
    try:
        wandb.init(
            project=run.project,
            id=run.id,
            resume="allow",
            reinit=True,
            settings=wandb.Settings(silent=True),
        )
        wandb.alert(
            title=f"[skyrl-monitor] Faulty run: {run.name}",
            text=f"{reason}\n\nRun: {run.url}\nDetected at: {ts}",
            level=wandb.AlertLevel.ERROR,
        )
        wandb.finish(quiet=True)
        print(f"[MONITOR] wandb.alert() sent.", flush=True)
    except Exception as e:
        print(f"[MONITOR] wandb.alert() failed: {e}", flush=True)

    try:
        run.summary.update({"fault_reason": reason, "fault_time": ts})
    except Exception as e:
        print(f"[MONITOR] Could not annotate run summary: {e}", flush=True)

    if sky_cluster:
        print(f"[MONITOR] Running: sky down {sky_cluster} -y", flush=True)
        try:
            subprocess.run(["sky", "down", sky_cluster, "-y"], check=True)
            print(f"[MONITOR] sky down succeeded.", flush=True)
        except Exception as e:
            print(f"[MONITOR] sky down failed: {e}", flush=True)
            print(f"[MONITOR] Kill manually:  sky down {sky_cluster} -y", flush=True)
    else:
        print("[MONITOR] No --sky-cluster supplied. Kill the cluster manually.", flush=True)

    sys.exit(1)


def _find_run(api: wandb.Api, entity: str | None, project: str, run_name: str,
              find_timeout_min: int, poll_interval: int) -> wandb.apis.public.Run:
    deadline = time.monotonic() + find_timeout_min * 60
    path = f"{entity}/{project}" if entity else project
    print(f"[MONITOR] Searching for run '{run_name}' in {path} ...", flush=True)
    while True:
        runs = api.runs(path=path, filters={"displayName": run_name}, per_page=5)
        for r in runs:
            if r.name == run_name:
                print(f"[MONITOR] Found run {r.id} (state={r.state})", flush=True)
                return r
        if time.monotonic() > deadline:
            print(
                f"[MONITOR] Run '{run_name}' not found in {path} after {find_timeout_min} min.",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            f"[MONITOR] Run not found yet, retrying in {poll_interval}s ...", flush=True
        )
        time.sleep(poll_interval)


def monitor(run: wandb.apis.public.Run, grace_minutes: int, stall_minutes: int,
            poll_interval: int, sky_cluster: str | None) -> None:
    grace_s = grace_minutes * 60
    stall_s = stall_minutes * 60

    start_time = time.monotonic()
    last_step: int | None = None
    last_step_time = time.monotonic()
    first_step_seen = False

    print(
        f"[MONITOR] Watching run '{run.name}' (id={run.id})",
        flush=True,
    )
    print(
        f"[MONITOR] Grace={grace_minutes}m  Stall={stall_minutes}m  Poll={poll_interval}s",
        flush=True,
    )
    print(f"[MONITOR] Run URL: {run.url}", flush=True)

    while True:
        time.sleep(poll_interval)

        run.load(force=True)  # refresh from API
        state = run.state
        elapsed = time.monotonic() - start_time

        # --- DEAD: WandB itself reports failure ---
        if state in ("crashed", "failed"):
            _fault(run, f"WandB reports run state='{state}'", sky_cluster)

        summary = dict(run.summary)
        summary_keys = set(summary.keys())
        current_step = _get_step(summary)

        # --- GHOST: No training metrics after grace period ---
        if not first_step_seen and elapsed >= grace_s:
            if not _has_training_metrics(summary_keys):
                visible = sorted(
                    k for k in summary_keys if not k.startswith("_")
                )
                _fault(
                    run,
                    f"No reward/timing metrics after {grace_minutes} min "
                    f"(only system metrics visible). Keys seen: {visible[:20]}",
                    sky_cluster,
                )

        # --- Track step advancement ---
        if current_step is not None:
            if not first_step_seen:
                print(
                    f"[MONITOR] First training step logged: step={current_step} "
                    f"(elapsed {elapsed/60:.1f}m)",
                    flush=True,
                )
                first_step_seen = True

            if last_step is None or current_step > last_step:
                if last_step is not None:
                    print(
                        f"[MONITOR] Step advanced {last_step} → {current_step} "
                        f"(elapsed {elapsed/60:.1f}m)",
                        flush=True,
                    )
                last_step = current_step
                last_step_time = time.monotonic()
            else:
                stall_elapsed = time.monotonic() - last_step_time
                print(
                    f"[MONITOR] Step={current_step} unchanged for "
                    f"{stall_elapsed/60:.1f}m (limit={stall_minutes}m, "
                    f"state={state})",
                    flush=True,
                )
                if stall_elapsed >= stall_s:
                    _fault(
                        run,
                        f"Step stalled at {current_step} for {stall_minutes} min",
                        sky_cluster,
                    )
        else:
            print(
                f"[MONITOR] No step logged yet (elapsed {elapsed/60:.1f}m, "
                f"state={state}, keys={len(summary_keys)})",
                flush=True,
            )

        # Run finished cleanly
        if state in ("finished",):
            print(f"[MONITOR] Run finished cleanly. Exiting.", flush=True)
            sys.exit(0)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor a WandB run and kill if faulty.")
    run_group = p.add_mutually_exclusive_group(required=True)
    run_group.add_argument(
        "--run-path",
        metavar="ENTITY/PROJECT/RUN_ID",
        help="Full WandB run path.",
    )
    run_group.add_argument(
        "--run-name",
        metavar="NAME",
        help="WandB display name (requires --project).",
    )
    p.add_argument("--project", metavar="PROJECT")
    p.add_argument("--entity", metavar="ENTITY", default=None)
    p.add_argument(
        "--grace-minutes",
        type=int,
        default=60,
        help="Minutes to wait for first reward/timing metric before declaring GHOST (default 60).",
    )
    p.add_argument(
        "--stall-minutes",
        type=int,
        default=90,
        help="Minutes with no step advance before declaring STALL (default 90).",
    )
    p.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Seconds between WandB API polls (default 120).",
    )
    p.add_argument(
        "--find-timeout-min",
        type=int,
        default=30,
        help="Minutes to wait for the run to appear in WandB before giving up (default 30).",
    )
    p.add_argument(
        "--sky-cluster",
        metavar="CLUSTER",
        default=None,
        help="SkyPilot cluster name to `sky down` on fault (e.g. taste-add).",
    )
    args = p.parse_args()

    if args.run_name and not args.project:
        p.error("--run-name requires --project")

    api = wandb.Api()

    if args.run_path:
        run = api.run(args.run_path)
    else:
        run = _find_run(
            api,
            args.entity,
            args.project,
            args.run_name,
            args.find_timeout_min,
            args.poll_interval,
        )

    monitor(
        run,
        grace_minutes=args.grace_minutes,
        stall_minutes=args.stall_minutes,
        poll_interval=args.poll_interval,
        sky_cluster=args.sky_cluster,
    )


if __name__ == "__main__":
    main()
