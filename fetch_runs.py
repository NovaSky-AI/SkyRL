#!/usr/bin/env python3
"""
Pull val/generations tables and trajectory artifacts from WandB taste-trial runs.

Usage:
    WANDB_API_KEY=... python3 fetch_runs.py

Outputs:
    wandb-runs/<run_id>/meta.json            -- run metadata + summary
    wandb-runs/<run_id>/val_generations.jsonl -- per-step eval samples
    wandb-runs/index.json                    -- combined index for the viewer
"""

import sys as _sys
# The repo root contains a wandb/ run-storage directory that shadows the
# installed wandb package when Python adds cwd to sys.path. Remove it.
_sys.path = [p for p in _sys.path if not p in ("", ".")]

import json
import os
import sys
from pathlib import Path

ENTITY = "thefleet"
PROJECT = "fleet-browser-use-grpo"

RUNS = {
    "zd3sk2db": "baseline",
    "gjfocn7r": "judge",
    "x09ot84k": "abl-screenshots",
    "s24aawb9": "rel-sonnet",
}

OUT_DIR = Path("wandb-runs")


def _parse_table_artifact(artifact_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(artifact_dir.rglob("*.table.json")):
        try:
            data = json.loads(f.read_text())
            cols = data.get("columns", [])
            for row in data.get("data", []):
                rows.append(dict(zip(cols, row)))
        except Exception as e:
            print(f"    [warn] could not parse {f}: {e}")
    return rows


def fetch_run(api, run_id: str, label: str) -> list[dict]:
    import wandb

    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    print(f"  name={run.name!r}  state={run.state}  steps={run.summary.get('_step', '?')}")

    out_dir = OUT_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── metadata ──────────────────────────────────────────────────────────────
    meta = {
        "run_id": run_id,
        "label": label,
        "name": run.name,
        "state": run.state,
        "url": run.url,
        "config": dict(run.config),
        "summary": {k: v for k, v in run.summary.items() if not k.startswith("_")},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # ── val/generations artifacts ──────────────────────────────────────────────
    trajectories: list[dict] = []

    # Canonical WandB artifact name for tables logged via wandb.log({"val/generations": tbl})
    artifact_name = f"{ENTITY}/{PROJECT}/run-{run_id}-val_generations:latest"
    try:
        artifact = api.artifact(artifact_name)
        dl_dir = artifact.download(root=str(out_dir / "artifacts" / "val_generations"))
        rows = _parse_table_artifact(Path(dl_dir))
        print(f"    val/generations artifact: {len(rows)} rows")
        for row in rows:
            row.update({"_run_id": run_id, "_label": label, "_run_name": run.name})
        trajectories.extend(rows)
    except wandb.errors.CommError:
        print(f"    val/generations artifact not found — trying logged_artifacts()...")

    # Fallback: scan all logged artifacts for tables
    if not trajectories:
        for artifact in run.logged_artifacts():
            if "val" not in artifact.name.lower():
                continue
            try:
                dl_dir = artifact.download(root=str(out_dir / "artifacts" / artifact.name.replace("/", "_")))
                rows = _parse_table_artifact(Path(dl_dir))
                if rows:
                    print(f"    {artifact.name}: {len(rows)} rows")
                    for row in rows:
                        row.update({"_run_id": run_id, "_label": label, "_run_name": run.name})
                    trajectories.extend(rows)
            except Exception as e:
                print(f"    [warn] {artifact.name}: {e}")

    # ── dump_training_trajectories artifacts ───────────────────────────────────
    # If the run uploaded JSONL trajectory dumps as artifacts, grab those too.
    traj_rows: list[dict] = []
    for artifact in run.logged_artifacts():
        if artifact.type != "trajectory" and "traj" not in artifact.name.lower():
            continue
        try:
            dl_dir = Path(artifact.download(root=str(out_dir / "artifacts" / artifact.name.replace("/", "_"))))
            for f in sorted(dl_dir.rglob("*.jsonl")):
                for line in f.read_text().splitlines():
                    if line.strip():
                        entry = json.loads(line)
                        entry.update({"_run_id": run_id, "_label": label, "_run_name": run.name, "_artifact": artifact.name})
                        traj_rows.append(entry)
            print(f"    training traj artifact {artifact.name}: {len(traj_rows)} entries")
        except Exception as e:
            print(f"    [warn] {artifact.name}: {e}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_file = out_dir / "val_generations.jsonl"
    with open(out_file, "w") as f:
        for row in trajectories:
            f.write(json.dumps(row, default=str) + "\n")

    if traj_rows:
        traj_file = out_dir / "training_trajectories.jsonl"
        with open(traj_file, "w") as f:
            for row in traj_rows:
                f.write(json.dumps(row, default=str) + "\n")

    print(f"    → {len(trajectories)} val-gen rows, {len(traj_rows)} training traj rows")
    return trajectories


def main() -> None:
    try:
        import wandb
    except ImportError:
        sys.exit("wandb not installed — run: pip install wandb")

    if not os.environ.get("WANDB_API_KEY"):
        print("[warn] WANDB_API_KEY not set — wandb will use cached credentials if available")

    api = wandb.Api()
    OUT_DIR.mkdir(exist_ok=True)

    index: list[dict] = []
    all_rows: list[dict] = []

    for run_id, label in RUNS.items():
        print(f"\n[{label}] {run_id}")
        rows = fetch_run(api, run_id, label)
        all_rows.extend(rows)

        meta_path = OUT_DIR / run_id / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            index.append({"run_id": run_id, "label": label, "name": meta["name"],
                          "state": meta["state"], "url": meta.get("url", ""),
                          "n_val_rows": len(rows)})

    # Combined index for the viewer
    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))

    # Combined val_generations for quick access
    combined = OUT_DIR / "all_val_generations.jsonl"
    with open(combined, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"\nDone. {len(all_rows)} total val-gen rows across {len(index)} runs.")
    print(f"Start the viewer: python3 wandb_viewer.py")


if __name__ == "__main__":
    main()
