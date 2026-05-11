"""
Create R2EGym dataset subsets from base difficulty-bucket directories.

The subsets used by the R2EGym 32B ThunderAgent recipe are:
  r2egym-train256-medium-hard-v1   (256 tasks, trivial=4  easy=16 medium=120 hard=116)
  r2egym-eval64-medium-hard-v1     (64 tasks,  trivial=1  easy=4  medium=30  hard=29)

Both are deterministic selections from the base buckets
  r2egym-trivial / r2egym-easy / r2egym-medium / r2egym-hard
which must already exist under DATA_ROOT (default: ~/data/harbor/).
The output directories contain task symlinks that the shared HarborTaskDataset
can scan directly, plus MANIFEST.json for reproducibility and image pre-pull.

Usage:

    # Create both subsets (default):
    python examples/train/thunder_agent/scripts/r2egym_32b/prepare_dataset.py

    # Create with explicit data root:
    python examples/train/thunder_agent/scripts/r2egym_32b/prepare_dataset.py \
        --data-root /data/harbor

    # Dry-run: print what would be created without writing files:
    python examples/train/thunder_agent/scripts/r2egym_32b/prepare_dataset.py --dry-run

If the base bucket directories do not exist, the script will print the
HuggingFace download command and exit.  The base datasets are hosted at:

    NovaSky-AI/r2egym-trivial
    NovaSky-AI/r2egym-easy
    NovaSky-AI/r2egym-medium
    NovaSky-AI/r2egym-hard

Download them first with:

    python examples/train_integrations/harbor/prepare_harbor_dataset.py \
        --dataset NovaSky-AI/r2egym-medium --output_dir ~/data/harbor/r2egym-medium
    # (repeat for trivial / easy / hard)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

HF_BASE_REPOS = {
    "trivial": "NovaSky-AI/r2egym-trivial",
    "easy": "NovaSky-AI/r2egym-easy",
    "medium": "NovaSky-AI/r2egym-medium",
    "hard": "NovaSky-AI/r2egym-hard",
}

BUCKET_DIRS = {
    "trivial": "r2egym-trivial",
    "easy": "r2egym-easy",
    "medium": "r2egym-medium",
    "hard": "r2egym-hard",
}


@dataclass
class SubsetSpec:
    name: str
    seed: str
    counts: Dict[str, int]
    # prefix_subset: tasks already confirmed from a smaller subset (kept first)
    prefix_tasks: Dict[str, List[str]] = field(default_factory=dict)
    # excluded_tasks: tasks that must not appear (reserved for eval / train overlap)
    excluded_tasks: set = field(default_factory=set)


TRAIN_256_SPEC = SubsetSpec(
    name="r2egym-train256-medium-hard-v1",
    seed="r2egym-medium-hard-v1-20260325",
    counts={"trivial": 4, "easy": 16, "medium": 120, "hard": 116},
    # prefix: keep existing train128 tasks in order
    # excluded: reserve all eval64 tasks
)

EVAL_64_SPEC = SubsetSpec(
    name="r2egym-eval64-medium-hard-v1",
    seed="r2egym-medium-hard-v1-20260325",
    counts={"trivial": 1, "easy": 4, "medium": 30, "hard": 29},
    # prefix: keep existing eval32 tasks in order
    # excluded: tasks already in train128
)

# Prerequisite subsets that must already exist (or be located) under data_root.
PREREQUISITE_SUBSETS = {
    "eval32": "r2egym-eval32-medium-major-v1",
    "train128": "r2egym-train128-medium-major-v1",
}


def _sha256_key(seed: str, bucket: str, rel_path: str) -> str:
    return hashlib.sha256(f"{seed}:{bucket}:{rel_path}".encode()).hexdigest()


def _scan_bucket(bucket_dir: Path, bucket_name: str) -> List[str]:
    """Return sorted rel-paths for valid task directories in a bucket."""
    tasks = []
    for task_dir in sorted(bucket_dir.iterdir()):
        if task_dir.is_dir() and (task_dir / "instruction.md").is_file():
            tasks.append(f"{BUCKET_DIRS[bucket_name]}/{task_dir.name}")
    return tasks


def _select_tasks(
    all_tasks: List[str],
    count: int,
    seed: str,
    bucket: str,
    prefix: Optional[List[str]] = None,
    excluded: Optional[set] = None,
) -> List[str]:
    """
    Select `count` tasks deterministically:
      1. Start with prefix tasks (kept as-is, in order).
      2. From the remainder, remove excluded; sort by sha256(seed:bucket:path).
      3. Append until count is reached.
    """
    prefix = list(prefix or [])
    excluded = excluded or set()

    # Remove excluded from prefix (shouldn't happen, but be safe)
    prefix = [t for t in prefix if t not in excluded]

    prefix_set = set(prefix)
    remaining = [t for t in all_tasks if t not in prefix_set and t not in excluded]
    remaining.sort(key=lambda t: _sha256_key(seed, bucket, t))

    selected = prefix[:]
    for task in remaining:
        if len(selected) >= count:
            break
        selected.append(task)

    if len(selected) < count:
        raise ValueError(
            f"Not enough tasks in bucket '{bucket}': " f"need {count}, found {len(selected)} (after exclusions)."
        )
    return selected


def build_manifest(
    spec: SubsetSpec,
    data_root: Path,
    prefix_manifest: Optional[Dict] = None,
    excluded_manifest: Optional[Dict] = None,
) -> Dict:
    tasks: Dict[str, List[str]] = {}
    for bucket, count in spec.counts.items():
        bucket_dir = data_root / BUCKET_DIRS[bucket]
        all_tasks = _scan_bucket(bucket_dir, bucket)

        prefix: List[str] = []
        if prefix_manifest:
            prefix = list(prefix_manifest.get("tasks", {}).get(bucket, []))

        excluded: set = set()
        if excluded_manifest:
            excluded = set(excluded_manifest.get("tasks", {}).get(bucket, []))

        tasks[bucket] = _select_tasks(all_tasks, count, spec.seed, bucket, prefix=prefix, excluded=excluded)

    total = sum(len(v) for v in tasks.values())
    return {
        "subset_name": spec.name,
        "seed": spec.seed,
        "counts": spec.counts,
        "total": total,
        "tasks": tasks,
    }


def _subset_entry_name(rel_task_path: str) -> str:
    return rel_task_path.replace("/", "__")


def _clean_subset_entries(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in output_dir.iterdir():
        if child.name == "MANIFEST.json":
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _link_or_copy_task(src: Path, dst: Path, materialize: bool) -> None:
    if materialize:
        shutil.copytree(src, dst, symlinks=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def write_subset(
    manifest: Dict,
    data_root: Path,
    output_dir: Path,
    materialize: bool = False,
    dry_run: bool = False,
) -> None:
    task_rels = [task for bucket_tasks in manifest["tasks"].values() for task in bucket_tasks]
    if dry_run:
        print(f"[dry-run] Would create {output_dir}  ({manifest['total']} tasks)")
        print(f"[dry-run] Would write {output_dir}/MANIFEST.json")
        for bucket, tasks in manifest["tasks"].items():
            print(f"  {bucket}: {len(tasks)}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_subset_entries(output_dir)

    seen_names = set()
    for rel_task_path in task_rels:
        src = (data_root / rel_task_path).resolve()
        dst_name = _subset_entry_name(rel_task_path)
        if dst_name in seen_names:
            raise ValueError(f"duplicate subset entry name: {dst_name}")
        seen_names.add(dst_name)
        _link_or_copy_task(src, output_dir / dst_name, materialize=materialize)

    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    mode = "copied" if materialize else "symlinked"
    print(f"Wrote {manifest_path} and {mode} {manifest['total']} tasks into {output_dir}")
    for bucket, tasks in manifest["tasks"].items():
        print(f"  {bucket}: {len(tasks)}")


def check_base_buckets(data_root: Path) -> bool:
    missing = []
    for bucket, dirname in BUCKET_DIRS.items():
        bucket_dir = data_root / dirname
        if not bucket_dir.is_dir():
            missing.append((bucket, dirname))
    if missing:
        print("ERROR: the following base bucket directories are missing:")
        for bucket, dirname in missing:
            print(f"  {data_root / dirname}  (bucket: {bucket})")
        print()
        print("Download them with:")
        for bucket, dirname in missing:
            hf_repo = HF_BASE_REPOS[bucket]
            print(
                f"  python examples/train_integrations/harbor/prepare_harbor_dataset.py"
                f" --dataset {hf_repo}"
                f" --output_dir {data_root / dirname}"
            )
        return False
    return True


def load_prerequisite(data_root: Path, key: str) -> Optional[Dict]:
    """Load a prerequisite subset MANIFEST.json, warning if absent."""
    dirname = PREREQUISITE_SUBSETS[key]
    path = data_root / dirname / "MANIFEST.json"
    if not path.exists():
        print(f"WARNING: prerequisite subset '{dirname}' not found at {path}.")
        print("  Selection will proceed without it; results may differ from the reference.")
        return None
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        default="~/data/harbor",
        help="Root directory containing r2egym-{trivial,easy,medium,hard}/ (default: ~/data/harbor)",
    )
    parser.add_argument(
        "--subset",
        choices=["train256", "eval64", "all"],
        default="all",
        help="Which subset(s) to create (default: all)",
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Copy task directories instead of creating symlinks",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing files")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    print(f"Data root: {data_root}")

    if not check_base_buckets(data_root):
        raise SystemExit(1)

    # Load prerequisite subsets (eval32, train128) for prefix/exclusion logic.
    eval32_manifest = load_prerequisite(data_root, "eval32")
    train128_manifest = load_prerequisite(data_root, "train128")

    # Build eval64 first (prefix=eval32, excluded=train128).
    # train256 must not overlap with eval64, so eval64 goes first.
    eval_manifest: Optional[Dict] = None
    if args.subset in ("eval64", "all"):
        print("\nBuilding r2egym-eval64-medium-hard-v1 ...")
        eval_manifest = build_manifest(
            EVAL_64_SPEC,
            data_root,
            prefix_manifest=eval32_manifest,
            excluded_manifest=train128_manifest,
        )
        write_subset(
            eval_manifest,
            data_root,
            data_root / EVAL_64_SPEC.name,
            materialize=args.materialize,
            dry_run=args.dry_run,
        )

    if args.subset in ("train256", "all"):
        # If we didn't build eval64 above, try to load the existing one.
        if eval_manifest is None:
            eval_path = data_root / EVAL_64_SPEC.name / "MANIFEST.json"
            if eval_path.exists():
                eval_manifest = json.loads(eval_path.read_text())

        print("\nBuilding r2egym-train256-medium-hard-v1 ...")
        # prefix=train128 (keep existing tasks in order), excluded=eval64
        train_manifest = build_manifest(
            TRAIN_256_SPEC,
            data_root,
            prefix_manifest=train128_manifest,
            excluded_manifest=eval_manifest,
        )
        write_subset(
            train_manifest,
            data_root,
            data_root / TRAIN_256_SPEC.name,
            materialize=args.materialize,
            dry_run=args.dry_run,
        )

    print("\nDone.")
    if not args.dry_run:
        print(
            f"\nSet in your run script:\n"
            f"  TRAIN_DATA=\"['{data_root / TRAIN_256_SPEC.name}']\"\n"
            f"  EVAL_DATA=\"['{data_root / EVAL_64_SPEC.name}']\""
        )


if __name__ == "__main__":
    main()
