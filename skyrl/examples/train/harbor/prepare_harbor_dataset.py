"""
Prepare Harbor task datasets from HuggingFace Hub.

Downloads a dataset and extracts Harbor tasks if they are stored as parquet
files with tar-archived task directories. Datasets that already contain task
directories are symlinked directly.

Output directory defaults to ~/data/harbor/<repo-name> (derived from the
dataset name), so each dataset automatically gets its own directory.

Usage:
    # Prepare training data -> ~/data/harbor/OpenThoughts-Agent-v1-RL/
    python examples/harbor/prepare_harbor_dataset.py \
        --dataset open-thoughts/OpenThoughts-Agent-v1-RL

    # Prepare eval data -> ~/data/harbor/OpenThoughts-TB-dev/
    python examples/harbor/prepare_harbor_dataset.py \
        --dataset open-thoughts/OpenThoughts-TB-dev

    # Prepare code-contests training data -> ~/data/harbor/code-contests-sandboxes-with-tests/
    python examples/harbor/prepare_harbor_dataset.py \
        --dataset DCAgent/code-contests-sandboxes-with-tests

    # Override output directory
    python examples/harbor/prepare_harbor_dataset.py \
        --dataset open-thoughts/OpenThoughts-Agent-v1-RL \
        --output_dir ~/my-custom-path
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                (dest_dir / member_name).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)


def _count_valid_tasks(directory: Path) -> int:
    """Count subdirectories containing instruction.md (valid Harbor tasks)."""
    if not directory.exists() or not directory.is_dir():
        return 0
    count = 0
    for d in directory.iterdir():
        if d.is_dir() and (d / "instruction.md").exists():
            count += 1
    return count


def _find_task_parquets(snapshot_dir: str) -> list:
    """Find parquet files that contain path/task_binary columns."""
    import pyarrow.parquet as pq

    parquets = []
    for f in Path(snapshot_dir).glob("*.parquet"):
        try:
            schema = pq.read_schema(f)
            if "path" in schema.names and "task_binary" in schema.names:
                parquets.append(f)
        except Exception as e:
            print(f"  Warning: Could not read schema from {f}: {e}")
            continue
    return parquets


def _extract_from_parquet(parquet_path: Path, output_dir: Path) -> int:
    """Extract tasks from a parquet file containing task_binary column.

    Returns number of tasks extracted.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    path_col = table.column("path").to_pylist()
    data_col = table.column("task_binary").to_pylist()

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    total = len(path_col)

    for i, (rel_path, data) in enumerate(zip(path_col, data_col)):
        if i % 100 == 0:
            print(f"  Extracting {i}/{total}...")

        if not isinstance(rel_path, str):
            continue
        if not isinstance(data, (bytes, bytearray, memoryview)):
            continue

        safe_rel = PurePosixPath(rel_path)
        parts = [p for p in safe_rel.parts if p not in ("..", "")]
        rel_norm = Path(*parts) if parts else Path(f"task_{i}")
        target_dir = (output_dir / rel_norm).resolve()

        if not _is_within(output_dir, target_dir):
            continue

        if target_dir.exists():
            # Skip already-extracted tasks
            if (target_dir / "instruction.md").exists():
                extracted += 1
                continue

        try:
            _safe_extract_tar(bytes(data), target_dir)
            extracted += 1
        except Exception as e:
            print(f"  Warning: Failed to extract {rel_path}: {e}")

    return extracted


def _default_output_dir(dataset_name: str) -> str:
    """Derive a default output directory from the dataset name.

    E.g. "open-thoughts/OpenThoughts-Agent-v1-RL" -> "~/data/harbor/OpenThoughts-Agent-v1-RL"
    """
    repo_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    return os.path.join("~/data/harbor", repo_name)


def prepare(dataset_name: str, output_dir: str | None = None, force: bool = False) -> str:
    """Download and prepare a Harbor task dataset.

    Args:
        dataset_name: HuggingFace dataset repo (e.g. "open-thoughts/OpenThoughts-Agent-v1-RL")
        output_dir: Directory to write/symlink extracted tasks to. If None, derived from
            dataset name (e.g. ~/data/harbor/OpenThoughts-Agent-v1-RL).
        force: If True, re-extract even if output_dir already has tasks

    Returns:
        Path to the directory containing ready-to-use task directories.
    """
    from huggingface_hub import snapshot_download

    if output_dir is None:
        output_dir = _default_output_dir(dataset_name)
    output_path = Path(os.path.expanduser(output_dir)).resolve()

    # Check if already prepared
    if not force and _count_valid_tasks(output_path) > 0:
        n = _count_valid_tasks(output_path)
        print(f"Already prepared: {output_path} ({n} tasks). Use --force to re-extract.")
        return str(output_path)

    # If it's a stale symlink or force, clean up
    if output_path.is_symlink():
        output_path.unlink()
    elif force and output_path.exists():
        shutil.rmtree(output_path)

    # Download dataset
    print(f"Downloading {dataset_name}...")
    snapshot_dir = snapshot_download(repo_id=dataset_name, repo_type="dataset")
    print(f"Downloaded to {snapshot_dir}")

    # Check for parquet files with task_binary column
    task_parquets = _find_task_parquets(snapshot_dir)

    if task_parquets:
        # Parquet-based dataset: extract tar archives
        print(f"Found {len(task_parquets)} task parquet file(s), extracting...")
        total_extracted = 0
        for pq_file in task_parquets:
            print(f"  Processing {pq_file.name}...")
            total_extracted += _extract_from_parquet(pq_file, output_path)
        print(f"Extracted {total_extracted} tasks to {output_path}")
    else:
        # Direct task directories: symlink to snapshot
        n = _count_valid_tasks(Path(snapshot_dir))
        if n > 0:
            print(f"Dataset contains {n} task directories directly, creating symlink...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.symlink_to(snapshot_dir)
        else:
            print(f"Error: No task parquets or task directories found in {snapshot_dir}")
            sys.exit(1)

    n = _count_valid_tasks(output_path)
    print(f"Done! {n} tasks ready at {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Harbor task dataset from HuggingFace Hub")
    parser.add_argument(
        "--dataset",
        required=True,
        help="HuggingFace dataset name (e.g. open-thoughts/OpenThoughts-Agent-v1-RL)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for extracted tasks (default: ~/data/harbor/<dataset-repo-name>)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output_dir already has tasks",
    )
    args = parser.parse_args()

    prepare(args.dataset, args.output_dir, args.force)
