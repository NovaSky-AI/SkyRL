#!/usr/bin/env python3
"""List or pull Harbor/R2EGym Docker base images for a recipe dataset spec."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_dataset_spec(raw: str) -> list[str]:
    value = ast.literal_eval(raw)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"dataset spec must be a string or list of strings: {raw!r}")


def manifest_tasks(dataset_dir: Path) -> list[Path] | None:
    manifest_path = dataset_dir / "MANIFEST.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"invalid MANIFEST.json tasks field: {manifest_path}")
    data_root = dataset_dir.parent
    paths: list[Path] = []
    for bucket_tasks in tasks.values():
        if not isinstance(bucket_tasks, list):
            raise ValueError(f"invalid MANIFEST.json bucket list: {manifest_path}")
        for rel_path in bucket_tasks:
            paths.append(data_root / rel_path)
    return paths


def task_dirs(dataset_dir: Path) -> list[Path]:
    manifest_paths = manifest_tasks(dataset_dir)
    if manifest_paths is not None:
        return manifest_paths
    return sorted(
        child for child in dataset_dir.iterdir() if child.is_dir() and (child / "environment" / "Dockerfile").exists()
    )


def dockerfile_base_image(dockerfile: Path) -> str | None:
    for raw_line in dockerfile.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("FROM "):
            image = line.split(None, 1)[1].split(" AS ", 1)[0].split(" as ", 1)[0]
            return image.strip()
    return None


def collect_images(dataset_specs: list[str]) -> list[str]:
    images: set[str] = set()
    for raw_path in dataset_specs:
        dataset_dir = Path(os.path.expandvars(os.path.expanduser(raw_path))).resolve()
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"missing dataset directory: {dataset_dir}")
        for task_dir in task_dirs(dataset_dir):
            dockerfile = task_dir / "environment" / "Dockerfile"
            if not dockerfile.exists():
                raise FileNotFoundError(f"missing Dockerfile for Harbor task: {dockerfile}")
            image = dockerfile_base_image(dockerfile)
            if image:
                images.add(image)
    return sorted(images)


def pull_images(images: list[str], retries: int, retry_sleep: int) -> None:
    failed: list[str] = []
    for index, image in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] docker pull {image}", flush=True)
        for attempt in range(1, retries + 1):
            result = subprocess.run(["docker", "pull", image], text=True)
            if result.returncode == 0:
                break
            if attempt < retries:
                print(
                    f"docker pull failed for {image}; retrying in {retry_sleep}s " f"({attempt}/{retries})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(retry_sleep)
        else:
            failed.append(image)
    if failed:
        print("Failed to pull these images:", file=sys.stderr)
        for image in failed:
            print(image, file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--mode", choices=["list", "pull"], default="list")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=30)
    args = parser.parse_args()

    specs = parse_dataset_spec(args.train_data) + parse_dataset_spec(args.eval_data)
    images = collect_images(specs)
    for image in images:
        print(image)
    print(f"image_count={len(images)}", file=sys.stderr)

    if args.mode == "pull":
        pull_images(images, retries=args.retries, retry_sleep=args.retry_sleep)


if __name__ == "__main__":
    main()
