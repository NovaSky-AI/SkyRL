"""Extract CodeContests Harbor tasks and point each at a prebuilt registry image.

The Ray Sandbox gRPC facade runs prebuilt registry images only, and the
CodeContests tasks ship just a Dockerfile (ubuntu:24.04 + python3, WORKDIR
/app), which Harbor would ask the SDK to build. Each extracted task gets an
`[environment] docker_image` instead, so Harbor uses `Image.from_registry`.

    uv run --no-project --with pyarrow python examples/train_integrations/harbor/ray_sandbox/prepare_tasks.py \
        --parquet tasks.parquet --out examples/train_integrations/harbor/ray_sandbox/tasks --count 10

Write the tasks inside the checkout, so the job's working_dir carries them to
every node. The parquet is the one `prepare_harbor_dataset.py` downloads:
https://huggingface.co/datasets/open-thoughts/CodeContests/resolve/main/tasks.parquet
"""

import argparse
import io
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pyarrow.parquet as pq

# python:3.13-slim-bookworm + WORKDIR /app, public on Docker Hub, amd64 only.
# Matches what the task Dockerfile provides (python3 and WORKDIR /app); the
# verifier apt-installs everything else it needs.
DEFAULT_IMAGE = "alexgshaw/cancel-async-tasks:20251031"


def _extract(archive: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        for member in tf.getmembers():
            parts = [p for p in PurePosixPath(member.name).parts if p not in ("..", ".", "/")]
            if not parts:
                continue
            target = dest.joinpath(*parts)
            if not target.resolve().is_relative_to(dest.resolve()):
                raise SystemExit(f"archive member {member.name!r} escapes {dest}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tf.extractfile(member).read())


def _set_docker_image(task_toml: Path, image: str) -> None:
    text = task_toml.read_text()
    if "[environment]" in text:
        raise SystemExit(f"{task_toml}: already has an [environment] section")
    task_toml.write_text(
        text.rstrip("\n")
        + "\n\n[environment]\n"
        + "# The Ray Sandbox facade runs prebuilt registry images only; see prepare_tasks.py.\n"
        + f'docker_image = "{image}"\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    rows = sorted(table.to_pylist(), key=lambda row: row["path"])[: args.count]
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for row in rows:
        name = PurePosixPath(row["path"]).name
        if name in ("", ".", ".."):
            raise SystemExit(f"bad task path {row['path']!r}")
        task_dir = out / name
        _extract(row["task_binary"], task_dir)
        _set_docker_image(task_dir / "task.toml", args.docker_image)
    print(f"wrote {len(rows)} tasks to {out} (docker_image={args.docker_image})")


if __name__ == "__main__":
    main()
