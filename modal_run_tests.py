from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

import modal


APP_NAME = os.environ.get("MODAL_TEST_APP_NAME", "skyrl-tests")
WORKDIR = "/root/SkyRL"


app = modal.App(APP_NAME)


def _ignore_local_artifacts(path: Path) -> bool:
    parts = set(path.parts)
    return (
        ".venv" in parts
        or ".git" in parts
        or "__pycache__" in parts
        or ".pytest_cache" in parts
        or ".mypy_cache" in parts
        or ".ruff_cache" in parts
        or ".idea" in parts
        or ".vscode" in parts
        or path.name.endswith(".pyc")
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ninja-build",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            # WSL-style hardlinking issues do not apply in Modal, but copy mode is safer and more predictable.
            "UV_LINK_MODE": "copy",
        }
    )
    .add_local_dir(".", remote_path=WORKDIR, copy=True, ignore=_ignore_local_artifacts)
    .workdir(WORKDIR)
    .run_commands("uv sync --extra skyrl-train --extra dev --frozen")
)


def _get_gpu_config():
    gpu_type = os.environ.get("MODAL_TEST_GPU", "L4")
    gpu_count = int(os.environ.get("MODAL_TEST_GPU_COUNT", "1"))
    return f"{gpu_type}:{gpu_count}"


@app.function(
    image=image,
    gpu=_get_gpu_config(),
    cpu=int(os.environ.get("MODAL_TEST_CPU", "8")),
    memory=int(os.environ.get("MODAL_TEST_MEMORY_MB", "32768")),
    timeout=int(os.environ.get("MODAL_TEST_TIMEOUT_SECS", str(60 * 60))),
)
def run_tests(pytest_args: Optional[List[str]] = None):
    cmd = ["uv", "run", "python", "-m", "pytest"]
    if pytest_args:
        cmd.extend(pytest_args)

    print("Running tests on Modal")
    print("Command:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=WORKDIR)


@app.local_entrypoint()
def main(pytest_args: str = "tests/train/test_fully_async_trainer.py"):
    run_tests.remote(shlex.split(pytest_args))
