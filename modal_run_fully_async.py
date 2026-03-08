from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import modal


APP_NAME = os.environ.get("MODAL_APP_NAME", "skyrl-fully-async-gsm8k")
WORKDIR = "/root/SkyRL"
DATA_MOUNT = "/vol/data"
CKPT_MOUNT = "/vol/ckpts"
WANDB_SECRET_NAME = "wandb-secret"


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


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ninja-build",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(".", remote_path=WORKDIR, copy=True, ignore=_ignore_local_artifacts)
    .workdir(WORKDIR)
    .run_commands("uv sync --extra fsdp --frozen")
)

test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "ninja-build",
    )
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin"})
    .add_local_dir(".", remote_path=WORKDIR, copy=True, ignore=_ignore_local_artifacts)
    .workdir(WORKDIR)
    .run_commands("uv sync --extra skyrl-train --extra dev --frozen")
)


data_volume = modal.Volume.from_name(os.environ.get("MODAL_DATA_VOLUME", "skyrl-data"), create_if_missing=True)
ckpt_volume = modal.Volume.from_name(os.environ.get("MODAL_CKPT_VOLUME", "skyrl-ckpts"), create_if_missing=True)


def _get_env(local_overrides: Optional[Dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    if local_overrides:
        env.update(local_overrides)

    train_batch_size = env.get("TRAIN_BATCH_SIZE", "256")
    env.update(
        {
            "HOME": "/tmp",
            "DATA_DIR": env.get("DATA_DIR", f"{DATA_MOUNT}/gsm8k"),
            "CKPT_ROOT": env.get("CKPT_ROOT", CKPT_MOUNT),
            "NUM_INFERENCE_GPUS": env.get("NUM_INFERENCE_GPUS", "1"),
            "NUM_POLICY_GPUS": env.get("NUM_POLICY_GPUS", "1"),
            "LOGGER": env.get("LOGGER", "console"),
            "INFERENCE_BACKEND": env.get("INFERENCE_BACKEND", "vllm"),
            "TRAIN_BATCH_SIZE": train_batch_size,
            "POLICY_MINI_BATCH_SIZE": env.get("POLICY_MINI_BATCH_SIZE", train_batch_size),
            "MAX_STALENESS_STEPS": env.get("MAX_STALENESS_STEPS", "4"),
            "NUM_PARALLEL_GENERATION_WORKERS": env.get(
                "NUM_PARALLEL_GENERATION_WORKERS", train_batch_size
            ),
        }
    )
    return env


def _get_gpu_config():
    gpu_type = os.environ.get("MODAL_GPU", "L4")
    gpu_count = int(os.environ.get("MODAL_GPU_COUNT", "2"))
    return f"{gpu_type}:{gpu_count}"


def _get_cli_override(extra_args: Optional[List[str]], key: str) -> Optional[str]:
    if not extra_args:
        return None
    prefix = f"{key}="
    for arg in reversed(extra_args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def _resolve_run_name(env: dict[str, str], extra_args: Optional[List[str]]) -> str:
    if override := _get_cli_override(extra_args, "trainer.run_name"):
        return override
    return (
        "gsm8k-fully-async-qwen2.5_1.5B-useTIS_token-"
        f"maxStale{env['MAX_STALENESS_STEPS']}-"
        f"numCon{env['NUM_PARALLEL_GENERATION_WORKERS']}-"
        f"{env['NUM_POLICY_GPUS']}train{env['NUM_INFERENCE_GPUS']}gen"
    )


def _run_fully_async_impl(extra_args: Optional[List[str]] = None, env_overrides: Optional[Dict[str, str]] = None):
    env = _get_env(env_overrides)
    resolved_extra_args = list(extra_args) if extra_args else []
    if _get_cli_override(resolved_extra_args, "trainer.ckpt_path") is None:
        run_name = _resolve_run_name(env, resolved_extra_args)
        resolved_extra_args.append(f"trainer.ckpt_path={env['CKPT_ROOT']}/{run_name}")

    script_path = Path(WORKDIR) / "examples/train/fully_async/fully_async_run_gsm8k.sh"
    cmd = ["bash", str(script_path)]
    if resolved_extra_args:
        cmd.extend(resolved_extra_args)

    print("Running fully async GSM8K on Modal with:")
    for key in [
        "DATA_DIR",
        "CKPT_ROOT",
        "NUM_POLICY_GPUS",
        "NUM_INFERENCE_GPUS",
        "TRAIN_BATCH_SIZE",
        "POLICY_MINI_BATCH_SIZE",
        "MAX_STALENESS_STEPS",
        "NUM_PARALLEL_GENERATION_WORKERS",
        "LOGGER",
        "INFERENCE_BACKEND",
    ]:
        print(f"  {key}={env[key]}")
    print("  CKPT_PATH=", _get_cli_override(resolved_extra_args, "trainer.ckpt_path"), sep="")

    print("Command:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, env=env, cwd=WORKDIR)


@app.function(
    image=gpu_image,
    gpu=_get_gpu_config(),
    cpu=int(os.environ.get("MODAL_CPU", "16")),
    memory=int(os.environ.get("MODAL_MEMORY_MB", "65536")),
    timeout=int(os.environ.get("MODAL_TIMEOUT_SECS", str(12 * 60 * 60))),
    volumes={
        DATA_MOUNT: data_volume,
        CKPT_MOUNT: ckpt_volume,
    },
)
def run_fully_async(extra_args: Optional[List[str]] = None, env_overrides: Optional[Dict[str, str]] = None):
    _run_fully_async_impl(extra_args, env_overrides)


@app.function(
    image=gpu_image,
    gpu=_get_gpu_config(),
    cpu=int(os.environ.get("MODAL_CPU", "16")),
    memory=int(os.environ.get("MODAL_MEMORY_MB", "65536")),
    timeout=int(os.environ.get("MODAL_TIMEOUT_SECS", str(12 * 60 * 60))),
    volumes={
        DATA_MOUNT: data_volume,
        CKPT_MOUNT: ckpt_volume,
    },
    secrets=[modal.Secret.from_name(WANDB_SECRET_NAME)],
)
def run_fully_async_with_wandb(
    extra_args: Optional[List[str]] = None, env_overrides: Optional[Dict[str, str]] = None
):
    _run_fully_async_impl(extra_args, env_overrides)


@app.function(
    image=test_image,
    cpu=int(os.environ.get("MODAL_TEST_CPU", "8")),
    memory=int(os.environ.get("MODAL_TEST_MEMORY_MB", "32768")),
    timeout=int(os.environ.get("MODAL_TEST_TIMEOUT_SECS", str(60 * 60))),
)
def run_tests(pytest_args: Optional[List[str]] = None):
    cmd = ["uv", "run", "pytest"]
    if pytest_args:
        cmd.extend(pytest_args)
    else:
        cmd.append("tests/train/test_fully_async_trainer.py")

    print("Running tests on Modal")
    print("Command:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=WORKDIR)


@app.local_entrypoint()
def main(mode: str = "train", extra_args: str = ""):
    parsed_extra_args = shlex.split(extra_args) if extra_args else None
    env_overrides = {
        key: value
        for key in [
            "DATA_DIR",
            "CKPT_ROOT",
            "NUM_INFERENCE_GPUS",
            "NUM_POLICY_GPUS",
            "LOGGER",
            "INFERENCE_BACKEND",
            "TRAIN_BATCH_SIZE",
            "POLICY_MINI_BATCH_SIZE",
            "MAX_STALENESS_STEPS",
            "NUM_PARALLEL_GENERATION_WORKERS",
        ]
        if (value := os.environ.get(key)) is not None
    }
    if mode == "train":
        if env_overrides.get("LOGGER", os.environ.get("LOGGER", "console")) == "wandb":
            run_fully_async_with_wandb.remote(parsed_extra_args, env_overrides)
        else:
            run_fully_async.remote(parsed_extra_args, env_overrides)
    elif mode == "test":
        run_tests.remote(parsed_extra_args)
    else:
        raise ValueError("mode must be 'train' or 'test'")
