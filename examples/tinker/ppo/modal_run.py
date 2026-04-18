"""
Modal entrypoint that runs the Tinker PPO example end-to-end.

What this does inside a single Modal container:
    1. Preprocesses GSM8K into ~/data/gsm8k/{train,validation}.parquet
       (unless already present in the persistent Volume).
    2. Launches the Tinker API server (run_tinker_server.sh) in the background.
    3. Waits for http://localhost:8000 to become ready.
    4. Runs ppo_client.py against that local server.
    5. Tears the server down on exit.

Server and client share the same container, so the client can keep the
default `--base-url http://localhost:8000`. Four GPUs are requested because
the server-side defaults in run_tinker_server.sh put policy, ref, critic, and
the vLLM engines on 4 GPUs with colocate_all=true.

Usage (from the repo root):

    modal run examples/tinker/ppo/modal_run.py

    # override client limits / gpu class
    modal run examples/tinker/ppo/modal_run.py \
        --max-train-steps 2 --max-eval-steps 1 --gpu-type "H100:4"

    # skip data prep if train/validation parquet already exist in the Volume
    modal run examples/tinker/ppo/modal_run.py --skip-data-prep

The client uses TINKER_API_KEY=tml-dummy, matching the bundled SkyRL Tinker
server's default (see the NovaSky blog post and run_tinker_server.sh).
"""

from __future__ import annotations

import os
import pathlib
import shlex

import modal


REMOTE_REPO = "/root/SkyRL"
DATA_DIR = "/root/data/gsm8k"
CKPT_DIR = "/root/ckpts/gsm8k_1.5B_ckpt_ppo"
HF_CACHE = "/root/.cache/huggingface"


def _is_repo_root(path: pathlib.Path) -> bool:
    return (
        path.exists()
        and (path / "pyproject.toml").exists()
        and (path / "examples").exists()
        and (path / "skyrl").exists()
        and (path / "skyrl-gym").exists()
    )


def _find_repo_root() -> pathlib.Path:
    candidates: list[pathlib.Path] = []

    env_repo_root = os.environ.get("SKYRL_REPO_ROOT")
    if env_repo_root:
        candidates.append(pathlib.Path(env_repo_root))

    candidates.append(pathlib.Path(REMOTE_REPO))

    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        base = start if start.is_dir() else start.parent
        candidates.extend([base, *base.parents])

    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_repo_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate the SkyRL repo root. "
        "Set SKYRL_REPO_ROOT or run this script from inside the SkyRL repository."
    )


REPO_ROOT = _find_repo_root()


# Persistent volumes: keep the preprocessed dataset, the HF model cache, and
# the checkpoints / metrics across runs so repeated invocations are cheap.
data_volume = modal.Volume.from_name("skyrl-tinker-ppo-data", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("skyrl-tinker-ppo-ckpts", create_if_missing=True)
hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)


image = (
    # Base image with CUDA runtime + Python 3.12. flash-attn and vllm wheels in
    # the fsdp extra are built against CUDA 12.x, so use a matching CUDA base.
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    # Install uv (the repo's build tool of choice).
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SKYRL_REPO_ROOT": REMOTE_REPO,
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{REMOTE_REPO}/.venv",
        }
    )
    # Copy the whole repo into the image. `copy=True` makes it part of the
    # image layer (rather than a runtime mount) so `uv sync` can see it.
    .add_local_dir(str(REPO_ROOT), REMOTE_REPO, copy=True, ignore=[".venv", "**/__pycache__"])
    .workdir(REMOTE_REPO)
    # Resolve and install the two extras the example needs. This is the
    # expensive step (flash-attn, vllm, torch-cu128) and is cached in the
    # image layer.
    .run_commands(
        "uv sync --extra tinker --extra fsdp",
        gpu="any",  # some wheels probe CUDA during install
    )
)


app = modal.App("skyrl-tinker-ppo")


def _forward_optional_env(env: dict[str, str], keys: list[str]) -> None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            env[key] = value


@app.function(
    image=image,
    gpu="H100:4",
    timeout=24 * 60 * 60,  # PPO loop is long; give it a full day.
    volumes={
        "/root/data": data_volume,
        "/root/ckpts": ckpt_volume,
        HF_CACHE: hf_volume,
    },
)
def run_ppo(
    max_train_steps: int | None = None,
    max_eval_steps: int | None = None,
    skip_data_prep: bool = False,
    backend_config: str | None = None,
) -> None:
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time
    import urllib.error
    import urllib.request

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    env = os.environ.copy()
    # Matches the SkyRL Tinker README / NovaSky blog post: the bundled server
    # accepts the literal "tml-dummy" key, no Modal Secret needed.
    env["TINKER_API_KEY"] = "tml-dummy"
    env["HOME"] = "/root"  # the client expands ~/data/gsm8k and ~/ckpts/...
    _forward_optional_env(
        env,
        [
            "WANDB_API_KEY",
            "WANDB_PROJECT",
            "WANDB_ENTITY",
            "WANDB_RUN_NAME",
            "WANDB_TAGS",
        ],
    )
    if backend_config:
        env["BACKEND_CONFIG"] = backend_config

    train_parquet = os.path.join(DATA_DIR, "train.parquet")
    val_parquet = os.path.join(DATA_DIR, "validation.parquet")

    # -------- 1. GSM8K data prep --------
    data_ready = os.path.exists(train_parquet) and os.path.exists(val_parquet)
    if not data_ready and not skip_data_prep:
        print(">>> [modal] Preparing GSM8K parquet files")
        subprocess.run(
            [
                "uv", "run", "--extra", "tinker", "--with", "datasets",
                "python", "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir", DATA_DIR,
            ],
            cwd=REMOTE_REPO,
            env=env,
            check=True,
        )
        data_volume.commit()
    elif data_ready:
        print(">>> [modal] GSM8K parquet files already present; skipping prep")
    else:
        print(">>> [modal] --skip-data-prep set but dataset missing; the client will fail")

    # -------- 2. Launch Tinker server in the background --------
    print(">>> [modal] Starting Tinker API server")
    server = subprocess.Popen(
        ["bash", "examples/tinker/ppo/run_tinker_server.sh"],
        cwd=REMOTE_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    # Stream server logs to the Modal run output on a daemon thread.
    def _pump_logs(stream, prefix: str) -> None:
        for line in stream:
            print(f"[{prefix}] {line}", end="")

    threading.Thread(
        target=_pump_logs, args=(server.stdout, "server"), daemon=True
    ).start()

    def _kill_server() -> None:
        if server.poll() is not None:
            return
        print(">>> [modal] Stopping Tinker API server")
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            server.wait(timeout=30)

    # -------- 3. Wait for readiness on :8000 --------
    # Model download + vLLM engine warmup can take a long time; cap at 30 min.
    base_url = "http://localhost:8000"
    deadline = time.time() + 30 * 60
    ready = False
    while time.time() < deadline:
        if server.poll() is not None:
            _kill_server()
            raise RuntimeError(
                f"Tinker server exited early with code {server.returncode}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/docs", timeout=5) as resp:
                if resp.status < 500:
                    ready = True
                    break
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)

    if not ready:
        _kill_server()
        raise RuntimeError("Tinker server did not become ready within 30 minutes")

    print(">>> [modal] Tinker API server is ready; launching PPO client")

    # -------- 4. Run the PPO client --------
    client_cmd: list[str] = [
        "uv", "run", "--extra", "tinker", "--with", "datasets", "--with", "torch",
        "python", "examples/tinker/ppo/ppo_client.py",
        "--base-url", base_url,
        "--data-dir", DATA_DIR,
        "--output-dir", CKPT_DIR,
    ]
    if max_train_steps is not None:
        client_cmd += ["--max-train-steps", str(max_train_steps)]
    if max_eval_steps is not None:
        client_cmd += ["--max-eval-steps", str(max_eval_steps)]

    print(">>> [modal] client command:", " ".join(shlex.quote(c) for c in client_cmd))

    try:
        client = subprocess.run(client_cmd, cwd=REMOTE_REPO, env=env)
        exit_code = client.returncode
    finally:
        _kill_server()
        ckpt_volume.commit()

    if exit_code != 0:
        sys.exit(exit_code)

    print(">>> [modal] PPO client finished successfully")


@app.local_entrypoint()
def main(
    max_train_steps: int | None = None,
    max_eval_steps: int | None = None,
    skip_data_prep: bool = False,
    backend_config: str | None = None,
) -> None:
    run_ppo.remote(
        max_train_steps=max_train_steps,
        max_eval_steps=max_eval_steps,
        skip_data_prep=skip_data_prep,
        backend_config=backend_config,
    )
