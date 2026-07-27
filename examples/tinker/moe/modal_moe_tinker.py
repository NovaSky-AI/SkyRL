"""Serve an MoE model behind SkyRL's Tinker API server on Modal, and submit client jobs to it.


demo -- one gpu container runs tinker server + gsm8k client 

serve -- long lived gpu container running tinker server, exposed through modal tunnel 

client -- cpu only container that runs the client against server's modal url 

engine-test -- isolation test: boot ONE plain vLLM server (no Tinker, no megatron, empty GPUs) and time it to first /health 200. Distinguishes 'vLLM+model is slow/stuck' from 'the colocated Tinker path breaks engine boot'

Qwen3-30B-a3B-base recipe: TP2/EP8/ETP1 for training, 2 vllm engines x TP4 for inference

Usage (from the SkyRL repo root):
    # long-lived server; prints the tunnel URL, stays up until you stop it
    modal run examples/tinker/moe/modal_moe_tinker.py --mode serve

    # then submit client job against running server 
    modal run examples/tinker/moe/modal_moe_tinker.py --mode client --base-url <tunnel-url>
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex
from typing import Optional

import modal

REMOTE_REPO = "/root/SkyRL"
HF_CACHE = "/root/.cache/huggingface"
CKPT_DIR = "/root/ckpts/tinker_moe"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"

DEFAULT_BASE_MODEL = "Qwen/Qwen3-30B-A3B-Base"
GPU_SPEC = os.environ.get("SKYRL_TINKER_GPU", "H200:8")
NUM_GPUS = int(GPU_SPEC.split(":")[1]) if ":" in GPU_SPEC else 1


DEFAULT_BACKEND_CONFIG: dict = {
    "trainer.placement.colocate_all": True,
    "trainer.placement.policy_num_gpus_per_node": NUM_GPUS,

    "trainer.policy.megatron_config.tensor_model_parallel_size": 2,
    "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
    "trainer.policy.megatron_config.expert_model_parallel_size": NUM_GPUS,
    "trainer.policy.megatron_config.expert_tensor_parallel_size": 1,
    "trainer.micro_forward_batch_size_per_gpu": 2,
    "trainer.micro_train_batch_size_per_gpu": 1,

    "generator.inference_engine.num_engines": 2,
    "generator.inference_engine.tensor_parallel_size": NUM_GPUS // 2,
    "generator.inference_engine.backend": "vllm",
    "generator.inference_engine.run_engines_locally": True,
    "generator.inference_engine.weight_sync_backend": "nccl",

    "generator.inference_engine.gpu_memory_utilization": 0.5,
    "generator.batched": True,

    "trainer.log_path": f"{CKPT_DIR}/logs",
}


def _is_repo_root(path: pathlib.Path) -> bool:
    try:
        return (
            path.exists()
            and (path / "pyproject.toml").exists()
            and (path / "examples").exists()
            and (path / "skyrl").exists()
            and (path / "skyrl-gym").exists()
        )
    except OSError:
        # e.g. PermissionError probing /root/SkyRL when running locally as a
        # non-root user (Path.exists() raises on EACCES rather than returning
        # False on older Pythons).
        return False


def _find_repo_root() -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    env_repo_root = os.environ.get("SKYRL_REPO_ROOT")
    if env_repo_root:
        candidates.append(pathlib.Path(env_repo_root))
    candidates.append(pathlib.Path(REMOTE_REPO))
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        base = start if start.is_dir() else start.parent
        candidates.extend([base, *base.parents])
    for candidate in dict.fromkeys(candidates):
        if _is_repo_root(candidate):
            return candidate
    raise RuntimeError(
        "Could not locate the SkyRL repo root. Set SKYRL_REPO_ROOT or run from inside the repository."
    )


REPO_ROOT = _find_repo_root()

hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)
ckpt_volume = modal.Volume.from_name("skyrl-tinker-moe-ckpts", create_if_missing=True)

wandb_secret = modal.Secret.from_name(os.environ.get("SKYRL_WANDB_SECRET", "wandb-secret"))


VENV_DIR = "/root/venvs/skyrl"


image = (
    modal.Image.from_registry("novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SKYRL_REPO_ROOT": REMOTE_REPO,
            "UV_LINK_MODE": "copy",
        }
    )

    .add_local_file(f"{REPO_ROOT}/pyproject.toml", f"{REMOTE_REPO}/pyproject.toml", copy=True)
    .add_local_file(f"{REPO_ROOT}/uv.lock", f"{REMOTE_REPO}/uv.lock", copy=True)
    .add_local_file(f"{REPO_ROOT}/skyrl-gym/pyproject.toml", f"{REMOTE_REPO}/skyrl-gym/pyproject.toml", copy=True)
    .workdir(REMOTE_REPO)
    .run_commands(
        f"UV_PROJECT_ENVIRONMENT={VENV_DIR} uv sync --frozen --extra tinker --extra megatron "
        "--no-install-project --no-install-package skyrl-gym",
        gpu="any",
    )
    # --- Code layer: runtime mount, no rebuild on edits.
    .add_local_dir(
        str(REPO_ROOT),
        REMOTE_REPO,
        ignore=[".venv", "**/__pycache__", ".git", "*.egg-info", ".pytest_cache"],
    )
)

app = modal.App("skyrl-tinker-moe")


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------


def _pump_logs(stream, prefix: str) -> None:
    for line in stream:
        print(f"[{prefix}] {line}", end="")


def _start_server(env: dict, base_model: str, backend_config: dict):
    import subprocess
    import threading

    # Sync the venv exactly ONCE per container (installs the editable
    # skyrl/skyrl-gym from the runtime mount and any packages the baked layer
    # is missing), then run everything with UV_NO_SYNC=1. Without this, the
    # engine child process and every Ray worker actor re-run `uv run` against
    # the SAME shared venv concurrently; their syncs flip-flop the editable
    # install path (repo mount vs ray working_dir), and an actor importing
    # transformer-engine during a sibling's uninstall->install window dies
    # with "Found empty `transformer-engine` meta package installed".
    sync_cmd = ["uv", "sync", "--frozen", "--extra", "tinker", "--extra", "megatron"]
    print(">>> [modal] pre-sync command:", " ".join(shlex.quote(c) for c in sync_cmd))
    subprocess.run(sync_cmd, cwd=REMOTE_REPO, env=env, check=True)
    env = {**env, "UV_NO_SYNC": "1"}

    # NOT --isolated: the server must run in the stable UV_PROJECT_ENVIRONMENT
    # venv so Ray workers' uv re-invocations resolve to the same environment.
    cmd = [
        "uv",
        "run",
        "--extra",
        "tinker",
        "--extra",
        "megatron",
        "-m",
        "skyrl.tinker.api",
        "--base-model",
        base_model,
        "--backend",
        "megatron",
        "--port",
        str(PORT),
        "--backend-config",
        json.dumps(backend_config),
        "--checkpoints-base",
        CKPT_DIR,
    ]
    print(">>> [modal] server command:", " ".join(shlex.quote(c) for c in cmd))
    server = subprocess.Popen(
        cmd,
        cwd=REMOTE_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    threading.Thread(target=_pump_logs, args=(server.stdout, "server"), daemon=True).start()
    return server


def _kill_server(server) -> None:
    import signal
    import subprocess

    if server.poll() is not None:
        return
    print(">>> [modal] Stopping Tinker API server")
    try:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        server.wait(timeout=30)


def _wait_ready(server, timeout_s: int = 45 * 60) -> None:
    """Poll /api/v1/healthz (all Tinker routes live under /api/v1) until the
    server answers 200. The API answers quickly; the heavy model init happens
    lazily on the first create_model/sample request."""
    import time
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Tinker server exited early with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/v1/healthz", timeout=5) as resp:
                if resp.status == 200:
                    print(">>> [modal] Tinker API server is ready")
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(5)
    raise RuntimeError(f"Tinker server not ready within {timeout_s}s")


def _merged_backend_config(overrides_json: Optional[str]) -> dict:
    cfg = dict(DEFAULT_BACKEND_CONFIG)
    if overrides_json:
        cfg.update(json.loads(overrides_json))
    return cfg


def _server_env() -> dict:
    env = os.environ.copy()
    env["HOME"] = "/root"
    env["TINKER_API_KEY"] = "tml-dummy"
    env["UV_PROJECT_ENVIRONMENT"] = VENV_DIR
    env.setdefault("SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S", "1800")

    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    env.setdefault("RAY_DEDUP_LOGS", "0")
    return env


def _run_client(
    base_url: str,
    base_model: str,
    client_args: str,
    metrics_path: Optional[str] = None,
) -> None:
    import subprocess


    env = os.environ.copy()
    env["HOME"] = "/root"
    env["TINKER_API_KEY"] = "tml-dummy"
    env["PYTHONUNBUFFERED"] = "1"  # stream step prints live instead of block-buffering
    
    # run in baked venv 
    env["UV_PROJECT_ENVIRONMENT"] = VENV_DIR
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "--with",
        "datasets",
        "--with",
        "wandb",
        "python",
        "examples/tinker/moe/moe_smoke_client.py",
        "--base-url",
        base_url,
        "--base-model",
        base_model,
    ]
    if metrics_path and "--metrics-path" not in client_args:
        cmd += ["--metrics-path", metrics_path]
    if client_args.strip():
        cmd.extend(shlex.split(client_args))
    print(">>> [modal] client command:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=REMOTE_REPO, env=env, check=True)


# ---------------------------------------------------------------------------
# Modal functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=24 * 60 * 60,
    volumes={HF_CACHE: hf_volume, "/root/ckpts": ckpt_volume},
)
def serve(base_model: str = DEFAULT_BASE_MODEL, backend_config: Optional[str] = None) -> None:
    """Run the Tinker server behind a Modal tunnel until the job is stopped."""
    import time

    os.makedirs(CKPT_DIR, exist_ok=True)
    server = _start_server(_server_env(), base_model, _merged_backend_config(backend_config))
    try:
        # expose public server (todo: modal.web_server? For this it's overkill i think)
        with modal.forward(PORT) as tunnel:
            print(f">>> [modal] Tinker tunnel URL (may still be warming up): {tunnel.url}")
            _wait_ready(server)
            print(">>> [modal] ================================================")
            print(f">>> [modal] Tinker server ready: {tunnel.url}")
            print(">>> [modal]   TINKER_API_KEY=tml-dummy")
            print(f">>> [modal]   base_model={base_model}")
            print(">>> [modal] Stop with: modal app stop skyrl-tinker-moe")
            print(">>> [modal] ================================================")
            while server.poll() is None:
                time.sleep(10)
            raise RuntimeError(f"Tinker server exited with code {server.returncode}")
    finally:
        _kill_server(server)
        ckpt_volume.commit()


@app.function(image=image, timeout=24 * 60 * 60, secrets=[wandb_secret])  # CPU-only: HTTP calls
def client(base_url: str, base_model: str = DEFAULT_BASE_MODEL, client_args: str = "") -> None:
    """Submit the smoke client as its own Modal job against a running server."""
    _run_client(base_url, base_model, client_args)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    timeout=6 * 60 * 60,
    volumes={HF_CACHE: hf_volume, "/root/ckpts": ckpt_volume},
    secrets=[wandb_secret],
)
def demo(base_model: str = DEFAULT_BASE_MODEL, backend_config: Optional[str] = None, client_args: str = "") -> None:
    """Server + client in one container under two processes""" 
    os.makedirs(CKPT_DIR, exist_ok=True)
    server = _start_server(_server_env(), base_model, _merged_backend_config(backend_config))
    try:
        _wait_ready(server)
        # Metrics land on the ckpt volume so they survive the container.
        _run_client(BASE_URL, base_model, client_args, metrics_path=f"{CKPT_DIR}/moe_gsm8k_metrics.jsonl")
    finally:
        _kill_server(server)
        ckpt_volume.commit()
    print(">>> [modal] demo finished successfully")



@app.local_entrypoint()
def main(
    mode: str = "demo",
    base_model: str = DEFAULT_BASE_MODEL,
    backend_config: Optional[str] = None,
    base_url: Optional[str] = None,
    client_args: str = "",
) -> None:
    if mode == "demo":
        demo.remote(base_model=base_model, backend_config=backend_config, client_args=client_args)
    elif mode == "serve":
        serve.remote(base_model=base_model, backend_config=backend_config)
    elif mode == "client":
        if not base_url:
            raise ValueError("--mode client requires --base-url (the tunnel URL printed by --mode serve)")
        client.remote(base_url=base_url, base_model=base_model, client_args=client_args)
    else:
        raise ValueError("choose demo serve or client")
