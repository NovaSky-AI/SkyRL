"""
Modal entrypoint that runs the Tinker PPO example and/or the standard PPO script.

What this can do from one local Modal entrypoint:
    1. Preprocess GSM8K into ~/data/gsm8k/{train,validation}.parquet
       (unless already present in the persistent Volume).
    2. Launch the Tinker PPO flow in one Modal job.
    3. Launch the standard `examples/train/ppo/run_ppo.sh` flow in another Modal job.
    4. Submit either job alone, or both jobs together for W&B comparison.

The Tinker server and client share a single container when running the Tinker
path, so the client can keep the default `--base-url http://localhost:8000`.
Each training job requests four GPUs because both launchers assume the same
colocated 4-GPU FSDP/vLLM layout with policy and critic training.

Usage (from the repo root):

    # launch both runs as separate Modal jobs
    modal run examples/tinker/ppo/modal_run.py

    # launch only the Tinker path
    modal run examples/tinker/ppo/modal_run.py --experiment tinker

    # launch only the standard PPO path
    modal run examples/tinker/ppo/modal_run.py --experiment standard

    # override Tinker client limits
    modal run examples/tinker/ppo/modal_run.py \
        --experiment tinker --max-train-steps 2 --max-eval-steps 1

    # pass extra Hydra overrides to the standard PPO launcher
    modal run examples/tinker/ppo/modal_run.py \
        --experiment standard \
        --standard-extra-args "trainer.epochs=1 trainer.eval_interval=1"

The Tinker client uses TINKER_API_KEY=tml-dummy, matching the bundled SkyRL
Tinker server's default (see the NovaSky blog post and run_tinker_server.sh).
"""

from __future__ import annotations

import os
import pathlib
import shlex

import modal


REMOTE_REPO = "/root/SkyRL"
DATA_DIR = "/root/data/gsm8k"
TINKER_CKPT_DIR = "/root/ckpts/gsm8k_1.5B_ckpt_ppo_tinker"
STANDARD_CKPT_DIR = "/root/ckpts/gsm8k_1.5B_ckpt_ppo_standard"
HF_CACHE = "/root/.cache/huggingface"
DEFAULT_WANDB_PROJECT = "gsm8k"
DEFAULT_TINKER_WANDB_RUN_NAME = "gsm8k_tinker_ppo"
DEFAULT_STANDARD_WANDB_RUN_NAME = "gsm8k_ppo_modal"


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
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
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


def _build_job_env(
    default_run_name: str,
    specific_run_name_key: str,
    *,
    wandb_api_key: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_tags: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = "/root"
    env["WANDB_PROJECT"] = wandb_project or os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    env["WANDB_RUN_NAME"] = os.environ.get(specific_run_name_key, default_run_name)
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_tags:
        env["WANDB_TAGS"] = wandb_tags
    return env


def _prepare_gsm8k_data(env: dict[str, str], skip_data_prep: bool) -> None:
    import subprocess

    train_parquet = os.path.join(DATA_DIR, "train.parquet")
    val_parquet = os.path.join(DATA_DIR, "validation.parquet")
    data_ready = os.path.exists(train_parquet) and os.path.exists(val_parquet)

    if not data_ready and not skip_data_prep:
        print(">>> [modal] Preparing GSM8K parquet files")
        subprocess.run(
            [
                "uv",
                "run",
                "--extra",
                "tinker",
                "python",
                "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir",
                DATA_DIR,
            ],
            cwd=REMOTE_REPO,
            env=env,
            check=True,
        )
        data_volume.commit()
    elif data_ready:
        print(">>> [modal] GSM8K parquet files already present; skipping prep")
    else:
        print(">>> [modal] --skip-data-prep set but dataset missing; the run will fail")


@app.function(
    image=image,
    timeout=2 * 60 * 60,
    volumes={
        "/root/data": data_volume,
        HF_CACHE: hf_volume,
    },
)
def prepare_gsm8k_data(
    skip_data_prep: bool = False,
    wandb_api_key: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_tags: str | None = None,
) -> None:
    import os

    os.makedirs(DATA_DIR, exist_ok=True)
    env = _build_job_env(
        DEFAULT_TINKER_WANDB_RUN_NAME,
        "WANDB_RUN_NAME_TINKER",
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
    )
    _prepare_gsm8k_data(env, skip_data_prep)


def _log_wandb_config(job_name: str, env: dict[str, str]) -> None:
    api_key_present = bool(env.get("WANDB_API_KEY"))
    print(
        ">>> [modal] "
        f"{job_name} wandb config: enabled={api_key_present} "
        f"project={env.get('WANDB_PROJECT')} "
        f"run_name={env.get('WANDB_RUN_NAME')} "
        f"entity={env.get('WANDB_ENTITY')}"
    )
    if not api_key_present:
        print(f">>> [modal] {job_name} wandb logging is disabled because WANDB_API_KEY is not set")


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
def run_tinker_ppo(
    max_train_steps: int | None = None,
    max_eval_steps: int | None = None,
    skip_data_prep: bool = False,
    backend_config: str | None = None,
    wandb_api_key: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_tags: str | None = None,
) -> None:
    from collections import deque
    import os
    import signal
    import subprocess
    import threading
    import time
    import urllib.error
    import urllib.request

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TINKER_CKPT_DIR, exist_ok=True)

    env = _build_job_env(
        DEFAULT_TINKER_WANDB_RUN_NAME,
        "WANDB_RUN_NAME_TINKER",
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
    )
    # Matches the SkyRL Tinker README / NovaSky blog post: the bundled server
    # accepts the literal "tml-dummy" key, no Modal Secret needed.
    env["TINKER_API_KEY"] = "tml-dummy"
    if backend_config:
        env["BACKEND_CONFIG"] = backend_config

    _log_wandb_config("tinker", env)
    _prepare_gsm8k_data(env, skip_data_prep)

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

    threading.Thread(target=_pump_logs, args=(server.stdout, "server"), daemon=True).start()

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
            raise RuntimeError(f"Tinker server exited early with code {server.returncode}")
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
        "uv",
        "run",
        "--extra",
        "tinker",
        "python",
        "examples/tinker/ppo/ppo_client.py",
        "--base-url",
        base_url,
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        TINKER_CKPT_DIR,
    ]
    if max_train_steps is not None:
        client_cmd += ["--max-train-steps", str(max_train_steps)]
    if max_eval_steps is not None:
        client_cmd += ["--max-eval-steps", str(max_eval_steps)]

    print(">>> [modal] client command:", " ".join(shlex.quote(c) for c in client_cmd))

    client_log_tail: deque[str] = deque(maxlen=80)
    try:
        client = subprocess.Popen(
            client_cmd,
            cwd=REMOTE_REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert client.stdout is not None
        for line in client.stdout:
            print(f"[client] {line}", end="")
            client_log_tail.append(line.rstrip())
        exit_code = client.wait()
    finally:
        _kill_server()
        ckpt_volume.commit()

    if exit_code != 0:
        tail = "\n".join(client_log_tail).strip()
        if tail:
            raise RuntimeError(f"Tinker PPO client exited with code {exit_code}. " f"Last client log lines:\n{tail}")
        raise RuntimeError(
            f"Tinker PPO client exited with code {exit_code}. " "No client log lines were captured before exit."
        )

    print(">>> [modal] PPO client finished successfully")


@app.function(
    image=image,
    gpu="H100:4",
    timeout=24 * 60 * 60,
    volumes={
        "/root/data": data_volume,
        "/root/ckpts": ckpt_volume,
        HF_CACHE: hf_volume,
    },
)
def run_standard_ppo(
    skip_data_prep: bool = False,
    standard_extra_args: str = "",
    wandb_api_key: str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_tags: str | None = None,
) -> None:
    import os
    import subprocess
    import threading

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(STANDARD_CKPT_DIR, exist_ok=True)

    env = _build_job_env(
        DEFAULT_STANDARD_WANDB_RUN_NAME,
        "WANDB_RUN_NAME_STANDARD",
        wandb_api_key=wandb_api_key,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_tags=wandb_tags,
    )
    _log_wandb_config("standard", env)
    _prepare_gsm8k_data(env, skip_data_prep)

    cmd: list[str] = [
        "bash",
        "examples/train/ppo/run_ppo.sh",
        f"trainer.project_name={env['WANDB_PROJECT']}",
        f"trainer.run_name={env['WANDB_RUN_NAME']}",
        f"trainer.ckpt_path={STANDARD_CKPT_DIR}",
    ]
    if standard_extra_args.strip():
        cmd.extend(shlex.split(standard_extra_args))

    print(">>> [modal] standard command:", " ".join(shlex.quote(c) for c in cmd))
    print(">>> [modal] Launching standard PPO via examples/train/ppo/run_ppo.sh")
    process = subprocess.Popen(
        cmd,
        cwd=REMOTE_REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def _pump_logs(stream, prefix: str) -> None:
        for line in stream:
            print(f"[{prefix}] {line}", end="")

    threading.Thread(target=_pump_logs, args=(process.stdout, "standard"), daemon=True).start()
    result = process.wait()
    ckpt_volume.commit()

    if result != 0:
        raise RuntimeError(f"Standard PPO run exited with code {result}")

    print(">>> [modal] Standard PPO run finished successfully")


@app.local_entrypoint()
def main(
    experiment: str = "both",
    max_train_steps: int | None = None,
    max_eval_steps: int | None = None,
    skip_data_prep: bool = False,
    backend_config: str | None = None,
    standard_extra_args: str = "",
    parallel: bool = True,
) -> None:
    if experiment not in {"tinker", "standard", "both"}:
        raise ValueError("experiment must be one of: tinker, standard, both")

    local_wandb_api_key = os.environ.get("WANDB_API_KEY")
    local_wandb_entity = os.environ.get("WANDB_ENTITY")
    local_wandb_project = os.environ.get("WANDB_PROJECT")
    local_wandb_tags = os.environ.get("WANDB_TAGS")

    if experiment in {"standard", "both"} and not local_wandb_api_key:
        raise RuntimeError(
            "WANDB_API_KEY must be set before launching the standard PPO run so both jobs can log to the same W&B project."
        )

    if experiment == "both":
        print(">>> [local] Preparing GSM8K data once before launching both jobs")
        prepare_gsm8k_data.remote(
            skip_data_prep=skip_data_prep,
            wandb_api_key=local_wandb_api_key,
            wandb_entity=local_wandb_entity,
            wandb_project=local_wandb_project,
            wandb_tags=local_wandb_tags,
        )
        skip_data_prep = True

    jobs: list[tuple[str, object]] = []
    can_spawn_tinker = hasattr(run_tinker_ppo, "spawn")
    can_spawn_standard = hasattr(run_standard_ppo, "spawn")

    if experiment == "both" and parallel and (not can_spawn_tinker or not can_spawn_standard):
        raise RuntimeError(
            "This Modal installation does not expose parallel job submission for both runs. "
            "Upgrade Modal or run the two experiments separately."
        )

    if experiment in {"tinker", "both"}:
        tinker_kwargs = {
            "max_train_steps": max_train_steps,
            "max_eval_steps": max_eval_steps,
            "skip_data_prep": skip_data_prep,
            "backend_config": backend_config,
            "wandb_api_key": local_wandb_api_key,
            "wandb_entity": local_wandb_entity,
            "wandb_project": local_wandb_project,
            "wandb_tags": local_wandb_tags,
        }
        if experiment == "both" and parallel:
            print(">>> [local] Submitting Tinker PPO Modal job")
            jobs.append(("tinker", run_tinker_ppo.spawn(**tinker_kwargs)))
        else:
            print(">>> [local] Running Tinker PPO Modal job")
            run_tinker_ppo.remote(**tinker_kwargs)

    if experiment in {"standard", "both"}:
        standard_kwargs = {
            "skip_data_prep": skip_data_prep,
            "standard_extra_args": standard_extra_args,
            "wandb_api_key": local_wandb_api_key,
            "wandb_entity": local_wandb_entity,
            "wandb_project": local_wandb_project,
            "wandb_tags": local_wandb_tags,
        }
        if experiment == "both" and parallel:
            print(">>> [local] Submitting standard PPO Modal job")
            jobs.append(("standard", run_standard_ppo.spawn(**standard_kwargs)))
        else:
            print(">>> [local] Running standard PPO Modal job")
            run_standard_ppo.remote(**standard_kwargs)

    if jobs:
        for label, _ in jobs:
            print(f">>> [local] Waiting for {label} job")
        for _, job in jobs:
            job.get()
