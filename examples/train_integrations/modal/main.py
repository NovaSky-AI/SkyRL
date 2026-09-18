import os
from pathlib import Path

import modal

_ISOEXEC_FA3_SOURCE = (
    "git+https://github.com/Dao-AILab/flash-attention.git@"
    "e81a70ce92d6dd96e213c08adbadd7c7e3d17f01#subdirectory=hopper"
)
_ISOEXEC_FA3_WHEEL_DIR = "/opt/isoexec-wheels"
_ISOEXEC_ONLY = os.environ.get("MODAL_ISOEXEC_ONLY") == "1"
_ISOEXEC_FA3_BUILD_ENV = {
    "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
    "FLASH_ATTENTION_DISABLE_SM80": "TRUE",
    "FLASH_ATTENTION_DISABLE_FP16": "TRUE",
    "FLASH_ATTENTION_DISABLE_FP8": "TRUE",
    "FLASH_ATTENTION_DISABLE_LOCAL": "TRUE",
    "FLASH_ATTENTION_DISABLE_SOFTCAP": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM64": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM96": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM128": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM192": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIMDIFF64": "TRUE",
    "FLASH_ATTENTION_DISABLE_HDIMDIFF192": "TRUE",
    "MAX_JOBS": "8",
    "NVCC_THREADS": "2",
}


def _find_local_repo_root() -> Path:
    """Computes full path of local SkyRL repo robustly

    Raises:
        Exception: if cannot find local SkyRL repo

    Returns:
        Path: path object describing full path of local SkyRL repo
    """
    # If running inside Modal container, use the environment variable
    if "SKYRL_REPO_ROOT" in os.environ:
        return Path(os.environ["SKYRL_REPO_ROOT"])

    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if base.exists() and (base / "skyrl-gym").exists():
                return base
    raise Exception("SkyRL root repo path not found")


def create_modal_image() -> modal.Image:
    """Creates a Modal image for Modal container. This uses the SkyRL container as
    a base image. It also mounts the local SkyRL repo to the container

    Returns:
        modal.Image: container image
    """

    local_repo_path = _find_local_repo_root()
    print(f"Root path: {local_repo_path}")

    envs = {
        "SKYRL_REPO_ROOT": "/root/SkyRL",  # where to put SkyRL in container
    }

    return (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0")
        .env(envs)
        .add_local_dir(
            local_path=str(local_repo_path),
            remote_path="/root/SkyRL",
            ignore=[
                ".venv",
                "*.pyc",
                "__pycache__",
                ".git",
                "*.egg-info",
                ".pytest_cache",
                "node_modules",
                ".DS_Store",
            ],
        )
    )


def create_isoexec_modal_image() -> modal.Image:
    """Create the opt-in Megatron image with both local repositories mounted.

    The normal ``run_script`` image remains unchanged. A missing sibling IsoExec
    checkout affects only the dedicated combined smoke function.
    """
    skyrl_root = _find_local_repo_root()
    isoexec_root = skyrl_root.parent / "IsoExec"
    image = (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron")
        .env({"SKYRL_REPO_ROOT": "/root/SkyRL", "ISOEXEC_REPO_ROOT": "/root/IsoExec"})
        .pip_install("huggingface_hub")
    )
    image = image.run_commands(
        f"mkdir -p {_ISOEXEC_FA3_WHEEL_DIR}",
        "/home/ray/anaconda3/bin/python -m pip wheel --no-deps --no-build-isolation "
        f"--wheel-dir {_ISOEXEC_FA3_WHEEL_DIR} '{_ISOEXEC_FA3_SOURCE}'",
        env=_ISOEXEC_FA3_BUILD_ENV,
    )
    image = image.add_local_dir(
        local_path=str(skyrl_root),
        remote_path="/root/SkyRL",
        ignore=[
            ".venv",
            "*.pyc",
            "__pycache__",
            ".git",
            "*.egg-info",
            ".pytest_cache",
            ".ruff_cache",
            ".claude",
            ".github",
            "docs",
            "skyrl-agent",
            "skyrl-tx",
            "tests",
        ],
    )
    if isoexec_root.is_dir():
        image = image.add_local_dir(
            local_path=str(isoexec_root),
            remote_path="/root/IsoExec",
            ignore=[".venv", "*.pyc", "__pycache__", ".git", "*.egg-info", ".pytest_cache", ".ruff_cache"],
        )
    return image


def create_modal_volume(volume_name: str = "skyrl-data") -> dict[str, modal.Volume]:
    """Creates volume to attach to container.

    Args:
        volume_name (str, optional): Name of volume. Creates a new
        volume if given name does not exist. Defaults to "skyrl-data".

    Returns:
        dict[str, modal.Volume]: location in container to attach the volume & volume itself
    """
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return {"/root/data": data_volume}  # mounts volume at /root/data inside container


app = modal.App(os.getenv("MODAL_APP_NAME", "my_skyrl_app"))
isoexec_image = create_isoexec_modal_image() if _ISOEXEC_ONLY else None
image = isoexec_image if isoexec_image is not None else create_modal_image()
volume = create_modal_volume()


@app.function(
    image=image,
    gpu=os.environ.get("MODAL_GPU", "L4:1"),
    volumes=volume,
    timeout=3600,  # 1 hour
)
def run_script(command: str):
    """
    Runs COMMAND inside SkyRL/
    """
    import os
    import subprocess

    # The repo root is already set in the image environment
    repo_root = os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL")

    # Print current environment for debugging
    print(f"Container repo root: {repo_root}")
    print(f"Initial working directory: {os.getcwd()}")

    # Change to the root directory
    run_command_dir = os.path.join(repo_root)
    os.chdir(run_command_dir)
    print(f"Changed to directory: {os.getcwd()}")

    # Ensure skyrl-gym exists inside working_dir so uv can resolve editable path
    gym_src = os.path.join("..", "skyrl-gym")
    gym_dst = os.path.join(".", "skyrl-gym")
    if not os.path.exists(gym_dst):
        if os.path.exists(gym_src):
            print("Copying ../skyrl-gym into working_dir for uv packaging")
            subprocess.run(
                f"cp -r {gym_src} {gym_dst}",
                shell=True,
                check=True,
            )
        else:
            raise Exception("Cannot find skyrl-gym source")

    print("Initializing ray cluster in command line")
    subprocess.run(
        "ray start --head",
        shell=True,
        check=True,
    )
    # Use 'auto' to automatically detect the Ray cluster instead of hardcoded IP
    os.environ["RAY_ADDRESS"] = "auto"

    print(f"Running command: {command}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)

    # Run the command with live output streaming
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
        universal_newlines=True,
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end="")  # Print each line as it comes

    # Wait for process to complete
    returncode = process.wait()

    print("=" * 60)
    if returncode != 0:
        raise Exception(f"Command failed with exit code {returncode}")


def isoexec_full_distribution_command(
    python: str = "/tmp/skyrl-isoexec-env/bin/python",
) -> list[str]:
    """Return the bounded one-step command used by the combined local-repo smoke."""
    engine_kwargs = '{"max_model_len":256,"max_num_seqs":4,' '"compilation_config":{"mode":0,"cudagraph_mode":"NONE"}}'
    overrides = [
        "data.train_data=['/root/data/gsm8k/train.parquet']",
        "data.val_data=['/root/data/gsm8k/validation.parquet']",
        "trainer.enable_isoexec=true",
        "trainer.rollout_logprob_comparison=full",
        "generator.inference_engine.logprob_output=full",
        "trainer.policy.model.path=/root/data/models/Qwen3.5-0.8B",
        "trainer.policy.language_model_only=true",
        "trainer.ref.language_model_only=true",
        "trainer.strategy=megatron",
        "trainer.bf16=true",
        "trainer.remove_microbatch_padding=true",
        "trainer.fused_lm_head_logprob=false",
        "trainer.mtp.enabled=false",
        "trainer.placement.colocate_all=true",
        "trainer.placement.policy_num_nodes=1",
        "trainer.placement.policy_num_gpus_per_node=1",
        "trainer.placement.ref_num_gpus_per_node=1",
        "trainer.policy.megatron_config.tensor_model_parallel_size=1",
        "trainer.policy.megatron_config.pipeline_model_parallel_size=1",
        "trainer.policy.megatron_config.context_parallel_size=1",
        "trainer.policy.megatron_config.expert_model_parallel_size=1",
        "trainer.policy.megatron_config.expert_tensor_parallel_size=1",
        "trainer.epochs=1",
        "trainer.max_training_steps=1",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=-1",
        "trainer.update_epochs_per_batch=1",
        "trainer.train_batch_size=1",
        "trainer.policy_mini_batch_size=1",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.ckpt_interval=-1",
        "trainer.hf_save_interval=-1",
        "trainer.max_prompt_length=128",
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.temperature=1.0",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.algorithm.dynamic_sampling.type=null",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        "trainer.enable_ray_gpu_monitor=false",
        "trainer.print_example_interval=0",
        "trainer.dump_eval_results=false",
        "trainer.logger=console",
        "trainer.resume_mode=null",
        "trainer.log_path=/root/data/logs",
        "trainer.ckpt_path=/root/data/checkpoints",
        "trainer.export_path=/root/data/exports",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.language_model_only=true",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.num_engines=1",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.pipeline_parallel_size=1",
        "generator.inference_engine.expert_parallel_size=1",
        "generator.inference_engine.data_parallel_size=1",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.gpu_memory_utilization=0.5",
        "generator.inference_engine.max_num_batched_tokens=256",
        "generator.inference_engine.enable_prefix_caching=false",
        "generator.inference_engine.enable_chunked_prefill=true",
        "generator.inference_engine.enforce_eager=true",
        f"generator.inference_engine.engine_init_kwargs={engine_kwargs}",
        "generator.batched=true",
        "generator.max_turns=1",
        "generator.n_samples_per_prompt=2",
        "generator.sampling_params.max_generate_length=2",
        "generator.sampling_params.temperature=1.0",
        "generator.sampling_params.logprobs=1",
        "environment.env_class=gsm8k",
    ]
    return [
        python,
        "-m",
        "skyrl.train.entrypoints.main_base",
        *overrides,
    ]


def _run_isoexec_full_distribution_impl():
    """Run one combined local SkyRL/IsoExec update and require full-row evidence."""
    import subprocess
    import threading

    from huggingface_hub import snapshot_download

    skyrl_root = os.environ["SKYRL_REPO_ROOT"]
    isoexec_root = os.environ["ISOEXEC_REPO_ROOT"]
    if not os.path.isdir(os.path.join(isoexec_root, "isoexec")):
        raise RuntimeError("Local sibling IsoExec checkout was not mounted; clone it next to SkyRL")
    fa3_wheels = sorted(Path(_ISOEXEC_FA3_WHEEL_DIR).glob("flash_attn_3-*.whl"))
    if len(fa3_wheels) != 1:
        raise RuntimeError(f"expected one cached FA3 wheel, found {fa3_wheels}")

    env = dict(os.environ)
    inherited_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(value for value in (isoexec_root, skyrl_root, inherited_pythonpath) if value)
    env["RAY_ADDRESS"] = "auto"
    env["SKYRL_DUMP_INFRA_LOG_TO_STDOUT"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    venv = Path("/tmp/skyrl-isoexec-env")
    env["UV_PROJECT_ENVIRONMENT"] = str(venv)
    subprocess.run(["uv", "sync", "--frozen", "--extra", "megatron"], cwd=skyrl_root, check=True, env=env)
    python = str(venv / "bin/python")
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            python,
            "--reinstall",
            "--no-deps",
            "nvidia-cutlass-dsl==4.5.2",
            "nvidia-cutlass-dsl-libs-base==4.5.2",
            "nvidia-cutlass-dsl-libs-cu13==4.5.2",
        ],
        cwd=skyrl_root,
        check=True,
        env=env,
    )
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-config",
            "--python",
            python,
            "--reinstall",
            "--no-deps",
            str(fa3_wheels[0]),
        ],
        cwd=skyrl_root,
        check=True,
        env=env,
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", python, "--no-deps", "--editable", isoexec_root],
        cwd=skyrl_root,
        check=True,
        env=env,
    )
    subprocess.run(
        [
            python,
            "-c",
            "from importlib.metadata import version; import flash_attn_interface, numba, numpy, torch, torchvision; "
            "cutlass = version('nvidia-cutlass-dsl'); "
            "print(torch.__version__, torchvision.__version__, cutlass, numpy.__version__); "
            "assert cutlass == '4.5.2', cutlass",
        ],
        cwd=skyrl_root,
        check=True,
        env=env,
    )
    subprocess.run([str(venv / "bin/ray"), "start", "--head"], check=True, env=env)

    model_path = Path("/root/data/models/Qwen3.5-0.8B")
    if not (model_path / "config.json").is_file():
        snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=model_path)
    train_path = Path("/root/data/gsm8k/train.parquet")
    if not train_path.is_file():
        subprocess.run(
            [
                python,
                "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir",
                "/root/data/gsm8k",
                "--max_train_dataset_length",
                "1",
            ],
            cwd=skyrl_root,
            check=True,
            env=env,
        )
    volume["/root/data"].commit()

    process = subprocess.Popen(
        isoexec_full_distribution_command(python),
        cwd=skyrl_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output: list[str] = []

    def stream_output() -> None:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)

    reader = threading.Thread(target=stream_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        returncode = process.wait(timeout=1500)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.terminate()
        try:
            returncode = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = process.wait()
    reader.join(timeout=30)
    stdout = "".join(output)
    Path("/root/data/isoexec-full-distribution.log").write_text(stdout)
    volume["/root/data"].commit()
    if timed_out:
        raise RuntimeError("combined SkyRL/IsoExec smoke exceeded its 25-minute training timeout")
    if returncode:
        raise RuntimeError(f"combined SkyRL/IsoExec smoke failed with exit code {returncode}")
    required = ("[ISOEXEC-", "policy/full_logprobs_verified_rows", "Training done!")
    missing = [marker for marker in required if marker not in stdout]
    if missing:
        raise RuntimeError(f"combined smoke exited zero but missed required evidence: {missing}")


if isoexec_image is not None:
    run_isoexec_full_distribution = app.function(
        image=isoexec_image,
        gpu=os.environ.get("MODAL_ISOEXEC_GPU", "H100!"),
        volumes=volume,
        retries=0,
        timeout=1800,
        name="run_isoexec_full_distribution",
    )(_run_isoexec_full_distribution_impl)
else:
    run_isoexec_full_distribution = None


@app.local_entrypoint()
def main(command: str = "nvidia-smi", isoexec_full_distribution: bool = False):
    """Main entry-point for running a command in Modal-integrated SkyRL environmenmt.
    The given command will be run inside SkyRL/

    Args:
        command (str, optional): Command to run. Defaults to "nvidia-smi".
        isoexec_full_distribution: Run the bounded combined local SkyRL/IsoExec
            one-step smoke instead of ``command``. Defaults to False.

    Examples:
        modal run main.py --command "uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k"
        MODAL_GPU=A100:4 MODAL_APP_NAME=benji_skyrl_app modal run main.py --command "bash examples/train/gsm8k/run_gsm8k_modal.sh"
        MODAL_ISOEXEC_ONLY=1 MODAL_ISOEXEC_GPU='H100!' modal run main.py --isoexec-full-distribution
    """
    if isoexec_full_distribution:
        if run_isoexec_full_distribution is None:
            raise RuntimeError(
                "Set MODAL_ISOEXEC_ONLY=1 before --isoexec-full-distribution so the bounded combined image is built"
            )
        print(f"{'=' * 5} Submitting combined local SkyRL/IsoExec full-distribution smoke {'=' * 5}")
        run_isoexec_full_distribution.remote()
    else:
        print(f"{'=' * 5} Submitting command to Modal: {command} {'=' * 5}")
        run_script.remote(command)
    print(f"\n{'=' * 5} Command completed successfully {'=' * 5}")
