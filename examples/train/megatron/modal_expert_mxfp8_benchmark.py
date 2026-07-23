"""Benchmark BF16 versus expert-only MXFP8 on one Modal node.

Run:
    uv run --isolated --with modal modal run --detach \
        examples/train/megatron/modal_expert_mxfp8_benchmark.py \
        --mode both --steps 10
"""

from __future__ import annotations

import os
import pathlib

import modal

APP_NAME = "skyrl-expert-mxfp8-benchmark"
MODEL = "Qwen/Qwen3-30B-A3B"
REMOTE_REPO = "/root/SkyRL"
HF_HOME = "/root/hf-cache"
DATA_DIR = "/root/data/gsm8k"
PROFILE_ROOT = "/root/profiles"  # profiler
PROFILE_COMMIT_INTERVAL_S = 120  # profiler
GPU = os.environ.get("MODAL_GPU", "B200:8")


def _repo_root() -> pathlib.Path:
    for start in (pathlib.Path(__file__).resolve(), pathlib.Path.cwd().resolve()):
        candidate = start if start.is_dir() else start.parent
        for path in (candidate, *candidate.parents):
            if (path / "pyproject.toml").exists() and (path / "skyrl").exists():
                return path
    raise RuntimeError("Run the benchmark from a SkyRL checkout")


repo_root = _repo_root()
hf_volume = modal.Volume.from_name("skyrl-hf-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("skyrl-expert-mxfp8-data", create_if_missing=True)
profile_volume = modal.Volume.from_name("skyrl-expert-mxfp8-profiles", create_if_missing=True)  # profiler

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential", "ca-certificates", "libnuma1", "numactl")
    .pip_install("huggingface-hub")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env(
        {
            "PATH": "/root/.local/bin:/usr/local/cuda/bin:${PATH}",
            "HF_HOME": HF_HOME,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "NVTE_FLASH_ATTN": "0",
            "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S": "1800",
            "SKYRL_DUMP_INFRA_LOG_TO_STDOUT": "1",
            "UV_LINK_MODE": "copy",
            "UV_PROJECT_ENVIRONMENT": f"{REMOTE_REPO}/.venv",
            "VLLM_USE_FLASHINFER_MOE_FP16": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(
        str(repo_root),
        REMOTE_REPO,
        copy=True,
        ignore=[".venv", ".git", "**/__pycache__"],
    )
    .workdir(REMOTE_REPO)
    .run_commands("uv sync --extra megatron", gpu="any")
    .run_commands(f"rm -rf {HF_HOME}")
)

app = modal.App(APP_NAME)


def _run(mode: str, steps: int) -> None:
    import subprocess
    import time

    train_file = f"{DATA_DIR}/train.parquet"
    val_file = f"{DATA_DIR}/validation.parquet"
    enabled = str(mode == "mxfp8").lower()
    run_name = f"qwen3-30b-a3b-{mode}"
    command = [
        "uv",
        "run",
        "--frozen",
        "--no-sync",
        "--extra",
        "megatron",
        "-m",
        "skyrl.train.entrypoints.main_base",
        f"data.train_data=['{train_file}']",
        f"data.val_data=['{val_file}']",
        "trainer.strategy=megatron",
        f"trainer.policy.model.path={MODEL}",
        f"trainer.policy.model.expert_mxfp8.enabled={enabled}",
        f"trainer.policy.model.expert_mxfp8.persistent={enabled}",
        f"trainer.policy.megatron_config.ddp_config.fp8_param_gather={enabled}",
        "trainer.placement.colocate_all=true",
        "trainer.placement.policy_num_nodes=1",
        "trainer.placement.policy_num_gpus_per_node=8",
        "generator.inference_engine.num_engines=1",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.data_parallel_size=8",
        "generator.inference_engine.expert_parallel_size=8",
        "generator.inference_engine.distributed_executor_backend=mp",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.gpu_memory_utilization=0.6",
        "trainer.policy.megatron_config.tensor_model_parallel_size=1",
        "trainer.policy.megatron_config.pipeline_model_parallel_size=1",
        "trainer.policy.megatron_config.context_parallel_size=1",
        "trainer.policy.megatron_config.expert_model_parallel_size=8",
        "trainer.policy.megatron_config.expert_tensor_parallel_size=1",
        "trainer.algorithm.advantage_estimator=grpo",
        "trainer.algorithm.use_kl_loss=false",
        "trainer.train_batch_size=64",
        "trainer.policy_mini_batch_size=64",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        "trainer.eval_before_train=false",
        "trainer.eval_interval=0",
        "trainer.ckpt_interval=0",
        "trainer.hf_save_interval=0",
        "trainer.resume_mode=none",
        f"trainer.max_training_steps={steps}",
        "trainer.max_prompt_length=512",
        "generator.sampling_params.max_generate_length=512",
        "generator.n_samples_per_prompt=8",
        "generator.batched=true",
        "environment.env_class=gsm8k",
        # profiler
        "trainer.policy.torch_profiler_config.enable=true",
        "trainer.policy.torch_profiler_config.ranks=[0]",
        f"trainer.policy.torch_profiler_config.save_path={PROFILE_ROOT}/trainer/{mode}",
        "trainer.policy.torch_profiler_config.skip_first=3",
        "trainer.policy.torch_profiler_config.wait=0",
        "trainer.policy.torch_profiler_config.warmup=1",
        "trainer.policy.torch_profiler_config.active=2",
        "trainer.policy.torch_profiler_config.repeat=1",
        "trainer.policy.torch_profiler_config.record_shapes=false",
        "trainer.policy.torch_profiler_config.with_stack=false",
        "trainer.logger=wandb",
        "trainer.project_name=skyrl-expert-mxfp8",
        f"trainer.run_name={run_name}",
    ]
    if mode == "mxfp8":
        command.append("generator.inference_engine.fp8_weight_sync_mode=serialized_mxfp8")
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=REMOTE_REPO)
    while True:
        try:
            returncode = process.wait(timeout=PROFILE_COMMIT_INTERVAL_S)
            break
        except subprocess.TimeoutExpired:
            profile_volume.commit()  # profiler
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    print(f"{mode} benchmark completed in {time.perf_counter() - started:.1f}s")


@app.function(
    image=image,
    volumes={HF_HOME: hf_volume, "/root/data": data_volume},
    timeout=2 * 60 * 60,
)
def prepare_assets() -> None:
    import subprocess

    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    hf_volume.commit()
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(f"{DATA_DIR}/train.parquet"):
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--no-sync",
                "--extra",
                "megatron",
                "examples/train/gsm8k/gsm8k_dataset.py",
                "--output_dir",
                DATA_DIR,
            ],
            cwd=REMOTE_REPO,
            check=True,
        )
        data_volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        HF_HOME: hf_volume,
        "/root/data": data_volume,
        PROFILE_ROOT: profile_volume,
    },  # profiler
    timeout=24 * 60 * 60,
)
def benchmark(mode: str, steps: int) -> None:
    try:
        _run(mode, steps)
    finally:
        profile_volume.commit()  # profiler


@app.local_entrypoint()
def main(mode: str = "both", steps: int = 10) -> None:
    if mode not in {"bf16", "mxfp8", "both"}:
        raise ValueError("mode must be bf16, mxfp8, or both")
    prepare_assets.remote()
    modes = ("bf16", "mxfp8") if mode == "both" else (mode,)
    for selected in modes:
        benchmark.remote(selected, steps)
