"""
Single-file Modal benchmark for comparing basic pipeline parallelism (PP)
against virtual pipeline parallelism (VPP) on a small SkyRL Megatron
train-step workload.

The Modal wrapper launches this same file in an internal benchmark mode under:
    uv run --isolated --extra megatron python ...

That keeps the benchmark self-contained while still using the repo's locked
Megatron environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import modal
except ImportError:
    modal = None


INTERNAL_STEP_BENCHMARK_FLAG = "--internal-step-benchmark"
APP_NAME = os.getenv("MODAL_APP_NAME", "skyrl-vpp-benchmark")
DATA_ROOT = Path("/root/data/vpp_benchmark")
HF_CACHE_DIR = Path("/root/data/hf-cache")
DEFAULT_BENCHMARK_MODEL = "Qwen/Qwen3-0.6B"
LEGACY_QWEN2_BENCHMARK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _find_local_repo_root() -> Path:
    if "SKYRL_REPO_ROOT" in os.environ:
        return Path(os.environ["SKYRL_REPO_ROOT"])

    candidates = [Path(__file__).resolve(), Path.cwd()]
    for start in candidates:
        for base in [start] + list(start.parents):
            if base.exists() and (base / "skyrl-gym").exists():
                return base
    raise RuntimeError("SkyRL root repo path not found")


def create_modal_image():
    if modal is None:
        raise RuntimeError("modal is required to build the Modal benchmark image")

    local_repo_path = _find_local_repo_root()
    print(f"Root path: {local_repo_path}")

    envs = {
        "SKYRL_REPO_ROOT": "/root/SkyRL",
    }

    return (
        modal.Image.from_registry("novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8-megatron")
        .pip_install("huggingface_hub")
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


def create_modal_volume(volume_name: str = "skyrl-data"):
    if modal is None:
        raise RuntimeError("modal is required to create the Modal volume")
    data_volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    return {"/root/data": data_volume}


def _run_command(
    args: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    check: bool = True,
) -> tuple[str, float]:
    print(f"$ {shlex.join(args)}")
    start = time.perf_counter()
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    returncode = process.wait()
    elapsed = time.perf_counter() - start
    output = "".join(output_lines)
    if check and returncode != 0:
        tail = "".join(output_lines[-200:]).strip()
        message = f"Command failed with exit code {returncode}: {shlex.join(args)}"
        if tail:
            message = f"{message}\n--- command output tail ---\n{tail}"
        raise RuntimeError(message)
    return output, elapsed


def _extract_result_json(output: str) -> dict[str, Any]:
    matches = re.findall(r"^RESULT_JSON=(\{.*\})$", output, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError("Internal benchmark runner did not emit RESULT_JSON output")
    return json.loads(matches[-1])


def _print_summary(results: dict[str, dict[str, Any]]) -> None:
    pp = results["pp"]
    vpp = results["vpp"]

    def _speedup(metric: str) -> float | None:
        pp_value = pp.get(metric)
        vpp_value = vpp.get(metric)
        if not pp_value or not vpp_value:
            return None
        return pp_value / vpp_value

    summary = {
        "wall_clock_speedup": _speedup("elapsed_s"),
        "avg_step_speedup": _speedup("avg_step_s"),
        "avg_train_phase_speedup": _speedup("avg_train_phase_s"),
        "avg_policy_train_speedup": _speedup("avg_policy_train_s"),
    }

    print("\n=== PP vs VPP Summary ===")
    for label in ("pp", "vpp"):
        run = results[label]
        print(
            f"{label.upper():>3} | elapsed={run['elapsed_s']:.2f}s | "
            f"steps={run['steps']} | "
            f"avg_step={run['avg_step_s'] if run['avg_step_s'] is not None else 'n/a'} | "
            f"avg_train_phase={run['avg_train_phase_s'] if run['avg_train_phase_s'] is not None else 'n/a'} | "
            f"avg_policy_train={run['avg_policy_train_s'] if run['avg_policy_train_s'] is not None else 'n/a'}"
        )

    print("Speedups (PP / VPP):")
    for key, value in summary.items():
        pretty = f"{value:.3f}x" if value is not None else "n/a"
        print(f"  {key}: {pretty}")


def _build_internal_subprocess_args(
    *,
    model_name: str,
    num_gpus: int,
    pp_size: int,
    vpp_size: int | None,
    policy_mini_batch_size: int,
    micro_batch_size: int,
    measure_steps: int,
    warmup_steps: int,
) -> list[str]:
    args = [
        "uv",
        "run",
        "--isolated",
        "--extra",
        "megatron",
        "python",
        "examples/train_integrations/modal/vpp_vs_pp_benchmark.py",
        INTERNAL_STEP_BENCHMARK_FLAG,
        "--model-name",
        model_name,
        "--num-gpus",
        str(num_gpus),
        "--pp-size",
        str(pp_size),
        "--policy-mini-batch-size",
        str(policy_mini_batch_size),
        "--micro-batch-size",
        str(micro_batch_size),
        "--measure-steps",
        str(measure_steps),
        "--warmup-steps",
        str(warmup_steps),
    ]
    if vpp_size is not None:
        args.extend(["--vpp-size", str(vpp_size)])
    return args


def _build_cfg(
    *,
    model_name: str,
    num_gpus: int,
    pp_size: int,
    vpp_size: int | None,
    policy_mini_batch_size: int,
    micro_batch_size: int,
):
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils.utils import validate_cfg

    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.flash_attn = False
    cfg.trainer.use_sample_packing = False
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.ref.model.path = None
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.train_batch_size = policy_mini_batch_size
    cfg.trainer.policy_mini_batch_size = policy_mini_batch_size
    cfg.trainer.micro_forward_batch_size_per_gpu = micro_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = micro_batch_size
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp_size
    cfg.trainer.policy.megatron_config.virtual_pipeline_model_parallel_size = vpp_size
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["share_embeddings_and_output_weights"] = False
    validate_cfg(cfg)
    return cfg


def _make_dummy_training_batch(batch_size: int, seq_len: int = 128, num_actions: int = 32):
    import torch
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

    torch.manual_seed(42)
    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len), device="cpu"),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=int, device="cpu"),
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
            "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "response_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "rollout_logprobs": 0.2 * torch.ones((batch_size, num_actions), device="cpu"),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def _import_worker(strategy: str, worker_type: str):
    if strategy in ("fsdp", "fsdp2"):
        module_path = "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker"
    elif strategy == "megatron":
        module_path = "skyrl.backends.skyrl_train.workers.megatron.megatron_worker"
    else:
        raise ValueError(f"Unknown strategy type for {worker_type}: {strategy}")
    module = importlib.import_module(module_path)
    return getattr(module, f"{worker_type.capitalize()}Worker")


def _init_ray_internal() -> None:
    import ray

    if ray.is_initialized():
        ray.shutdown()

    env_vars = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NVTE_FUSED_ATTN": "0",
    }
    if os.environ.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = os.environ["PYTHONPATH"]
    if os.environ.get("LD_LIBRARY_PATH"):
        env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    ray.init(runtime_env={"env_vars": env_vars}, log_to_driver=True)


def _init_policy_actor_group(cfg, num_gpus: int):
    import ray
    from ray.util.placement_group import placement_group

    from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
    from skyrl.train.utils.utils import ResolvedPlacementGroup, get_ray_pg_ready_with_timeout

    raw_pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(raw_pg, timeout=60)
    pg = ResolvedPlacementGroup(raw_pg)
    worker_cls = _import_worker(cfg.trainer.strategy, "policy")
    actor_group = PPORayActorGroup(
        cfg.trainer,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    ray.get(actor_group.async_init_model(cfg.trainer.policy.model.path))
    return actor_group


def _run_single_case(
    *,
    model_name: str,
    num_gpus: int,
    pp_size: int,
    vpp_size: int | None,
    policy_mini_batch_size: int,
    micro_batch_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    import ray

    cfg = _build_cfg(
        model_name=model_name,
        num_gpus=num_gpus,
        pp_size=pp_size,
        vpp_size=vpp_size,
        policy_mini_batch_size=policy_mini_batch_size,
        micro_batch_size=micro_batch_size,
    )
    _init_ray_internal()
    actor_group = _init_policy_actor_group(cfg, num_gpus=num_gpus)
    batch = _make_dummy_training_batch(batch_size=policy_mini_batch_size)

    warmup_times = []
    step_times = []
    forward_backward_times = []
    optim_step_times = []
    last_result = None

    try:
        total_steps = warmup_steps + measure_steps
        for step_idx in range(total_steps):
            batch.metadata["global_step"] = step_idx

            forward_start = time.perf_counter()
            results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", batch))
            forward_elapsed = time.perf_counter() - forward_start

            optim_start = time.perf_counter()
            ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
            optim_elapsed = time.perf_counter() - optim_start

            step_elapsed = forward_elapsed + optim_elapsed
            if step_idx < warmup_steps:
                warmup_times.append(step_elapsed)
            else:
                step_times.append(step_elapsed)
                forward_backward_times.append(forward_elapsed)
                optim_step_times.append(optim_elapsed)
                last_result = results[0] if results else None
    finally:
        ray.shutdown()

    return {
        "steps": measure_steps,
        "warmup_steps": warmup_steps,
        "avg_step_s": statistics.fmean(step_times),
        "avg_forward_backward_s": statistics.fmean(forward_backward_times),
        "avg_optim_step_s": statistics.fmean(optim_step_times),
        "avg_warmup_step_s": statistics.fmean(warmup_times) if warmup_times else None,
        "sample_policy_loss": None if last_result is None else last_result.get("policy_loss"),
    }


def _run_internal_benchmark_from_cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(INTERNAL_STEP_BENCHMARK_FLAG, action="store_true")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--vpp-size", type=int, default=None)
    parser.add_argument("--policy-mini-batch-size", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=4)
    args = parser.parse_args()

    result = _run_single_case(
        model_name=args.model_name,
        num_gpus=args.num_gpus,
        pp_size=args.pp_size,
        vpp_size=args.vpp_size,
        policy_mini_batch_size=args.policy_mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    print(f"RESULT_JSON={json.dumps(result, sort_keys=True)}")


if modal is not None:
    app = modal.App(APP_NAME)
    image = create_modal_image()
    volume = create_modal_volume()

    @app.function(
        image=image,
        gpu=os.environ.get("MODAL_GPU", "A100:4"),
        volumes=volume,
        timeout=4 * 60 * 60,
    )
    def run_vpp_benchmark(
        model_name: str = DEFAULT_BENCHMARK_MODEL,
        num_gpus: int = 4,
        pp_size: int = 2,
        vpp_size: int = 2,
        train_examples: int = 64,
        val_examples: int = 16,
        train_batch_size: int = 16,
        policy_mini_batch_size: int = 16,
        micro_batch_size: int = 4,
        n_samples_per_prompt: int = 1,
        max_generate_length: int = 32,
    ) -> dict[str, Any]:
        from huggingface_hub import snapshot_download

        del val_examples
        del max_generate_length

        assert pp_size > 1, "pp_size must be greater than 1 for a PP vs VPP comparison"
        assert vpp_size > 1, "vpp_size must be greater than 1"
        assert num_gpus % pp_size == 0, "num_gpus must be divisible by pp_size"

        if model_name == LEGACY_QWEN2_BENCHMARK_MODEL:
            print(
                "Switching benchmark model from "
                f"{LEGACY_QWEN2_BENCHMARK_MODEL} to {DEFAULT_BENCHMARK_MODEL} "
                "because the current Megatron/VPP path in SkyRL is validated with Qwen3-0.6B."
            )
            model_name = DEFAULT_BENCHMARK_MODEL

        policy_dp_size = num_gpus // pp_size
        effective_policy_mini_batch_size = policy_mini_batch_size * n_samples_per_prompt
        policy_mini_batch_size_per_gpu = effective_policy_mini_batch_size // policy_dp_size
        assert (
            policy_mini_batch_size_per_gpu % micro_batch_size == 0
        ), "policy_mini_batch_size_per_gpu must be divisible by micro_batch_size"
        assert (
            policy_mini_batch_size_per_gpu // micro_batch_size
        ) % pp_size == 0, "num_microbatches must be divisible by pp_size for interleaved VPP"

        repo_root = Path(os.environ.get("SKYRL_REPO_ROOT", "/root/SkyRL"))
        gym_root = repo_root / "skyrl-gym"
        results_dir = DATA_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["SKYRL_REPO_ROOT"] = str(repo_root)
        env["HF_HOME"] = str(HF_CACHE_DIR)
        env["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
        env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["SKYRL_PYTHONPATH_EXPORT"] = "1"
        pythonpath_entries = [str(repo_root), str(gym_root)]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(pythonpath_entries)
        os.environ.update(
            {
                "HF_HOME": env["HF_HOME"],
                "HF_HUB_CACHE": env["HF_HUB_CACHE"],
                "TRANSFORMERS_CACHE": env["TRANSFORMERS_CACHE"],
                "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
                "SKYRL_PYTHONPATH_EXPORT": env["SKYRL_PYTHONPATH_EXPORT"],
                "PYTHONPATH": env["PYTHONPATH"],
            }
        )

        os.chdir(repo_root)

        print(f"Pre-downloading model to cache: {model_name}")
        snapshot_download(model_name, local_files_only=False)
        measure_steps = max(1, train_examples // max(1, effective_policy_mini_batch_size))
        if measure_steps < train_examples // max(1, train_batch_size):
            measure_steps = max(1, train_examples // max(1, train_batch_size))
        warmup_steps = 1

        configs = {
            "pp": _build_internal_subprocess_args(
                model_name=model_name,
                num_gpus=num_gpus,
                pp_size=pp_size,
                vpp_size=None,
                policy_mini_batch_size=effective_policy_mini_batch_size,
                micro_batch_size=micro_batch_size,
                measure_steps=measure_steps,
                warmup_steps=warmup_steps,
            ),
            "vpp": _build_internal_subprocess_args(
                model_name=model_name,
                num_gpus=num_gpus,
                pp_size=pp_size,
                vpp_size=vpp_size,
                policy_mini_batch_size=effective_policy_mini_batch_size,
                micro_batch_size=micro_batch_size,
                measure_steps=measure_steps,
                warmup_steps=warmup_steps,
            ),
        }

        results: dict[str, dict[str, Any]] = {}
        for label, args in configs.items():
            print(f"\n=== Running {label.upper()} benchmark ===")
            output, elapsed_s = _run_command(args, cwd=str(repo_root), env=env)
            (results_dir / label).mkdir(parents=True, exist_ok=True)
            (results_dir / label / "stdout.log").write_text(output)
            run_result = _extract_result_json(output)
            results[label] = {
                "label": label,
                "elapsed_s": elapsed_s,
                "steps": run_result["steps"],
                "avg_step_s": run_result["avg_step_s"],
                "avg_train_phase_s": run_result["avg_forward_backward_s"],
                "avg_policy_train_s": run_result["avg_forward_backward_s"],
                "avg_optim_step_s": run_result["avg_optim_step_s"],
                "warmup_steps": run_result["warmup_steps"],
            }

        pp_run = results["pp"]
        vpp_run = results["vpp"]
        results["speedups"] = {
            "wall_clock_pp_over_vpp": (pp_run["elapsed_s"] / vpp_run["elapsed_s"] if vpp_run["elapsed_s"] else None),
            "avg_step_pp_over_vpp": (
                pp_run["avg_step_s"] / vpp_run["avg_step_s"]
                if pp_run["avg_step_s"] is not None and vpp_run["avg_step_s"] not in (None, 0)
                else None
            ),
            "avg_policy_train_pp_over_vpp": (
                pp_run["avg_policy_train_s"] / vpp_run["avg_policy_train_s"]
                if pp_run["avg_policy_train_s"] is not None and vpp_run["avg_policy_train_s"] not in (None, 0)
                else None
            ),
        }

        _print_summary(results)

        output_path = results_dir / "benchmark_results.json"
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved benchmark results to {output_path}")
        return results

    @app.local_entrypoint()
    def main(
        model_name: str = DEFAULT_BENCHMARK_MODEL,
        num_gpus: int = 4,
        pp_size: int = 2,
        vpp_size: int = 2,
        train_examples: int = 64,
        val_examples: int = 16,
        train_batch_size: int = 16,
        policy_mini_batch_size: int = 16,
        micro_batch_size: int = 4,
        n_samples_per_prompt: int = 1,
        max_generate_length: int = 32,
    ) -> None:
        results = run_vpp_benchmark.remote(
            model_name=model_name,
            num_gpus=num_gpus,
            pp_size=pp_size,
            vpp_size=vpp_size,
            train_examples=train_examples,
            val_examples=val_examples,
            train_batch_size=train_batch_size,
            policy_mini_batch_size=policy_mini_batch_size,
            micro_batch_size=micro_batch_size,
            n_samples_per_prompt=n_samples_per_prompt,
            max_generate_length=max_generate_length,
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__" and INTERNAL_STEP_BENCHMARK_FLAG in sys.argv:
    _run_internal_benchmark_from_cli()
