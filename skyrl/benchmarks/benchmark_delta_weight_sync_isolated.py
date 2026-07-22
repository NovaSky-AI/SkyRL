"""Isolated checkpoint-delta weight-sync benchmark.

This intentionally reuses the Megatron GPU CI setup but avoids generation,
optimizer initialization, and policy training. It initializes an inference
deployment plus a Megatron policy worker group, applies deterministic sparse
in-place parameter updates, then exercises the normal SkyRL
``broadcast_to_inference_engines`` path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray, validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.delta_weight_sync_utils import (
    init_policy_worker_for_delta,
)
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
)


def _json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed)}")
    return parsed


def build_cfg(args: argparse.Namespace) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = args.trainer_num_nodes
    cfg.trainer.placement.policy_num_gpus_per_node = args.trainer_gpus_per_node
    cfg.trainer.placement.ref_num_nodes = args.trainer_num_nodes
    cfg.trainer.placement.ref_num_gpus_per_node = args.trainer_gpus_per_node
    cfg.trainer.log_path = args.infra_log_path
    cfg.trainer.run_name = args.run_name
    cfg.trainer.project_name = "delta-weight-sync-isolated"
    cfg.trainer.ckpt_path = str(Path(args.output_dir) / "ckpts")
    cfg.trainer.remove_microbatch_padding = args.remove_microbatch_padding
    cfg.trainer.flash_attn = args.flash_attn

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = args.megatron_tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = args.megatron_pp
    cfg.trainer.policy.megatron_config.context_parallel_size = args.megatron_cp
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = args.megatron_ep
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = args.megatron_etp
    if args.last_pipeline_stage_layers is not None:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["num_layers_in_last_pipeline_stage"] = (
            args.last_pipeline_stage_layers
        )

    cfg.trainer.policy.language_model_only = args.language_model_only
    cfg.trainer.ref.language_model_only = args.language_model_only

    ie_cfg = cfg.generator.inference_engine
    ie_cfg.backend = "vllm"
    ie_cfg.run_engines_locally = True
    ie_cfg.distributed_executor_backend = "ray"
    ie_cfg.num_engines = args.num_inference_engines
    ie_cfg.tensor_parallel_size = args.inference_tp
    ie_cfg.weight_sync_backend = "delta"
    ie_cfg.gpu_memory_utilization = args.gpu_memory_utilization
    ie_cfg.language_model_only = args.language_model_only
    ie_cfg.engine_init_kwargs = _json_dict(args.engine_init_kwargs)
    ie_cfg.delta_weight_sync.sync_dir = args.sync_dir
    ie_cfg.delta_weight_sync.local_checkpoint_dir = args.local_checkpoint_dir
    ie_cfg.delta_weight_sync.max_file_size_in_gb = args.max_file_size_in_gb
    ie_cfg.delta_weight_sync.publish_num_workers = args.publish_num_workers
    ie_cfg.delta_weight_sync.checkpoint_load_format = args.checkpoint_load_format
    ie_cfg.delta_weight_sync.multi_thread_safetensors_max_workers = args.multi_thread_safetensors_max_workers

    validate_cfg(cfg)
    return cfg


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.run_name}.json"
    cfg = build_cfg(args)
    engine_init_kwargs = _json_dict(args.engine_init_kwargs)
    if not ray.is_initialized():
        initialize_ray(cfg)

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model": args.model,
        "sync_dir": args.sync_dir,
        "local_checkpoint_dir": args.local_checkpoint_dir,
        "updates": [],
    }

    async with InferenceEngineState.create(
        cfg=cfg,
        model=args.model,
        use_local=True,
        tp_size=args.inference_tp,
        colocate_all=False,
        backend="vllm",
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_inference_engines=args.num_inference_engines,
        sleep_level=2,
        engine_init_kwargs=engine_init_kwargs,
        distributed_executor_backend="ray",
        language_model_only=args.language_model_only,
    ) as engines:
        client = engines.client
        policy = init_policy_worker_for_delta(
            cfg=cfg,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=args.trainer_gpus_per_node,
            num_nodes=args.trainer_num_nodes,
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )

        for update_idx in range(args.num_updates + 1):
            update: dict[str, Any] = {"index": update_idx}
            if update_idx == 0:
                update["kind"] = "seed"
            else:
                update["kind"] = "sparse_delta"
                update["perturb"] = ray.get(
                    policy.async_run_ray_method(
                        "pass_through",
                        "benchmark_apply_sparse_weight_delta",
                        sparsity=args.sparsity,
                        delta=args.delta,
                        phase=update_idx,
                    )
                )

            t0 = time.perf_counter()
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "broadcast_to_inference_engines",
                    client,
                    cfg.generator.inference_engine,
                )
            )
            update["sync_wall_s"] = time.perf_counter() - t0
            results["updates"].append(update)
            output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps({"event": "sync_complete", **update}, sort_keys=True), flush=True)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sync-dir", required=True)
    parser.add_argument("--local-checkpoint-dir", required=True)
    parser.add_argument("--infra-log-path", required=True)
    parser.add_argument("--num-updates", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=0.04)
    parser.add_argument("--delta", type=float, default=1.0e-3)
    parser.add_argument("--trainer-num-nodes", type=int, default=1)
    parser.add_argument("--trainer-gpus-per-node", type=int, default=8)
    parser.add_argument("--megatron-tp", type=int, default=2)
    parser.add_argument("--megatron-pp", type=int, default=1)
    parser.add_argument("--megatron-cp", type=int, default=1)
    parser.add_argument("--megatron-ep", type=int, default=8)
    parser.add_argument("--megatron-etp", type=int, default=1)
    parser.add_argument("--last-pipeline-stage-layers", type=int, default=None)
    parser.add_argument("--num-inference-engines", type=int, default=1)
    parser.add_argument("--inference-tp", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-file-size-in-gb", type=float, default=2.0)
    parser.add_argument("--publish-num-workers", type=int, default=None)
    parser.add_argument(
        "--checkpoint-load-format",
        default="vllm_fastsafetensors",
        choices=[
            "vllm_fastsafetensors",
            "vllm_multi_thread_safetensors",
        ],
    )
    parser.add_argument("--multi-thread-safetensors-max-workers", type=int, default=8)
    parser.add_argument("--engine-init-kwargs", default=None)
    parser.add_argument("--language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-microbatch-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flash-attn", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    os.environ.setdefault("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "1")
    results = asyncio.run(run_benchmark(args))
    output_path = Path(args.output_dir) / f"{args.run_name}.json"
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "benchmark_complete", "result_path": str(output_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
