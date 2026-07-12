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
import socket
import threading
import time
from pathlib import Path
from typing import Any

import ray

from skyrl.backends.skyrl_train.weight_sync.memory_debug import process_memory_snapshot
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray, validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    init_worker_with_type,
)


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                parts = raw.strip().split()
                if not parts:
                    continue
                try:
                    value = int(parts[0])
                except ValueError:
                    continue
                values[key] = value * 1024 if len(parts) > 1 and parts[1] == "kB" else value
    except OSError:
        return {}
    return values


def _top_processes(limit: int = 20) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status: dict[str, str] = {}
            with (entry / "status").open("r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, raw = line.split(":", 1)
                        status[key] = raw.strip()
            rss = status.get("VmRSS", "0 kB").split()[0]
            rss_bytes = int(rss) * 1024
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            processes.append(
                {
                    "pid": int(entry.name),
                    "rss_bytes": rss_bytes,
                    "threads": int(status.get("Threads", "0")),
                    "name": status.get("Name", ""),
                    "cmdline": cmdline[:500],
                }
            )
        except (OSError, ValueError):
            continue
    processes.sort(key=lambda item: int(item["rss_bytes"]), reverse=True)
    return processes[:limit]


@ray.remote(num_cpus=0)
class NodeMemoryMonitor:
    def __init__(self, run_name: str, output_dir: str, interval_s: float) -> None:
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.interval_s = interval_s
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"node_memory_{self.hostname}_{self.pid}.jsonl"

    def _sample(self) -> dict[str, Any]:
        return {
            "event": "node_memory",
            "run_name": self.run_name,
            "time": time.time(),
            "hostname": self.hostname,
            "monitor_pid": self.pid,
            "meminfo": _read_meminfo(),
            "top_processes": _top_processes(),
        }

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                f.write(json.dumps(self._sample(), sort_keys=True) + "\n")
                f.flush()
                self._stop.wait(self.interval_s)
            f.write(json.dumps(self._sample(), sort_keys=True) + "\n")
            f.flush()

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="skyrl-node-memory-monitor", daemon=True)
            self._thread.start()
        return str(self.path)

    def stop(self) -> str:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 2))
        return str(self.path)


def _node_resource_key(node: dict[str, Any]) -> str | None:
    for key in node.get("Resources", {}):
        if key.startswith("node:"):
            return key
    return None


def _start_node_monitors(args: argparse.Namespace) -> list[Any]:
    if args.memory_monitor_interval_s <= 0:
        return []
    output_dir = args.memory_monitor_output_dir or str(Path(args.output_dir) / "node_memory")
    monitors = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resource_key = _node_resource_key(node)
        options = {"resources": {resource_key: 0.001}} if resource_key else {}
        monitor = NodeMemoryMonitor.options(**options).remote(args.run_name, output_dir, args.memory_monitor_interval_s)
        path = ray.get(monitor.start.remote())
        print(json.dumps({"event": "node_memory_monitor_started", "path": path}, sort_keys=True), flush=True)
        monitors.append(monitor)
    return monitors


def _stop_node_monitors(monitors: list[Any]) -> None:
    for monitor in monitors:
        try:
            path = ray.get(monitor.stop.remote(), timeout=30)
            print(json.dumps({"event": "node_memory_monitor_stopped", "path": path}, sort_keys=True), flush=True)
        except Exception as exc:
            print(
                json.dumps({"event": "node_memory_monitor_stop_failed", "error": repr(exc)}, sort_keys=True), flush=True
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
    ie_cfg.delta_weight_sync.publisher_local_checkpoint_dir = args.publisher_local_checkpoint_dir
    ie_cfg.delta_weight_sync.max_file_size_in_gb = args.max_file_size_in_gb
    ie_cfg.delta_weight_sync.max_files_to_keep = args.max_files_to_keep

    validate_cfg(cfg)
    return cfg


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / f"{args.run_name}.json"
    cfg = build_cfg(args)
    engine_init_kwargs = _json_dict(args.engine_init_kwargs)
    if not ray.is_initialized():
        initialize_ray(cfg)
    monitors = _start_node_monitors(args)

    results: dict[str, Any] = {
        "run_name": args.run_name,
        "model": args.model,
        "sync_dir": args.sync_dir,
        "local_checkpoint_dir": args.local_checkpoint_dir,
        "publisher_local_checkpoint_dir": args.publisher_local_checkpoint_dir,
        "updates": [],
    }

    try:
        print(
            json.dumps(
                {"event": "driver_memory", **process_memory_snapshot("isolated_benchmark_before_inference")},
                sort_keys=True,
            ),
            flush=True,
        )
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
            policy = init_worker_with_type(
                "policy",
                shared_pg=None,
                colocate_all=False,
                num_gpus_per_node=args.trainer_gpus_per_node,
                num_nodes=args.trainer_num_nodes,
                cfg=cfg,
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

                print(
                    json.dumps(
                        {
                            "event": "driver_memory",
                            **process_memory_snapshot(f"isolated_benchmark_before_sync_{update_idx}"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
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
                print(
                    json.dumps(
                        {
                            "event": "driver_memory",
                            **process_memory_snapshot(f"isolated_benchmark_after_sync_{update_idx}"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        _stop_node_monitors(monitors)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sync-dir", required=True)
    parser.add_argument("--local-checkpoint-dir", required=True)
    parser.add_argument("--publisher-local-checkpoint-dir", required=True)
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
    parser.add_argument("--max-files-to-keep", type=int, default=5)
    parser.add_argument("--engine-init-kwargs", default=None)
    parser.add_argument("--memory-monitor-interval-s", type=float, default=5.0)
    parser.add_argument("--memory-monitor-output-dir", default=None)
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
