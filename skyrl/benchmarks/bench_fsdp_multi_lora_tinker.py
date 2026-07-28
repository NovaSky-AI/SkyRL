"""Benchmark end-to-end FSDP multi-LoRA training through the Tinker API.

This benchmark compares the single-resident adapter store with concurrent
grouped-MM LoRA under two workloads:

* ``underfilled``: every adapter submits one sequence per step. Concurrent
  execution can combine otherwise-small tenant batches.
* ``saturated``: every adapter independently submits enough sequences to fill
  ``max_tokens_per_microbatch``. This tests whether grouped MM helps once a
  single tenant is already compute-bound.

Each measured step includes Tinker request handling, forward, backward,
adapter swapping/routing, and the optimizer step. Run one implementation per
GPU and compare the resulting JSON files with ``--compare``.

Example:
    uv run --extra tinker --extra fsdp python -m skyrl.benchmarks.bench_fsdp_multi_lora_tinker \
        --implementation concurrent --model Qwen/Qwen3.5-4B --rank 32 \
        --adapter-counts 2,4 --sequence-length 512 --max-tokens-per-microbatch 4096 \
        --warmup-steps 5 --measured-steps 10 --output concurrent.json

    uv run python -m skyrl.benchmarks.bench_fsdp_multi_lora_tinker \
        --compare single.json concurrent.json --output comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

Implementation = Literal["single", "concurrent"]
Workload = Literal["underfilled", "saturated"]
IMPLEMENTATIONS = ("single", "concurrent")
WORKLOADS = ("underfilled", "saturated")
_BATCH_LOG_PATTERN = re.compile(r"process_batch_requests\((forward_backward|optim_step), n=(\d+)\)")


@dataclass(frozen=True)
class BenchmarkCase:
    workload: Workload
    active_adapters: int
    examples_per_adapter: int
    sequence_length: int
    max_tokens_per_microbatch: int

    @property
    def tokens_per_step(self) -> int:
        return self.active_adapters * self.examples_per_adapter * self.sequence_length


class GPUMemoryMonitor:
    """Poll aggregate visible-device memory while a benchmark case runs."""

    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self._peak_mib = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _used_memory_mib() -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return sum(int(value.strip()) for value in result.stdout.splitlines() if value.strip().isdigit())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            self._peak_mib = max(self._peak_mib, self._used_memory_mib())
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._peak_mib = self._used_memory_mib()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._peak_mib = max(self._peak_mib, self._used_memory_mib())
        return self._peak_mib


class TinkerServer:
    def __init__(
        self,
        *,
        model: str,
        implementation: Implementation,
        rank: int,
        max_adapters: int,
        max_tokens_per_microbatch: int,
        host: str,
        port: int,
        output_dir: Path,
        startup_timeout: float,
    ):
        self.model = model
        self.implementation = implementation
        self.rank = rank
        self.max_adapters = max_adapters
        self.max_tokens_per_microbatch = max_tokens_per_microbatch
        self.host = host
        self.port = port
        self.output_dir = output_dir
        self.startup_timeout = startup_timeout
        self.log_path = output_dir / "server.log"
        self.db_path = output_dir / "tinker.db"
        self.process: subprocess.Popen | None = None
        self._log_file = None

    def _backend_config(self) -> dict[str, Any]:
        return {
            "strategy": "fsdp",
            "max_lora_adapters": self.max_adapters,
            "trainer.placement.policy_num_gpus_per_node": 1,
            "trainer.placement.policy_num_nodes": 1,
            "trainer.placement.colocate_all": False,
            "trainer.remove_microbatch_padding": False,
            "trainer.gradient_checkpointing": True,
            "trainer.micro_train_batch_size_per_gpu": 1,
            "trainer.micro_forward_batch_size_per_gpu": 1,
            "trainer.max_tokens_per_microbatch": self.max_tokens_per_microbatch,
            "trainer.policy.language_model_only": True,
            "trainer.policy.model.lora.implementation": self.implementation,
            "trainer.policy.model.lora.max_loras": 1,
            "trainer.policy.model.lora.max_cpu_loras": self.max_adapters,
        }

    def _command(self) -> list[str]:
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to launch the Tinker server")
        return [
            uv,
            "run",
            "--no-sync",
            "-m",
            "skyrl.tinker.api",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--base-model",
            self.model,
            "--backend",
            "fsdp",
            "--backend-config",
            json.dumps(self._backend_config()),
            "--database-url",
            f"sqlite:///{self.db_path}",
        ]

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w")
        self.process = subprocess.Popen(
            self._command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        health_url = f"http://{self.host}:{self.port}/api/v1/healthz"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Tinker server exited during startup; see {self.log_path}")
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError(f"Tinker server was not ready after {self.startup_timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()
        ray = shutil.which("ray")
        if ray is None:
            venv_ray = Path(sys.executable).with_name("ray")
            ray = str(venv_ray) if venv_ray.exists() else None
        if ray is not None:
            subprocess.run([ray, "stop", "--force"], check=False, capture_output=True, text=True, timeout=30)


def build_cases(
    adapter_counts: list[int],
    workloads: list[Workload],
    sequence_length: int,
    max_tokens_per_microbatch: int,
) -> list[BenchmarkCase]:
    if sequence_length <= 0:
        raise ValueError("sequence-length must be positive")
    if max_tokens_per_microbatch < sequence_length:
        raise ValueError("max-tokens-per-microbatch must be at least sequence-length")
    saturated_examples = max_tokens_per_microbatch // sequence_length
    cases = []
    for active_adapters in adapter_counts:
        if active_adapters <= 0:
            raise ValueError("adapter counts must be positive")
        for workload in workloads:
            examples_per_adapter = 1 if workload == "underfilled" else saturated_examples
            cases.append(
                BenchmarkCase(
                    workload=workload,
                    active_adapters=active_adapters,
                    examples_per_adapter=examples_per_adapter,
                    sequence_length=sequence_length,
                    max_tokens_per_microbatch=max_tokens_per_microbatch,
                )
            )
    return cases


def summarize_samples(samples: list[float], tokens_per_step: int) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize an empty sample set")
    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    median_seconds = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "p95_seconds": ordered[p95_index],
        "min_seconds": ordered[0],
        "max_seconds": ordered[-1],
        "tokens_per_second": tokens_per_step / median_seconds,
    }


def compare_results(single: dict[str, Any], concurrent: dict[str, Any]) -> dict[str, Any]:
    if single["config"]["implementation"] != "single":
        raise ValueError("first comparison input must use implementation=single")
    if concurrent["config"]["implementation"] != "concurrent":
        raise ValueError("second comparison input must use implementation=concurrent")

    def case_key(case: dict[str, Any]) -> tuple[str, int, int, int]:
        cfg = case["case"]
        return (
            cfg["workload"],
            cfg["active_adapters"],
            cfg["sequence_length"],
            cfg["max_tokens_per_microbatch"],
        )

    single_cases = {case_key(case): case for case in single["cases"]}
    concurrent_cases = {case_key(case): case for case in concurrent["cases"]}
    if single_cases.keys() != concurrent_cases.keys():
        raise ValueError("single and concurrent result files contain different benchmark cases")

    comparisons = []
    for key in sorted(single_cases):
        single_case = single_cases[key]
        concurrent_case = concurrent_cases[key]
        single_rate = single_case["total"]["tokens_per_second"]
        concurrent_rate = concurrent_case["total"]["tokens_per_second"]
        comparisons.append(
            {
                "case": single_case["case"],
                "single_tokens_per_second": single_rate,
                "concurrent_tokens_per_second": concurrent_rate,
                "speedup": concurrent_rate / single_rate,
                "single_peak_gpu_memory_mib": single_case["peak_gpu_memory_mib"],
                "concurrent_peak_gpu_memory_mib": concurrent_case["peak_gpu_memory_mib"],
            }
        )
    return {
        "single_source": single.get("output_path"),
        "concurrent_source": concurrent.get("output_path"),
        "cases": comparisons,
    }


def _make_datum(tinker_types, tokenizer, sequence_length: int):
    seed_tokens = tokenizer.encode(
        "SkyRL multi tenant LoRA throughput benchmark. ",
        add_special_tokens=False,
    )
    if not seed_tokens:
        raise RuntimeError("tokenizer returned no tokens for benchmark seed text")
    tokens = (seed_tokens * math.ceil(sequence_length / len(seed_tokens)))[:sequence_length]
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokens[-1]
    targets = tokens[1:] + [eos_token_id]
    weights = [0.0] + [1.0] * (sequence_length - 1)
    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(tokens),
        loss_fn_inputs={"target_tokens": targets, "weights": weights},
    )


def _run_parallel(
    executor: ThreadPoolExecutor,
    clients: list[Any],
    operation,
    request_timeout: float,
) -> tuple[float, list[Any]]:
    barrier = threading.Barrier(len(clients) + 1)

    def invoke(client):
        barrier.wait()
        return operation(client).result(timeout=request_timeout)

    futures = [executor.submit(invoke, client) for client in clients]
    start = time.perf_counter()
    barrier.wait()
    results = [future.result(timeout=request_timeout + 30) for future in futures]
    return time.perf_counter() - start, results


def _run_case(
    *,
    case: BenchmarkCase,
    clients: list[Any],
    datum,
    tinker_types,
    warmup_steps: int,
    measured_steps: int,
    request_timeout: float,
) -> dict[str, Any]:
    active_clients = clients[: case.active_adapters]
    data = [datum] * case.examples_per_adapter
    adam = tinker_types.AdamParams(learning_rate=0.0)

    forward_backward_samples = []
    optimizer_samples = []
    total_samples = []
    output_metrics = []
    optimizer_metrics = []
    monitor = GPUMemoryMonitor()

    with ThreadPoolExecutor(max_workers=case.active_adapters) as executor:
        for step in range(warmup_steps + measured_steps):
            if step == warmup_steps:
                monitor.start()
            total_start = time.perf_counter()
            forward_seconds, outputs = _run_parallel(
                executor,
                active_clients,
                lambda client: client.forward_backward(data, "cross_entropy"),
                request_timeout,
            )
            if any(len(output.loss_fn_outputs) != case.examples_per_adapter for output in outputs):
                raise RuntimeError("forward_backward returned an unexpected number of outputs")
            optimizer_seconds, optimizer_outputs = _run_parallel(
                executor,
                active_clients,
                lambda client: client.optim_step(adam),
                request_timeout,
            )
            total_seconds = time.perf_counter() - total_start

            if step >= warmup_steps:
                forward_backward_samples.append(forward_seconds)
                optimizer_samples.append(optimizer_seconds)
                total_samples.append(total_seconds)
                output_metrics.append([output.metrics for output in outputs])
                optimizer_metrics.append([output.metrics for output in optimizer_outputs])

    peak_gpu_memory_mib = monitor.stop()
    return {
        "case": asdict(case),
        "forward_backward": summarize_samples(forward_backward_samples, case.tokens_per_step),
        "optimizer": summarize_samples(optimizer_samples, case.tokens_per_step),
        "total": summarize_samples(total_samples, case.tokens_per_step),
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "forward_backward_metrics": output_metrics,
        "optimizer_metrics": optimizer_metrics,
    }


def _batch_log_counts(log_path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {"forward_backward": {}, "optim_step": {}}
    if not log_path.exists():
        return counts
    for operation, batch_size in _BATCH_LOG_PATTERN.findall(log_path.read_text(errors="replace")):
        operation_counts = counts[operation]
        operation_counts[batch_size] = operation_counts.get(batch_size, 0) + 1
    return counts


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import tinker
        from tinker import types as tinker_types
    except ImportError as exc:
        raise RuntimeError("Install the tinker extra before running this benchmark") from exc

    adapter_counts = _parse_csv_ints(args.adapter_counts)
    workloads = args.workloads.split(",")
    if any(workload not in WORKLOADS for workload in workloads):
        raise ValueError(f"workloads must be a comma-separated subset of {WORKLOADS}")
    cases = build_cases(adapter_counts, workloads, args.sequence_length, args.max_tokens_per_microbatch)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"skyrl-{args.implementation}-", dir=args.work_dir))
    server = TinkerServer(
        model=args.model,
        implementation=args.implementation,
        rank=args.rank,
        max_adapters=max(adapter_counts),
        max_tokens_per_microbatch=args.max_tokens_per_microbatch,
        host=args.host,
        port=args.port,
        output_dir=run_dir,
        startup_timeout=args.startup_timeout,
    )
    result: dict[str, Any] = {
        "config": {
            "implementation": args.implementation,
            "model": args.model,
            "rank": args.rank,
            "adapter_counts": adapter_counts,
            "workloads": workloads,
            "sequence_length": args.sequence_length,
            "max_tokens_per_microbatch": args.max_tokens_per_microbatch,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "output_path": str(output_path),
        "run_directory": str(run_dir),
        "cases": [],
    }
    try:
        server.start()
        service_client = tinker.ServiceClient(
            base_url=f"http://{args.host}:{args.port}/",
            api_key="tml-dummy",
        )
        clients = [
            service_client.create_lora_training_client(base_model=args.model, rank=args.rank)
            for _ in range(max(adapter_counts))
        ]
        tokenizer = clients[0].get_tokenizer()
        datum = _make_datum(tinker_types, tokenizer, args.sequence_length)
        for case in cases:
            print(
                f"Running {args.implementation}: workload={case.workload}, "
                f"adapters={case.active_adapters}, examples/adapter={case.examples_per_adapter}",
                flush=True,
            )
            case_result = _run_case(
                case=case,
                clients=clients,
                datum=datum,
                tinker_types=tinker_types,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                request_timeout=args.request_timeout,
            )
            result["cases"].append(case_result)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        service_client.holder.close()
    finally:
        server.stop()

    result["server_batch_log_counts"] = _batch_log_counts(server.log_path)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--adapter-counts", default="2,4")
    parser.add_argument("--workloads", default="underfilled,saturated")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-tokens-per-microbatch", type=int, default=4096)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--work-dir", default="/tmp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare", nargs=2, metavar=("SINGLE_JSON", "CONCURRENT_JSON"))
    args = parser.parse_args()
    if args.compare is None and args.implementation is None:
        parser.error("--implementation is required unless --compare is used")
    return args


def main() -> None:
    args = _parse_args()
    if args.compare is not None:
        single = json.loads(Path(args.compare[0]).read_text())
        concurrent = json.loads(Path(args.compare[1]).read_text())
        result = compare_results(single, concurrent)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    else:
        result = run_benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
