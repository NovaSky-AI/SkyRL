#!/usr/bin/env python3
"""
Benchmark GRAM usage for skyrl-tx.

Usage:
    uv run --extra tinker --extra gpu python benchmarks/benchmark_memory.py \\
        --experiment-name my_test --mode sample --batch-sizes 32,64 --seq-lens 4096,8192

Output directory structure:
    tx_memory_benchmark_{experiment_name or timestamp}/
        config.json         # Benchmark configuration
        results.csv         # Results summary
        tinker.db           # SQLite database
        server_*.log        # Server logs for each test
        xla_dump_*/         # XLA HLO graphs per test (if --dump-xla enabled)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tinker
from tinker import types
from transformers import AutoTokenizer


# Default configuration
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Server configuration
    base_model: str = DEFAULT_BASE_MODEL
    tp_size: int = 8
    max_lora_adapters: int = 2
    train_micro_batch_size: int = 8
    sample_max_num_sequences: int = 32
    shard_attention_heads: bool = True
    enforce_eager: bool = False
    loss_chunk_size: int = 1024
    gradient_checkpointing: bool = True

    # Test configuration
    test_mode: Literal["sample", "train", "both"] = "both"
    batch_sizes: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    seq_lens: list[int] = field(default_factory=lambda: [8192])

    # Runtime configuration
    host: str = "localhost"
    port: int = 8001
    experiment_name: str | None = None
    output_root: Path = field(default_factory=lambda: Path("/tmp/skyrl_tx_memory_benchmark"))
    gpu_poll_interval: float = 1.0

    # JAX/XLA environment configuration
    xla_preallocate: bool = False
    gpu_allocator: str = ""  # Option: "cuda_malloc_async"
    jax_log_compiles: bool = False
    dump_xla: bool = False

    # Derived paths (set after output_dir is created)
    output_dir: Path = field(default_factory=lambda: Path("."))
    db_path: Path = field(default_factory=lambda: Path("/tmp/tinker_bench.db"))
    csv_path: Path = field(default_factory=lambda: Path("results.csv"))


@dataclass
class TestResult:
    """Result from a single benchmark test."""

    mode: str
    batch_size: int
    seq_len: int
    status: Literal["PASS", "FAIL", "ERROR"]
    peak_gpu_mem_mib: int
    jit_logs: list[str]
    client_e2e_sec: float | None
    error_message: str | None = None


class GPUMonitor:
    """Monitor GPU memory usage via nvidia-smi subprocess polling."""

    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_memory: int = 0
        self._lock = threading.Lock()

    def _poll_gpu_memory(self) -> list[int]:
        """Query current GPU memory usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0:
                return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
        return []

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            memory_values = self._poll_gpu_memory()
            if memory_values:
                current_peak = max(memory_values)
                with self._lock:
                    self._peak_memory = max(self._peak_memory, current_peak)
            self._stop_event.wait(self.poll_interval)

    def start(self) -> None:
        """Start background GPU monitoring thread."""
        self._stop_event.clear()
        self._peak_memory = 0
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        """Stop monitoring and return peak memory in MiB."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._lock:
            return self._peak_memory


class ServerManager:
    """Manage TX server subprocess lifecycle."""

    def __init__(self, config: BenchmarkConfig, test_name: str):
        self.config = config
        self.test_name = test_name
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_path = config.output_dir / f"server_{test_name}.log"

    def _build_backend_config(self) -> str:
        """Build backend config JSON from configuration."""
        return json.dumps(
            {
                "tensor_parallel_size": self.config.tp_size,
                "max_lora_adapters": self.config.max_lora_adapters,
                "train_micro_batch_size": self.config.train_micro_batch_size,
                "sample_max_num_sequences": self.config.sample_max_num_sequences,
                "shard_attention_heads": self.config.shard_attention_heads,
                "enforce_eager": self.config.enforce_eager,
                "loss_chunk_size": self.config.loss_chunk_size,
                "gradient_checkpointing": self.config.gradient_checkpointing,
            }
        )

    def _build_command(self) -> list[str]:
        """Build the server launch command."""
        return [
            "uv",
            "run",
            "--extra",
            "tinker",
            "--extra",
            "gpu",
            "-m",
            "tx.tinker.api",
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
            "--base-model",
            self.config.base_model,
            "--database-url",
            f"sqlite:///{self.config.db_path!s}",
            "--database-timeout",
            "300",
            "--backend-config",
            self._build_backend_config(),
        ]

    def start(self) -> None:
        """Start server subprocess."""
        # Clean up old database
        Path(self.config.db_path).unlink(missing_ok=True)

        # Open log file
        self.log_file = open(self.log_path, "w")

        # Set environment variables
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(self.config.xla_preallocate).lower()
        env["TF_GPU_ALLOCATOR"] = self.config.gpu_allocator
        if self.config.jax_log_compiles:
            env["JAX_LOG_COMPILES"] = "1"

        # Set up XLA dump if enabled
        if self.config.dump_xla:
            xla_dump_dir = self.config.output_dir / f"xla_dump_{self.test_name}"
            xla_dump_dir.mkdir(parents=True, exist_ok=True)
            env["XLA_FLAGS"] = f"--xla_dump_to={xla_dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*"

        # Start server
        cmd = self._build_command()
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,  # Create new process group for cleanup
        )

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Wait for server to respond to health check."""
        import httpx

        url = f"http://{self.config.host}:{self.config.port}/api/v1/healthz"
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            time.sleep(1.0)

        return False

    def stop(self) -> None:
        """Stop server subprocess gracefully."""
        if self.process is None:
            return

        # Terminate process group
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass

        # Wait with timeout
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            self.process.wait()

        # Close log file
        if self.log_file:
            self.log_file.close()

        self.process = None

    def get_jit_logs(self) -> list[str]:
        """Extract all JIT compilation log lines from server log."""
        if not self.log_path.exists():
            return []

        logs = []
        pattern = re.compile(r"JIT compilation.*took")
        try:
            with open(self.log_path) as f:
                for line in f:
                    if pattern.search(line):
                        logs.append(line.strip())
        except OSError:
            pass
        return logs

    @contextmanager
    def running(self):
        """Context manager for server lifecycle."""
        try:
            self.start()
            yield self
        finally:
            self.stop()


class BenchmarkRunner:
    """Main benchmark execution orchestrator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    def _kill_existing_processes(self) -> None:
        """Kill any existing server/client processes."""
        subprocess.run(["pkill", "-f", "test_long_sequence.py"], capture_output=True)
        subprocess.run(["pkill", "-f", "tx.tinker.api"], capture_output=True)
        time.sleep(2)

    def _make_datum(self, seq_len: int) -> types.Datum:
        """Create a training datum with specified sequence length."""
        all_tokens = list(range(1, seq_len + 1))
        target_tokens = all_tokens[1:] + [self.tokenizer.eos_token_id]
        weights = [1.0] * seq_len

        return types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights,
            },
        )

    def _test_sample(self, service_client, batch_size: int, seq_len: int) -> tuple[bool, float]:
        """Execute sampling test."""
        sampling_client = service_client.create_sampling_client(base_model=self.config.base_model)

        # Build prompt - half prompt, half generation
        prompt_len = seq_len // 2
        gen_len = seq_len - prompt_len
        base_tokens = self.tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True)
        prompt_tokens = (base_tokens * ((prompt_len // len(base_tokens)) + 1))[:prompt_len]
        prompt = types.ModelInput.from_ints(prompt_tokens)

        # Run sampling
        start_time = time.time()
        request = sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(temperature=0.7, max_tokens=gen_len, seed=42),
            num_samples=batch_size,
        )
        result = request.result()
        elapsed = time.time() - start_time

        return len(result.sequences) == batch_size, elapsed

    def _test_forward_backward(self, service_client, batch_size: int, seq_len: int) -> tuple[bool, float]:
        """Execute forward-backward test."""
        training_client = service_client.create_lora_training_client(base_model=self.config.base_model)

        # Create training data
        data = [self._make_datum(seq_len) for _ in range(batch_size)]

        # Run forward-backward
        start_time = time.time()
        fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
        result = fwdbwd_future.result()
        elapsed = time.time() - start_time

        return len(result.loss_fn_outputs) == batch_size, elapsed

    def _make_server_config(self, batch_size: int, mode: str) -> BenchmarkConfig:
        """Create server config with batch size adjusted for mode."""
        # Create a copy with adjusted batch size
        import copy

        config = copy.copy(self.config)
        if mode == "sample":
            config.sample_max_num_sequences = batch_size
        else:
            config.train_micro_batch_size = batch_size
        return config

    def run_single_test(self, batch_size: int, seq_len: int, mode: str) -> TestResult:
        """Run a single benchmark test with given parameters."""
        # Create server config with appropriate batch size
        server_config = self._make_server_config(batch_size, mode)
        test_name = f"{mode}_bs{batch_size}_seq{seq_len}"
        server = ServerManager(server_config, test_name)
        gpu_monitor = GPUMonitor(self.config.gpu_poll_interval)

        result = TestResult(
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            status="ERROR",
            peak_gpu_mem_mib=0,
            jit_logs=[],
            client_e2e_sec=None,
            error_message=None,
        )

        try:
            # Kill any existing server processes
            self._kill_existing_processes()

            # Start server
            print(f"  Starting server...")
            server.start()

            if not server.wait_ready(timeout=120):
                result.error_message = "Server failed to become ready"
                return result

            print(f"  Server ready, starting GPU monitoring...")

            # Start GPU monitoring
            gpu_monitor.start()

            # Create client and run test
            service_client = tinker.ServiceClient(
                base_url=f"http://{self.config.host}:{self.config.port}",
                api_key="dummy",
            )

            print(f"  Running {mode} test...")
            if mode == "sample":
                success, elapsed = self._test_sample(service_client, batch_size, seq_len)
            else:
                success, elapsed = self._test_forward_backward(service_client, batch_size, seq_len)

            # Collect results
            result.peak_gpu_mem_mib = gpu_monitor.stop()
            result.jit_logs = server.get_jit_logs()
            result.client_e2e_sec = elapsed
            result.status = "PASS" if success else "FAIL"

        except Exception as e:
            result.error_message = str(e)
            result.status = "ERROR"
            gpu_monitor.stop()

        finally:
            server.stop()

        return result

    def run_all_tests(self) -> list[TestResult]:
        """Run all configured benchmark tests."""
        results = []

        modes = ["sample", "train"] if self.config.test_mode == "both" else [self.config.test_mode]

        for mode in modes:
            for seq_len in self.config.seq_lens:
                for batch_size in self.config.batch_sizes:
                    print(f"\n{'='*60}")
                    print(f"Testing: mode={mode}, batch_size={batch_size}, seq_len={seq_len}")
                    print(f"{'='*60}")

                    result = self.run_single_test(batch_size, seq_len, mode)
                    results.append(result)

                    # Print immediate result
                    status_color = "\033[32m" if result.status == "PASS" else "\033[31m"
                    print(f"Result: {status_color}{result.status}\033[0m")
                    print(f"Peak GPU Memory: {result.peak_gpu_mem_mib} MiB")
                    print(f"Client E2E Time: {result.client_e2e_sec or 'N/A'}s")
                    if result.error_message:
                        print(f"Error: {result.error_message}")

                    # If test failed, skip remaining batch sizes for this seq_len
                    if result.status != "PASS":
                        print(f"\nSkipping remaining batch sizes for seq_len={seq_len} due to failure")
                        break

                    # Delay between tests
                    time.sleep(5)

        return results


class ResultsReporter:
    """Generate reports from benchmark results."""

    def __init__(self, results: list[TestResult], output_path: str):
        self.results = results
        self.output_path = output_path

    def write_csv(self) -> None:
        """Write results to CSV file."""
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "mode",
                    "batch_size",
                    "seq_len",
                    "status",
                    "peak_gpu_mem_mib",
                    "client_e2e_sec",
                ]
            )
            for r in self.results:
                writer.writerow(
                    [
                        r.mode,
                        r.batch_size,
                        r.seq_len,
                        r.status,
                        r.peak_gpu_mem_mib,
                        f"{r.client_e2e_sec:.2f}" if r.client_e2e_sec else "",
                    ]
                )

    def print_summary(self) -> None:
        """Print human-readable summary to terminal."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        # Group by mode
        by_mode: dict[str, list[TestResult]] = {}
        for r in self.results:
            by_mode.setdefault(r.mode, []).append(r)

        for mode, mode_results in by_mode.items():
            print(f"\n{mode.upper()} Results:")
            print("-" * 58)
            print(f"{'Batch':>8} {'SeqLen':>8} {'Status':>8} {'PeakMem':>12} {'E2E(s)':>10}")
            print("-" * 58)

            for r in mode_results:
                e2e = f"{r.client_e2e_sec:.2f}" if r.client_e2e_sec else "N/A"
                status_color = "\033[32m" if r.status == "PASS" else "\033[31m"
                print(
                    f"{r.batch_size:>8} {r.seq_len:>8} {status_color}{r.status:>8}\033[0m "
                    f"{r.peak_gpu_mem_mib:>10} MiB {e2e:>10}"
                )

        # Summary statistics
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status in ("FAIL", "ERROR"))
        max_mem = max((r.peak_gpu_mem_mib for r in self.results if r.status == "PASS"), default=0)

        print("\n" + "=" * 70)
        print(f"Total: {len(self.results)} tests | Passed: {passed} | Failed: {failed}")
        print(f"Peak Memory (successful tests): {max_mem} MiB")
        print(f"Results saved to: {Path(self.output_path).resolve()}")
        print("=" * 70)

        # Print JIT compilation logs
        print("\n" + "=" * 70)
        print("JIT COMPILATION LOGS")
        print("=" * 70)
        for r in self.results:
            if r.jit_logs:
                print(f"\n[{r.mode}] batch_size={r.batch_size}, seq_len={r.seq_len}:")
                for log in r.jit_logs:
                    print(f"  {log}")


# Global state for signal handling
_active_server: ServerManager | None = None
_active_monitor: GPUMonitor | None = None


def cleanup_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful cleanup."""
    print("\n\nReceived interrupt signal, cleaning up...")
    if _active_monitor:
        _active_monitor.stop()
    if _active_server:
        _active_server.stop()
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TX Memory Optimization Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server configuration group
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model name")
    server_group.add_argument("--tp-size", type=int, default=8, help="Tensor parallel size")
    server_group.add_argument("--max-lora-adapters", type=int, default=2, help="Max LoRA adapters")
    server_group.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing",
    )
    server_group.add_argument(
        "--loss-chunk-size",
        type=int,
        default=1024,
        help="Cross-entropy loss chunk size",
    )
    server_group.add_argument(
        "--shard-attention-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shard attention heads for TP",
    )
    server_group.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable JIT compilation",
    )

    # Test configuration group
    test_group = parser.add_argument_group("Test Configuration")
    test_group.add_argument(
        "--mode",
        choices=["sample", "train", "both"],
        default="both",
        help="Test mode",
    )
    test_group.add_argument(
        "--batch-sizes",
        default="4,8,16,32",
        type=lambda s: [int(x) for x in s.split(",")],
        help="Comma-separated batch sizes to test",
    )
    test_group.add_argument(
        "--seq-lens",
        default="8192",
        type=lambda s: [int(x) for x in s.split(",")],
        help="Comma-separated sequence lengths to test",
    )

    # Runtime configuration group
    runtime_group = parser.add_argument_group("Runtime Configuration")
    runtime_group.add_argument("--host", default="localhost", help="Server host")
    runtime_group.add_argument("--port", type=int, default=8001, help="Server port")
    runtime_group.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name for output directory (default: timestamp)",
    )
    runtime_group.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/skyrl_tx_memory_benchmark"),
        help="Root directory for benchmark output",
    )
    runtime_group.add_argument(
        "--gpu-poll-interval",
        type=float,
        default=1.0,
        help="GPU memory poll interval in seconds",
    )

    # JAX/XLA environment group
    env_group = parser.add_argument_group("JAX/XLA Environment")
    env_group.add_argument(
        "--xla-preallocate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable XLA memory preallocation",
    )
    env_group.add_argument(
        "--gpu-allocator",
        default="cuda_malloc_async",
        help="GPU allocator (cuda_malloc_async, default, etc.)",
    )
    env_group.add_argument(
        "--jax-log-compiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable JAX compilation logging",
    )
    env_group.add_argument(
        "--dump-xla",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dump XLA HLO graphs to output directory",
    )

    return parser.parse_args()


def setup_output_dir(experiment_name: str | None, output_root: Path) -> Path:
    """Create and return the output directory for this benchmark run."""
    if experiment_name:
        dir_name = f"tx_memory_benchmark_{experiment_name}"
    else:
        dir_name = f"tx_memory_benchmark_{datetime.now():%Y%m%d_%H%M%S}"

    output_dir = output_root / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> int:
    """CLI entry point."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    args = parse_args()

    # Create output directory
    output_dir = setup_output_dir(args.experiment_name, args.output_root)

    # Build configuration with derived paths
    config = BenchmarkConfig(
        base_model=args.base_model,
        tp_size=args.tp_size,
        max_lora_adapters=args.max_lora_adapters,
        gradient_checkpointing=args.gradient_checkpointing,
        loss_chunk_size=args.loss_chunk_size,
        shard_attention_heads=args.shard_attention_heads,
        enforce_eager=args.enforce_eager,
        test_mode=args.mode,
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        host=args.host,
        port=args.port,
        experiment_name=args.experiment_name,
        output_root=args.output_root,
        gpu_poll_interval=args.gpu_poll_interval,
        xla_preallocate=args.xla_preallocate,
        gpu_allocator=args.gpu_allocator,
        jax_log_compiles=args.jax_log_compiles,
        dump_xla=args.dump_xla,
        output_dir=output_dir,
        db_path=output_dir / "tinker.db",
        csv_path=output_dir / "results.csv",
    )

    # Save config to output directory
    config_dict = {
        "base_model": config.base_model,
        "tp_size": config.tp_size,
        "max_lora_adapters": config.max_lora_adapters,
        "gradient_checkpointing": config.gradient_checkpointing,
        "loss_chunk_size": config.loss_chunk_size,
        "shard_attention_heads": config.shard_attention_heads,
        "enforce_eager": config.enforce_eager,
        "test_mode": config.test_mode,
        "batch_sizes": config.batch_sizes,
        "seq_lens": config.seq_lens,
        "output_root": str(config.output_root),
        "xla_preallocate": config.xla_preallocate,
        "gpu_allocator": config.gpu_allocator,
        "jax_log_compiles": config.jax_log_compiles,
        "dump_xla": config.dump_xla,
        "timestamp": datetime.now().isoformat(),
    }
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Print configuration
    print("=" * 60)
    print("TX Memory Optimization Benchmark")
    print("=" * 60)
    print(f"Base Model: {config.base_model}")
    print(f"TP Size: {config.tp_size}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print(f"Loss Chunk Size: {config.loss_chunk_size}")
    print(f"Test Mode: {config.test_mode}")
    print(f"Batch Sizes: {config.batch_sizes}")
    print(f"Sequence Lengths: {config.seq_lens}")
    print(f"Dump XLA: {config.dump_xla}")
    print(f"Output Directory: {config.output_dir.resolve()}")
    print()

    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all_tests()

    # Report results
    reporter = ResultsReporter(results, str(config.csv_path))
    reporter.write_csv()
    reporter.print_summary()

    # Return exit code based on results
    return 0 if all(r.status == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
