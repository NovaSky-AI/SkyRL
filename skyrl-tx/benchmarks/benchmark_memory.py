#!/usr/bin/env python3
"""
Benchmark GPU memory (GRAM) usage for skyrl-tx training and sampling.

This script measures peak GPU memory consumption across different batch sizes
and sequence lengths for both sampling (inference) and training (forward-backward)
workloads. It automatically manages server lifecycle, monitors GPU memory via
nvidia-smi, and generates detailed reports.

Features:
    - Test sampling, training, or both modes
    - Sweep across multiple batch sizes and sequence lengths
    - Separate JIT compilation time vs post-JIT runtime measurement
    - Configurable number of post-JIT measurement iterations for averaging
    - Early termination: skips remaining batch sizes if one fails (e.g., OOM)
    - GPU memory monitoring via nvidia-smi polling
    - Per-test server logs with JIT compilation time extraction
    - Optional XLA HLO graph dumps for debugging
    - Configurable JAX/XLA environment variables
    - Pass-through additional backend config options via JSON
    - Server-only mode for manual testing

Usage:
    # Run full benchmark sweep
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --experiment-name my_test --mode both --batch-sizes 4,8,16,32 --seq-lens 4096,8192

    # Test sampling only with specific config
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --mode sample --batch-sizes 32,64 --seq-lens 8192 --tp-size 8

    # Launch server only for manual testing
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --server-only --batch-sizes 8 --seq-lens 8192

    # Enable XLA graph dumps for debugging
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --dump-xla --batch-sizes 4 --seq-lens 4096

    # Pass additional backend config options
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --backend-config '{"loss_chunk_size": 512, "enforce_eager": true}'

    # Run with multiple measurement iterations for more accurate post-JIT timing
    uv run --extra tinker python benchmarks/benchmark_memory.py \\
        --num-measurement-iters 3 --batch-sizes 8 --seq-lens 4096

Output directory (default: /tmp/skyrl_tx_memory_benchmark/):
    tx_memory_benchmark_{experiment_name}_{timestamp}/
        config.json         # Full benchmark configuration (JSON)
        results.csv         # Results table (mode, batch, seq, status, peak_mem, jit_time, post_jit_time)
        tinker.db           # SQLite database used by tinker API
        server_*.log        # Server stdout/stderr for each test run
        xla_dump_*/         # XLA HLO graphs per test (if --dump-xla enabled)

Exit codes:
    0 - All tests passed (or server-only mode completed)
    1 - One or more tests failed or errored
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import httpx
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
    gradient_checkpointing: bool = True
    train_micro_batch_size: int | None = None  # If None, uses batch_size
    sample_max_num_sequences: int | None = None  # If None, uses batch_size
    extra_backend_config: dict = field(default_factory=dict)  # Additional backend config options

    # Test configuration
    test_mode: Literal["sample", "train", "both"] = "both"
    batch_sizes: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    seq_lens: list[int] = field(default_factory=lambda: [8192])
    num_measurement_iters: int = 3  # Number of post-JIT measurement iterations
    server_only: bool = False

    # Runtime configuration
    host: str = "localhost"
    port: int = 8001
    experiment_name: str | None = None
    output_root: Path = field(default_factory=lambda: Path("/tmp/skyrl_tx_memory_benchmark"))
    gpu_poll_interval: float = 1.0

    # JAX/XLA environment configuration
    xla_preallocate: bool = False
    gpu_allocator: str = ""  # TF_GPU_ALLOCATOR (e.g., "cuda_malloc_async")
    jax_log_compiles: bool = False
    dump_xla: bool = False

    # Derived paths (set after output_dir is created)
    output_dir: Path = field(default_factory=lambda: Path("."))
    db_path: Path = field(default_factory=lambda: Path("/tmp/tinker_bench.db"))
    csv_path: Path = field(default_factory=lambda: Path("results.csv"))

    def __str__(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.__dict__.items()]
        return "\n".join(lines)


@dataclass
class TestResult:
    """Result from a single benchmark test."""

    mode: str
    batch_size: int
    seq_len: int
    status: Literal["PASS", "FAIL", "ERROR"]
    peak_gpu_mem_mib: int
    jit_logs: list[str]
    jit_e2e_sec: float | None  # First request (includes JIT compilation)
    post_jit_e2e_sec: float | None  # Average of subsequent requests (post-JIT)
    error_message: str | None = None


class GPUMonitor:
    """Monitor GPU memory usage via nvidia-smi subprocess polling."""

    @staticmethod
    def check_nvidia_smi() -> bool:
        """Check if nvidia-smi is available. Returns True if available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

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

    def __init__(self, config: BenchmarkConfig, test_name: str, batch_size: int, mode: str | None = None):
        """Initialize server manager.

        Args:
            config: Benchmark configuration
            test_name: Name for log files
            batch_size: Batch size for this test
            mode: "sample", "train", or None (sets both for server-only mode)
        """
        self.config = config
        self.test_name = test_name
        self.batch_size = batch_size
        self.mode = mode
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_path = config.output_dir / f"server_{test_name}.log"

    def _build_backend_config(self) -> str:
        """Build backend config JSON from configuration."""
        # Use CLI overrides if set, otherwise use batch_size
        train_micro_batch_size = self.config.train_micro_batch_size or self.batch_size
        sample_max_num_sequences = self.config.sample_max_num_sequences or self.batch_size

        config = {
            "tensor_parallel_size": self.config.tp_size,
            "max_lora_adapters": self.config.max_lora_adapters,
            "train_micro_batch_size": train_micro_batch_size,
            "sample_max_num_sequences": sample_max_num_sequences,
            "gradient_checkpointing": self.config.gradient_checkpointing,
        }
        # Merge extra backend config (allows overriding defaults)
        config.update(self.config.extra_backend_config)
        return json.dumps(config)

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
            "--backend-config",
            self._build_backend_config(),
        ]

    def start(self) -> None:
        """Start server subprocess."""
        # Clean up old database
        Path(self.config.db_path).unlink(missing_ok=True)

        try:
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
        except Exception:
            if self.log_file:
                self.log_file.close()
                self.log_file = None
            raise

    def is_alive(self) -> bool:
        """Check if server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Wait for server to respond to health check."""
        url = f"http://{self.config.host}:{self.config.port}/api/v1/healthz"
        deadline = time.time() + timeout

        while time.time() < deadline:
            # Check if process died
            if not self.is_alive():
                return False
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    return True
            except httpx.RequestError:
                pass
            time.sleep(1.0)

        return False

    def start_and_wait_ready(self, timeout: float = 120.0, stream_logs: bool = False) -> bool:
        """Start server and wait for it to become ready.

        Args:
            timeout: Maximum time to wait for server to be ready
            stream_logs: If True, stream log output while waiting

        Returns:
            True if server is ready, False if timeout
        """
        self.start()

        if not stream_logs:
            return self.wait_ready(timeout=timeout)

        # Stream logs while waiting for server
        with open(self.log_path, "r") as log:
            deadline = time.time() + timeout
            last_check = 0

            while time.time() < deadline:
                # Check if process died
                if not self.is_alive():
                    # Drain remaining logs
                    for line in log:
                        print(line, end="", flush=True)
                    return False

                line = log.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.05)

                # Check if server is ready (every 2 seconds)
                if time.time() - last_check > 2:
                    last_check = time.time()
                    if self.wait_ready(timeout=0.5):
                        return True

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

    def print_last_logs(self, n: int = 30) -> None:
        """Print last N lines of server log."""
        if not self.log_path.exists():
            return
        print(f"\n  Server log: {self.log_path}")
        print("  " + "-" * 40)
        with open(self.log_path, "r") as f:
            lines = f.readlines()
            for line in lines[-n:]:
                print(f"  {line}", end="")
        print("  " + "-" * 40)

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


class BenchmarkRunner:
    """Main benchmark execution orchestrator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)

    def _make_datum(self, seq_len: int, rng: random.Random) -> types.Datum:
        """Create a training datum with random tokens."""
        vocab_size = self.tokenizer.vocab_size
        all_tokens = [rng.randint(1, vocab_size - 1) for _ in range(seq_len)]
        target_tokens = all_tokens[1:] + [self.tokenizer.eos_token_id]
        weights = [1.0] * seq_len

        return types.Datum(
            model_input=types.ModelInput.from_ints(all_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "weights": weights,
            },
        )

    def _run_timed_requests(
        self,
        run_single_request: Callable[[], tuple[bool, float]],
        num_measurement_iters: int,
    ) -> tuple[bool, float, float]:
        """Run timed requests with JIT warmup and measurement iterations.

        Args:
            run_single_request: Callable that executes a single request and returns (success, elapsed_time)
            num_measurement_iters: Number of post-JIT measurement iterations

        Returns:
            Tuple of (success, jit_time, post_jit_avg_time)
        """
        # First request triggers JIT compilation
        print("    Running warmup request (JIT compilation)...")
        success, jit_time = run_single_request()
        if not success:
            return False, jit_time, 0.0

        # Subsequent requests measure post-JIT performance
        post_jit_times = []
        for i in range(num_measurement_iters):
            print(f"    Running measurement request {i + 1}/{num_measurement_iters}...")
            success, elapsed = run_single_request()
            if not success:
                return False, jit_time, 0.0
            post_jit_times.append(elapsed)

        avg_post_jit_time = sum(post_jit_times) / len(post_jit_times) if post_jit_times else 0.0
        return True, jit_time, avg_post_jit_time

    def _test_sample(
        self, service_client, server: ServerManager, batch_size: int, seq_len: int, num_measurement_iters: int
    ) -> tuple[bool, float, float]:
        """Execute sampling test with warmup iterations."""
        sampling_client = service_client.create_sampling_client(base_model=self.config.base_model)
        vocab_size = self.tokenizer.vocab_size
        rng = random.Random(42)

        # Half prompt, half generation
        prompt_len = seq_len // 2
        gen_len = seq_len - prompt_len

        def run_sample() -> tuple[bool, float]:
            """Run a single sample request and return (success, elapsed_time)."""
            prompt_tokens = [rng.randint(1, vocab_size - 1) for _ in range(prompt_len)]
            prompt = types.ModelInput.from_ints(prompt_tokens)

            start_time = time.time()
            request = sampling_client.sample(
                prompt=prompt,
                sampling_params=types.SamplingParams(temperature=0.7, max_tokens=gen_len, seed=42),
                num_samples=batch_size,
            )
            # Poll with small timeout to allow server aliveness checks
            while True:
                try:
                    result = request.result(timeout=5)
                    break
                except TimeoutError:
                    if not server.is_alive():
                        raise RuntimeError("Server crashed during test")
            elapsed = time.time() - start_time
            return len(result.sequences) == batch_size, elapsed

        return self._run_timed_requests(run_sample, num_measurement_iters)

    def _test_forward_backward(
        self, service_client, server: ServerManager, batch_size: int, seq_len: int, num_measurement_iters: int
    ) -> tuple[bool, float, float]:
        """Execute forward-backward test with warmup iterations."""
        training_client = service_client.create_lora_training_client(base_model=self.config.base_model)
        rng = random.Random(42)

        def run_forward_backward() -> tuple[bool, float]:
            """Run a single forward-backward request and return (success, elapsed_time)."""
            data = [self._make_datum(seq_len, rng) for _ in range(batch_size)]

            start_time = time.time()
            fwdbwd_future = training_client.forward_backward(data, "cross_entropy")
            # Poll with small timeout to allow server aliveness checks
            while True:
                try:
                    result = fwdbwd_future.result(timeout=5)
                    break
                except TimeoutError:
                    if not server.is_alive():
                        raise RuntimeError("Server crashed during test")
            elapsed = time.time() - start_time
            return len(result.loss_fn_outputs) == batch_size, elapsed

        return self._run_timed_requests(run_forward_backward, num_measurement_iters)

    def run_single_test(self, batch_size: int, seq_len: int, mode: str) -> TestResult:
        """Run a single benchmark test with given parameters."""
        test_name = f"{mode}_seq{seq_len}_bs{batch_size}"
        server = ServerManager(self.config, test_name, batch_size, mode)
        gpu_monitor = GPUMonitor(self.config.gpu_poll_interval)

        result = TestResult(
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            status="ERROR",
            peak_gpu_mem_mib=0,
            jit_logs=[],
            jit_e2e_sec=None,
            post_jit_e2e_sec=None,
            error_message=None,
        )

        try:
            # Start server (kills existing servers first)
            print("  Starting server...")
            if not server.start_and_wait_ready(timeout=120):
                if not server.is_alive():
                    result.error_message = "Server crashed during startup"
                    server.print_last_logs()
                else:
                    result.error_message = "Server timed out during startup"
                return result

            print("  Server ready, starting GPU monitoring...")

            # Start GPU monitoring
            gpu_monitor.start()

            # Create client and run test
            service_client = tinker.ServiceClient(
                base_url=f"http://{self.config.host}:{self.config.port}",
                api_key="tml-dummy",
            )

            try:
                print(f"  Running {mode} test...")
                if mode == "sample":
                    success, jit_time, post_jit_time = self._test_sample(
                        service_client, server, batch_size, seq_len, self.config.num_measurement_iters
                    )
                else:
                    success, jit_time, post_jit_time = self._test_forward_backward(
                        service_client, server, batch_size, seq_len, self.config.num_measurement_iters
                    )

                # Collect results
                result.peak_gpu_mem_mib = gpu_monitor.stop()
                result.jit_logs = server.get_jit_logs()
                result.jit_e2e_sec = jit_time
                result.post_jit_e2e_sec = post_jit_time
                result.status = "PASS" if success else "FAIL"
            finally:
                # Close client to stop heartbeat thread before server shutdown
                service_client.holder.close()

        except Exception as e:
            if not server.is_alive():
                result.error_message = f"Server crashed: {e}"
                server.print_last_logs()
            else:
                result.error_message = str(e)
            result.status = "ERROR"
            result.peak_gpu_mem_mib = gpu_monitor.stop()

        finally:
            server.stop()

        return result

    def run_all_tests(self, results_writer: ResultsWriter | None = None) -> list[TestResult]:
        """Run all configured benchmark tests.

        Args:
            results_writer: Optional writer for incremental CSV output
        """
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

                    # Write result to CSV immediately
                    if results_writer:
                        results_writer.append(result)

                    # Print immediate result
                    status_color = "\033[32m" if result.status == "PASS" else "\033[31m"
                    print(f"Result: {status_color}{result.status}\033[0m")
                    print(f"Peak GPU Memory: {result.peak_gpu_mem_mib} MiB")
                    jit_str = f"{result.jit_e2e_sec:.2f}s" if result.jit_e2e_sec else "N/A"
                    post_jit_str = f"{result.post_jit_e2e_sec:.2f}s" if result.post_jit_e2e_sec else "N/A"
                    print(f"JIT Time (1st request): {jit_str}")
                    print(f"Post-JIT Time (avg): {post_jit_str}")
                    if result.error_message:
                        print(f"Error: {result.error_message}")

                    # If test failed, skip remaining batch sizes for this seq_len
                    if result.status != "PASS":
                        print(f"\nSkipping remaining batch sizes for seq_len={seq_len} due to failure")
                        break

                    # Delay between tests
                    time.sleep(5)

        return results


class ResultsWriter:
    """Write benchmark results to CSV incrementally."""

    CSV_HEADER = ["mode", "batch_size", "seq_len", "status", "peak_gpu_mem_mib", "jit_e2e_sec", "post_jit_e2e_sec"]

    def __init__(self, output_path: Path):
        self.output_path = output_path
        # Write header immediately
        with open(self.output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_HEADER)

    def append(self, result: TestResult) -> None:
        """Append a single result to the CSV file."""
        with open(self.output_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    result.mode,
                    result.batch_size,
                    result.seq_len,
                    result.status,
                    result.peak_gpu_mem_mib,
                    f"{result.jit_e2e_sec:.2f}" if result.jit_e2e_sec else "",
                    f"{result.post_jit_e2e_sec:.2f}" if result.post_jit_e2e_sec else "",
                ]
            )


class ResultsReporter:
    """Generate reports from benchmark results."""

    def __init__(self, results: list[TestResult], output_path: str):
        self.results = results
        self.output_path = output_path

    def print_summary(self) -> None:
        """Print human-readable summary to terminal."""
        print("\n" + "=" * 85)
        print("BENCHMARK SUMMARY")
        print("=" * 85)

        # Group by mode
        by_mode: dict[str, list[TestResult]] = {}
        for r in self.results:
            by_mode.setdefault(r.mode, []).append(r)

        for mode, mode_results in by_mode.items():
            print(f"\n{mode.upper()} Results:")
            print("-" * 75)
            print(f"{'Batch':>8} {'SeqLen':>8} {'Status':>8} {'PeakMem':>12} {'JIT(s)':>10} {'PostJIT(s)':>12}")
            print("-" * 75)

            for r in mode_results:
                jit = f"{r.jit_e2e_sec:.2f}" if r.jit_e2e_sec else "N/A"
                post_jit = f"{r.post_jit_e2e_sec:.2f}" if r.post_jit_e2e_sec else "N/A"
                status_color = "\033[32m" if r.status == "PASS" else "\033[31m"
                print(
                    f"{r.batch_size:>8} {r.seq_len:>8} {status_color}{r.status:>8}\033[0m "
                    f"{r.peak_gpu_mem_mib:>10} MiB {jit:>10} {post_jit:>12}"
                )

        # Summary statistics
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status in ("FAIL", "ERROR"))
        max_mem = max((r.peak_gpu_mem_mib for r in self.results if r.status == "PASS"), default=0)

        print("\n" + "=" * 85)
        print(f"Total: {len(self.results)} tests | Passed: {passed} | Failed: {failed}")
        print(f"Peak Memory (successful tests): {max_mem} MiB")
        print(f"Results saved to: {Path(self.output_path).resolve()}")
        print("=" * 85)

        # Print JIT compilation logs
        print("\n" + "=" * 70)
        print("JIT COMPILATION LOGS")
        print("=" * 70)
        for r in self.results:
            if r.jit_logs:
                print(f"\n[{r.mode}] batch_size={r.batch_size}, seq_len={r.seq_len}:")
                for log in r.jit_logs:
                    print(f"  {log}")


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
        "--train-micro-batch-size",
        type=int,
        default=None,
        help="Override train micro batch size (default: use --batch-sizes value)",
    )
    server_group.add_argument(
        "--sample-max-num-sequences",
        type=int,
        default=None,
        help="Override sample max num sequences (default: use --batch-sizes value)",
    )
    server_group.add_argument(
        "--backend-config",
        type=json.loads,
        default={},
        help="Additional backend config as JSON (e.g., '{\"key\": value}')",
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
    test_group.add_argument(
        "--num-measurement-iters",
        type=int,
        default=3,
        help="Number of post-JIT measurement iterations (default: 3)",
    )
    test_group.add_argument(
        "--server-only",
        action="store_true",
        help="Only launch the server without running tests (for manual testing)",
    )

    # Runtime configuration group
    runtime_group = parser.add_argument_group("Runtime Configuration")
    runtime_group.add_argument("--host", default="localhost", help="Server host")
    runtime_group.add_argument("--port", type=int, default=8001, help="Server port")
    runtime_group.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name prefix for output directory (format: {name}_{timestamp})",
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
        default="",
        help="TF_GPU_ALLOCATOR value (e.g., 'cuda_malloc_async' for async memory pools)",
    )
    env_group.add_argument(
        "--jax-log-compiles",
        action=argparse.BooleanOptionalAction,
        default=False,
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
    """Create and return the output directory for this benchmark run.

    Directory name format: tx_memory_benchmark_{experiment_name}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        dir_name = f"tx_memory_benchmark_{experiment_name}_{timestamp}"
    else:
        dir_name = f"tx_memory_benchmark_{timestamp}"

    output_dir = output_root / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    # Check nvidia-smi availability
    if not GPUMonitor.check_nvidia_smi():
        print("WARNING: nvidia-smi not available, GPU memory monitoring will report 0 MiB", file=sys.stderr)

    # Create output directory
    output_dir = setup_output_dir(args.experiment_name, args.output_root)

    # Build configuration from args
    # Map CLI arg names to config field names where they differ
    arg_renames = {"mode": "test_mode", "backend_config": "extra_backend_config"}
    config_fields = {f.name for f in BenchmarkConfig.__dataclass_fields__.values()}

    config_kwargs = {}
    for key, value in vars(args).items():
        config_key = arg_renames.get(key, key)
        if config_key in config_fields:
            config_kwargs[config_key] = value

    # Add derived paths
    config_kwargs["output_dir"] = output_dir
    config_kwargs["db_path"] = output_dir / "tinker.db"
    config_kwargs["csv_path"] = output_dir / "results.csv"

    config = BenchmarkConfig(**config_kwargs)

    # Save config to output directory
    config_dict = asdict(config)
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    config_dict["timestamp"] = datetime.now().isoformat()
    with open(config.output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print("=" * 60)
    print("TX Memory Optimization Benchmark")
    print("=" * 60)
    print(config)
    print()

    # Server-only mode: launch server and stream logs
    if config.server_only:
        first_batch_size = config.batch_sizes[0]
        test_name = f"server_only_bs{first_batch_size}"

        print("Server-only mode: launching server...")
        print(f"Batch size: {first_batch_size}")
        print(f"Log file: {config.output_dir.resolve()}/server_{test_name}.log")
        print("-" * 60)

        server = ServerManager(config, test_name, first_batch_size)  # mode=None sets both
        try:
            # Start server and stream logs until ready
            if not server.start_and_wait_ready(timeout=300, stream_logs=True):
                print("\nERROR: Server failed to become ready")
                return 1

            print("-" * 60)
            print(f"Server ready at http://{config.host}:{config.port}")
            print("-" * 60, flush=True)

            # Continue streaming logs until interrupted
            with open(server.log_path, "r") as log:
                log.seek(0, 2)  # Seek to end
                while True:
                    line = log.readline()
                    if line:
                        print(line, end="", flush=True)
                    else:
                        time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n" + "-" * 60)
            print("Shutting down server...")
        finally:
            server.stop()
        return 0

    # Run benchmarks with incremental CSV output
    results_writer = ResultsWriter(config.csv_path)
    runner = BenchmarkRunner(config)
    results = runner.run_all_tests(results_writer)

    # Report results
    reporter = ResultsReporter(results, str(config.csv_path))
    reporter.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
