"""Shared test utilities and fixtures for tinker tests."""

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Callable

import pytest


def wait_for_condition(
    condition_fn: Callable[[], bool],
    timeout_sec: float = 10,
    poll_interval_sec: float = 0.1,
) -> bool:
    """Poll until condition_fn returns True or timeout is reached. Returns True if condition was met."""
    start_time = time.time()
    while time.time() - start_time < timeout_sec:
        if condition_fn():
            return True
        time.sleep(poll_interval_sec)
    return False


def api_server_is_up(port: int) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/healthz", timeout=2).read()
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
        return False


@contextmanager
def start_api_server(
    *,
    port: int,
    backend: str = "jax",
    base_model: str = "trl-internal-testing/tiny-Qwen3ForCausalLM",
    backend_config: dict | None = None,
    extras: tuple[str, ...] = ("tinker",),
    overrides: dict[str, str] | None = None,
    db_path: str | None = None,
    wait_for_up: bool = False,
    teardown_timeout: float = 5.0,
):
    """Start the Tinker API server in a subprocess and yield ``(process, log_path)``.

    The defaults are a CPU server: the JAX backend, the only one that runs without a GPU,
    with ``backend_config`` defaulting to ``{"max_lora_adapters": 4}``. GPU tests pass
    ``backend``, ``backend_config`` and ``extras`` for their backend. Prints the server log
    on failure.

    Args:
        port: Port the server listens on.
        backend: Training backend (``--backend``).
        base_model: Base model the server loads.
        backend_config: Backend config, serialized to ``--backend-config``.
        extras: uv extras for the server process. The engine subprocess inherits them and
            adds ``tinker`` and the backend's extra (``_build_uv_run_cmd_engine``).
        overrides: Further ``skyrl.tinker.api`` flags, without the leading ``--``.
        db_path: SQLite file to use. Pass one to keep it past the context, e.g. for tests
            that inspect persisted state after server shutdown.
        wait_for_up: Block until ``/api/v1/healthz`` answers, failing the test after 180s.
        teardown_timeout: Seconds to wait for a clean exit before killing the server.
    """
    if backend_config is None:
        backend_config = {"max_lora_adapters": 4}
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = os.path.join(tmp_dir, "server.log")
        resolved_db_path = db_path if db_path is not None else os.path.join(tmp_dir, "server.db")

        with open(log_path, "w") as log_file:
            flags = {
                "host": "0.0.0.0",
                "port": str(port),
                "base-model": base_model,
                "backend": backend,
                "backend-config": json.dumps(backend_config),
                "database-url": f"sqlite:///{resolved_db_path}",
            }
            if overrides:
                flags.update(overrides)
            cmd = ["uv", "run"]
            for extra in extras:
                cmd.extend(["--extra", extra])
            cmd.extend(["-m", "skyrl.tinker.api"])
            for key, value in flags.items():
                cmd.extend([f"--{key}", value])
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            print(f"Starting API server: {' '.join(cmd)}")
            try:
                if wait_for_up:
                    up_port = int(flags["port"])
                    if not wait_for_condition(lambda: api_server_is_up(up_port), timeout_sec=180, poll_interval_sec=2):
                        with open(log_path) as f:
                            print(f"=== Server failed to start ===\n{f.read()}")
                        pytest.fail("Tinker API server did not come up in time")
                yield process, log_path
            except Exception:
                with open(log_path) as f:
                    print(f"=== Test failed. Server log ({log_path}) ===\n{f.read()}")
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=teardown_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
