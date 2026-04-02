"""
VLLMRouter - Subprocess wrapper for vllm-router (data plane only).

Spawns the vllm-router binary as a subprocess with consistent_hash policy,
providing session-aware routing via consistent hashing.
"""

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import List, Optional

import httpx

from skyrl.backends.skyrl_train.inference_servers.common import get_node_ip
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)


class VLLMRouter:
    """
    Subprocess wrapper for vllm-router (data plane only).

    Spawns ``vllm-router`` as a child process with configurable routing policy.
    The default ``consistent_hash`` policy routes requests with the same
    ``X-Session-ID`` header to the same backend, matching SkyRL's session
    routing semantics.

    Usage:
        router = VLLMRouter(server_urls, host="0.0.0.0", port=8080)
        router_url = router.start()
        # ... use router_url for inference ...
        router.shutdown()
    """

    def __init__(
        self,
        server_urls: List[str],
        host: str = "0.0.0.0",
        port: int = 8080,
        policy: str = "consistent_hash",
        health_check_interval_secs: Optional[int] = None,
        max_concurrent_requests: Optional[int] = None,
        request_timeout_secs: Optional[int] = None,
        prometheus_port: Optional[int] = None,
    ):
        self._server_urls = server_urls
        self._host = host
        self._port = port
        self._policy = policy
        self._health_check_interval_secs = health_check_interval_secs
        self._max_concurrent_requests = max_concurrent_requests
        self._request_timeout_secs = request_timeout_secs
        self._prometheus_port = prometheus_port
        self._process: Optional[subprocess.Popen] = None

        logger.info(f"VLLMRouter: {len(server_urls)} servers, port={port}, policy={policy}")

    def _build_cmd(self) -> List[str]:
        """Build the vllm-router CLI command."""
        cmd = [
            "vllm-router",
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--policy",
            self._policy,
            "--worker-urls",
            *self._server_urls,
        ]
        if self._health_check_interval_secs is not None:
            cmd.extend(["--health-check-interval-secs", str(self._health_check_interval_secs)])
        if self._max_concurrent_requests is not None:
            cmd.extend(["--max-concurrent-requests", str(self._max_concurrent_requests)])
        if self._request_timeout_secs is not None:
            cmd.extend(["--request-timeout-secs", str(self._request_timeout_secs)])
        if self._prometheus_port is not None:
            cmd.extend(["--prometheus-port", str(self._prometheus_port)])
        return cmd

    @staticmethod
    def _drain_stream(stream, log_fn):
        """Read lines from a subprocess stream and forward to logger."""
        for line in iter(stream.readline, b""):
            log_fn(f"[vllm-router] {line.decode('utf-8', errors='replace').rstrip()}")
        stream.close()

    def start(self) -> str:
        """
        Start the vllm-router subprocess.

        Returns:
            Router URL (e.g., "http://192.168.1.1:8080")

        Raises:
            ImportError: If ``vllm-router`` binary is not found on PATH.
            RuntimeError: If the router fails to become healthy within the timeout.
        """
        if not self._server_urls:
            raise ValueError("No servers available")

        if shutil.which("vllm-router") is None:
            raise ImportError("vllm-router binary not found on PATH. " "Install it with: pip install vllm-router")

        cmd = self._build_cmd()
        logger.info(f"Starting vllm-router: {' '.join(cmd)}")

        env = os.environ.copy()
        env.setdefault("RUST_LOG", "warn")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Drain subprocess output to prevent pipe buffer blocking
        threading.Thread(
            target=self._drain_stream,
            args=(self._process.stdout, logger.info),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._drain_stream,
            args=(self._process.stderr, logger.warning),
            daemon=True,
        ).start()

        ip = get_node_ip()
        router_url = f"http://{ip}:{self._port}"
        self._wait_until_healthy(router_url)

        logger.info(f"VLLMRouter started at {router_url}")
        return router_url

    def _wait_until_healthy(
        self,
        router_url: str,
        timeout: float = SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S,
    ) -> None:
        """Poll health endpoint until the router is ready."""
        health_url = f"{router_url}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Fail fast if the process exited
            if self._process.poll() is not None:
                raise RuntimeError(f"vllm-router process exited with code {self._process.returncode}")
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"vllm-router failed to start within {timeout}s")

    def shutdown(self) -> None:
        """Shutdown the vllm-router subprocess."""
        if self._process is None or self._process.poll() is not None:
            return
        logger.info("Shutting down vllm-router...")
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("vllm-router did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.wait()
