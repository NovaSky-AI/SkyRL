"""
VLLMRouter - Process wrapper around vllm_router.Router.

Spawns the router in a child process from a ``RouterArgs`` dataclass,
providing ``start()`` / ``shutdown()`` lifecycle methods.
"""

import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from vllm_router.launch_router import launch_router
from vllm_router.router_args import RouterArgs

from skyrl.backends.skyrl_train.inference_servers.common import (
    find_and_reserve_port,
    get_node_ip,
)
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)


def _run_router_with_logging(router_args: RouterArgs, log_file: Optional[str]) -> None:
    """Target for the router child process.

    Redirects stdout/stderr to *log_file* (when provided) so that the
    Rust router's output is captured instead of lost in the daemon
    process.  Falls back to normal stdout/stderr when *log_file* is
    ``None``.
    """
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, sys.stdout.fileno())
        os.dup2(fd, sys.stderr.fileno())
        os.close(fd)
    launch_router(router_args)


class VLLMRouter:
    """
    Process wrapper around ``vllm_router.Router``.

    ``Router.start()`` blocks and exposes no ``stop()`` method, so we run it
    in a child process that can be terminated on ``shutdown()``.

    Usage::

        from skyrl.backends.skyrl_train.inference_servers.utils import build_router_args

        router_args = build_router_args(ie_cfg, server_urls=urls)
        router = VLLMRouter(router_args)
        router_url = router.start()
        # ... use router_url ...
        router.shutdown()
    """

    _DEFAULT_PROMETHEUS_PORT = 29000

    def __init__(self, router_args: RouterArgs, log_path: Optional[str] = None):
        """
        Args:
            router_args: Configuration for the vllm-router.
            log_path: Directory for router log files.  When set, a file
                ``router-YYMMDD_HHMMSS.log`` is created under this path
                and the child process's stdout/stderr are redirected there.
        """
        self._router_args = router_args
        self._log_path = log_path
        self._log_file: Optional[str] = None
        self._process: Optional[multiprocessing.Process] = None

        # Reserve the router port and prometheus port to prevent race conditions
        # between discovery and actual server startup.
        reserved_port, self._port_reservation = find_and_reserve_port(self._router_args.port)
        self._router_args.port = reserved_port

        prometheus_start = self._router_args.prometheus_port or self._DEFAULT_PROMETHEUS_PORT
        reserved_prom_port, self._prometheus_port_reservation = find_and_reserve_port(prometheus_start)
        self._router_args.prometheus_port = reserved_prom_port

        logger.info(f"VLLMRouter: port={self._router_args.port}, prometheus_port={self._router_args.prometheus_port}")

    def _release_port_reservations(self) -> None:
        """Close any held port reservation sockets."""
        for attr in ("_port_reservation", "_prometheus_port_reservation"):
            sock = getattr(self, attr, None)
            if sock is not None:
                sock.close()
                setattr(self, attr, None)

    def start(self) -> str:
        """Spawn the router process and return the router URL once healthy.

        Returns:
            Router URL, e.g. ``"http://10.0.0.1:30000"``.

        Raises:
            RuntimeError: If the router process crashes before becoming healthy.
        """
        if self._log_path is not None:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            self._log_file = str(Path(self._log_path) / f"router-{timestamp}.log")

        # Release port reservations right before the router rebinds.
        self._release_port_reservations()

        self._process = multiprocessing.Process(
            target=_run_router_with_logging,
            args=(self._router_args, self._log_file),
            daemon=True,
            name="vllm-router",
        )
        self._process.start()

        ip = get_node_ip()
        router_url = f"http://{ip}:{self._router_args.port}"
        self._wait_until_healthy(router_url)

        is_pd = self._router_args.vllm_pd_disaggregation or self._router_args.pd_disaggregation
        if is_pd:
            logger.info(
                f"VLLMRouter (PD) started at {router_url}: "
                f"{len(self._router_args.prefill_urls)} prefill, "
                f"{len(self._router_args.decode_urls)} decode"
            )
        else:
            logger.info(f"VLLMRouter started at {router_url}: " f"{len(self._router_args.worker_urls)} workers")

        if self._log_file:
            logger.info(f"VLLMRouter logs: {self._log_file}")

        return router_url

    # ---------------------------
    # Worker management
    # ---------------------------
    #
    # The router's membership is mutable at runtime; ``worker_urls`` is only the
    # starting set. These three are the synchronous, driver-side mirror of the same
    # calls on ``RemoteInferenceClient`` (which owns the async/data-plane copy) —
    # used by the fault-tolerance path's tests and by anything holding the router
    # object rather than a client.
    #
    # Verified against vllm-router 0.1.14.post1: the backend URL travels as the
    # ``url`` query parameter (any other spelling is a 400 "missing field `url`"),
    # add/remove answer with plain text, and ``/list_workers`` answers
    # ``{"urls": [...]}``.

    def _base_url(self) -> str:
        return f"http://{get_node_ip()}:{self._router_args.port}"

    def add_worker(self, url: str, timeout: float = 10.0) -> None:
        """Register a backend with the router.

        Not idempotent server-side — re-adding a known worker is a 400
        ``already exists`` — so that case is normalized to success here, since the
        post-condition (the worker is routable) holds either way.

        BLOCKING: the router health-checks the backend before accepting it and does
        not answer until it responds, so this cannot pre-register a URL that is not
        yet serving. Call it only once the engine is up.

        Raises:
            RuntimeError: the router rejected the call for any other reason.
            httpx.TimeoutException: the backend never became healthy within *timeout*.
        """
        resp = httpx.post(f"{self._base_url()}/add_worker", params={"url": url}, timeout=timeout)
        if resp.status_code == 400 and "already exists" in resp.text:
            return
        if resp.status_code != 200:
            raise RuntimeError(f"router add_worker({url}) failed [{resp.status_code}]: {resp.text[:200]}")

    def remove_worker(self, url: str, timeout: float = 10.0) -> None:
        """Drop a backend from the router's ring.

        Idempotent server-side: removing a URL the router does not have is a 200.
        ``consistent_hash`` rebuilds the ring on removal, so sessions pinned to the
        removed backend rehash onto survivors.

        Raises:
            RuntimeError: the router rejected the call.
        """
        resp = httpx.post(f"{self._base_url()}/remove_worker", params={"url": url}, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"router remove_worker({url}) failed [{resp.status_code}]: {resp.text[:200]}")

    def list_workers(self, timeout: float = 10.0) -> list[str]:
        """Backend URLs the router currently routes to."""
        resp = httpx.get(f"{self._base_url()}/list_workers", timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        return list(body.get("urls", []) if isinstance(body, dict) else body)

    def _wait_until_healthy(
        self,
        router_url: str,
        timeout: float = SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S,
    ) -> None:
        """Poll the ``/health`` endpoint until the router is ready."""
        health_url = f"{router_url}/health"
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Fail fast if the process died
            if not self._process.is_alive():
                raise RuntimeError(f"VLLMRouter process exited with code {self._process.exitcode}")
            try:
                with httpx.Client() as client:
                    if client.get(health_url, timeout=1).status_code == 200:
                        return
            except httpx.RequestError:
                time.sleep(0.1)
        raise RuntimeError(f"VLLMRouter failed to become healthy within {timeout}s")

    def shutdown(self) -> None:
        """Terminate the router process."""
        # release any port reservations if not already
        self._release_port_reservations()
        # check if router process is active and terminate if needed
        if self._process is None or not self._process.is_alive():
            return
        logger.info("Shutting down VLLMRouter...")
        self._process.terminate()
        self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning("VLLMRouter did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.join()
