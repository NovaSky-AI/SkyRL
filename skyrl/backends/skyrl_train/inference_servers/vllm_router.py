"""
VLLMRouter - Process wrapper around vllm_router.Router.

Spawns the router in a child process from a ``RouterArgs`` dataclass,
providing ``start()`` / ``shutdown()`` lifecycle methods.
"""

import logging
import multiprocessing
import time
from typing import Optional

import httpx
from vllm_router.launch_router import launch_router
from vllm_router.router_args import RouterArgs

from skyrl.backends.skyrl_train.inference_servers.common import get_node_ip
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)


class VLLMRouter:
    """
    Process wrapper around ``vllm_router.Router``.

    ``Router.start()`` blocks and exposes no ``stop()`` method, so we run it
    in a child process that can be terminated on ``shutdown()``.

    Usage::

        from skyrl.backends.skyrl_train.inference_servers.utils import build_router_args

        router_args = build_router_args(cfg, server_urls=urls)
        router = VLLMRouter(router_args)
        router_url = router.start()
        # ... use router_url ...
        router.shutdown()
    """

    def __init__(self, router_args: RouterArgs):
        self._router_args = router_args
        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> str:
        """Spawn the router process and return the router URL once healthy.

        Returns:
            Router URL, e.g. ``"http://10.0.0.1:30000"``.

        Raises:
            RuntimeError: If the router process crashes before becoming healthy.
        """
        self._process = multiprocessing.Process(
            target=launch_router,
            args=(self._router_args,),
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
        return router_url

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
        if self._process is None or not self._process.is_alive():
            return
        logger.info("Shutting down VLLMRouter...")
        self._process.terminate()
        self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning("VLLMRouter did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            self._process.join()
