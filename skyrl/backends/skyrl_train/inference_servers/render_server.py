"""
CPURenderServer - process wrapper around vLLM's GPU-less render server.

vLLM's render app (``vllm launch render``) serves ``/v1/chat/completions/render``
-- chat-template application, tokenization, and multi-modal preprocessing
(image placeholder tokens + serialized ``pixel_values`` / ``image_grid_thw``) --
without creating an engine: no model weights, no KV cache, no GPU memory.
This lets the training path render VLM inputs without initializing any
inference engine.

The stock CLI infers the device from the current platform, which fails on
GPU-less nodes (e.g. a Ray head node) with the CUDA wheel. The child process
here forces vLLM's CPU platform and an explicit ``cpu`` device instead, so the
server runs identically on CPU-only and GPU nodes.

Usage::

    server = CPURenderServer(model_path)
    url = server.start()
    client = RenderServerClient(url)
    # ... VLLMRenderer(client, model_path) ...
    server.shutdown()
"""

import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from skyrl.backends.skyrl_train.inference_servers.common import find_and_reserve_port
from skyrl.env_vars import SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S

logger = logging.getLogger(__name__)

_DEFAULT_START_PORT = 8801
_RENDER_HOST = "127.0.0.1"


def _run_render_server(model_path: str, port: int, log_file: Optional[str]) -> None:
    """Target for the render-server child process.

    Must be top-level for pickling with spawn. All vLLM imports happen here so
    the CPU platform can be seeded before anything resolves it.
    """
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, sys.stdout.fileno())
        os.dup2(fd, sys.stderr.fileno())
        os.close(fd)

    # Rendering is pure CPU preprocessing; hide GPUs so the child can never
    # touch device memory even on GPU nodes.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Force the CPU platform before any other vLLM import resolves it. The
    # CUDA wheel cannot infer a device on a GPU-less Linux box, and with GPUs
    # hidden above the same holds on GPU nodes.
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms._current_platform = CpuPlatform()

    import asyncio

    from vllm import envs
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import (
        build_and_serve_renderer,
        setup_server,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    async def _serve() -> None:
        # Mirrors vLLM's `vllm launch render` (entrypoints/cli/launch.py)
        # except for the explicit cpu DeviceConfig, which the CLI hardcodes
        # to "auto" (platform inference).
        parser = make_arg_parser(FlexibleArgumentParser())
        args = parser.parse_args(
            ["--model", model_path, "--host", _RENDER_HOST, "--port", str(port), "--trust-remote-code"]
        )

        listen_address, sock = setup_server(args)
        engine_args = AsyncEngineArgs.from_cli_args(args)
        model_config = engine_args.create_model_config()
        # Render servers preprocess data only -- no inference, no quantized
        # kernels. Clear quantization so VllmConfig skips quant validation.
        model_config.quantization = None
        # Render servers never allocate KV cache; suppress the spurious CPU
        # KV cache space warning from CpuPlatform.check_and_update_config.
        envs.VLLM_CPU_KVCACHE_SPACE = 0

        vllm_config = VllmConfig(model_config=model_config, device_config=DeviceConfig(device="cpu"))
        shutdown_task = await build_and_serve_renderer(vllm_config, listen_address, sock, args)
        try:
            await shutdown_task
        finally:
            sock.close()

    asyncio.run(_serve())


class CPURenderServer:
    """Process wrapper around vLLM's GPU-less render server.

    The server blocks its event loop while serving, so it runs in a child
    process that can be terminated on ``shutdown()``. Spawn (not fork) is used
    so the child gets a clean interpreter where the CPU platform is seeded
    before any vLLM import resolves it.
    """

    def __init__(self, model_path: str, log_path: Optional[str] = None):
        """
        Args:
            model_path: HuggingFace model name/path whose processor renders inputs.
            log_path: Directory for server log files. When set, a file
                ``render-server-YYMMDD_HHMMSS.log`` is created under this path
                and the child process's stdout/stderr are redirected there.
        """
        self._model_path = model_path
        self._log_path = log_path
        self._process: Optional[multiprocessing.Process] = None

        # Reserve the port to prevent races between discovery and server startup.
        self._port, self._port_reservation = find_and_reserve_port(_DEFAULT_START_PORT)
        logger.info(f"CPURenderServer: model={model_path}, port={self._port}")

    @property
    def url(self) -> str:
        return f"http://{_RENDER_HOST}:{self._port}"

    def start(self) -> str:
        """Spawn the render-server process and return its URL once healthy.

        Raises:
            RuntimeError: If the server process crashes or is not healthy
                within ``SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S``.
        """
        log_file = None
        if self._log_path is not None:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            log_file = str(Path(self._log_path) / f"render-server-{timestamp}.log")

        # Release the port reservation right before the server rebinds.
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None

        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_run_render_server,
            args=(self._model_path, self._port, log_file),
            daemon=True,
            name="cpu-render-server",
        )
        self._process.start()

        deadline = time.monotonic() + SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S
        while time.monotonic() < deadline:
            if not self._process.is_alive():
                raise RuntimeError(
                    f"CPU render server process exited with code {self._process.exitcode} "
                    f"before becoming healthy{f'; see {log_file}' if log_file else ''}"
                )
            try:
                resp = httpx.get(f"{self.url}/health", timeout=5.0)
                if resp.status_code == 200:
                    logger.info(f"CPU render server healthy at {self.url}")
                    return self.url
            except httpx.HTTPError:
                pass
            time.sleep(1.0)

        self.shutdown()
        raise RuntimeError(
            f"CPU render server not healthy within {SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S}s"
            f"{f'; see {log_file}' if log_file else ''}"
        )

    def shutdown(self) -> None:
        """Terminate the render-server process (idempotent)."""
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10)
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=5)
            self._process = None
            logger.info("CPU render server shut down")


class RenderServerClient:
    """Minimal async client for a render server, compatible with ``VLLMRenderer``.

    Implements only ``render_chat_completion`` with the same request/response
    contract as ``RemoteInferenceClient.render_chat_completion``: the payload
    is ``{"json": <OpenAI chat-completion request body>}`` and the response is
    the parsed JSON of ``/v1/chat/completions/render``.
    """

    def __init__(self, url: str, timeout: float = 300.0):
        self._url = url
        self._timeout = timeout

    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = dict(request_payload["json"])
        # Routing hints are meaningless for a single local server.
        body.pop("session_id", None)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._url}/v1/chat/completions/render",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
