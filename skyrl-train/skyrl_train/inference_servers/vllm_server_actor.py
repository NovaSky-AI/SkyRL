"""
vLLM Server Actor - Ray actor running a vLLM OpenAI-compatible API server.
"""

import asyncio
import logging
import os
import pickle
import time
from argparse import Namespace
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import Request

from skyrl_train.inference_servers.common import ServerInfo, get_node_ip, get_free_port

logger = logging.getLogger(__name__)


class VLLMServerActor:
    """
    Ray actor that runs a vLLM OpenAI-compatible API server.

    The server runs in the actor and exposes an HTTP endpoint that can be
    called from anywhere (other actors, driver, external processes).

    Custom endpoints added for SkyRL:
    - /init_weight_update_communicator: Initialize weight sync process group
    - /update_weights: Update model weights via NCCL broadcast
    - /finalize_weight_update: Post-processing after weight sync
    - /destroy_weights_update_group: Teardown weight sync
    - /sleep: Offload weights to CPU
    - /wake_up: Load weights back to GPU
    - /reset_prefix_cache: Clear KV cache
    - /get_server_info: Return parallelism info
    """

    @staticmethod
    def compute_num_gpus_per_server(engine_args: Namespace) -> int:
        """Compute the number of GPUs needed per server based on TP * PP."""
        return engine_args.tensor_parallel_size * engine_args.pipeline_parallel_size

    def __init__(
        self,
        engine_args: Namespace,
        start_port: int = 8000,
        server_idx: int = 0,
        dp_size: int = -1,
        dp_master_address: Optional[str] = None,
        dp_rpc_port: Optional[int] = None,
        # PD disaggregation settings
        enable_pd: bool = False,
        nixl_side_channel_base: int = 5600,
    ):
        """
        Initialize the vLLM server actor.

        Args:
            engine_args: vLLM engine configuration (Namespace from make_arg_parser).
                Required attributes: tensor_parallel_size, pipeline_parallel_size, model.
                Optional: uvicorn_log_level, ssl_*, disable_uvicorn_access_log, kv_transfer_config.
            start_port: Base port to start searching for free port
            server_idx: Index of this server in the group
            dp_size: Data parallel size (-1 to disable)
            dp_master_address: DP master address (for non-rank-0 servers)
            dp_rpc_port: DP RPC port (for non-rank-0 servers)
            enable_pd: Enable prefill-decode disaggregation
            nixl_side_channel_base: Base port for NIXL side channel
        """
        self._engine_args = engine_args
        self._ip = get_node_ip()
        self._port = get_free_port(start_port)
        self._server_idx = server_idx
        self._num_gpus_per_server = self.compute_num_gpus_per_server(engine_args)

        # Update args with our assigned host/port
        self._engine_args.host = "0.0.0.0"
        self._engine_args.port = self._port

        # PD disaggregation: setup NIXL side channel for KV transfer
        if enable_pd:
            self._setup_nixl_side_channel(nixl_side_channel_base)

        # Each engine needs to know its dp_rank and dp_size so DP process groups are formed
        if dp_size > 0:
            self._engine_args.data_parallel_size = dp_size
            self._engine_args.data_parallel_rank = server_idx
            # All DP ranks need to know the master's address and RPC port for handshake
            if server_idx == 0:
                dp_master_address, dp_rpc_port = self.get_dp_info()

            if dp_master_address is None or dp_rpc_port is None:
                raise ValueError("DP address and RPC port must be set for non-server 0")

            self._engine_args.data_parallel_address = dp_master_address
            self._engine_args.data_parallel_rpc_port = dp_rpc_port
            logger.info(
                f"Server {server_idx}: DP enabled - dp_size={dp_size}, dp_rank={server_idx}, "
                f"dp_master_address={dp_master_address}, dp_rpc_port={dp_rpc_port}"
            )

        # Compute bundle indices for this server's TP/PP workers
        # Each server uses a contiguous slice of bundles in the placement group
        start_bundle = server_idx * self._num_gpus_per_server
        bundle_indices = list(range(start_bundle, start_bundle + self._num_gpus_per_server))
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"Server {server_idx}: using bundle indices {bundle_indices}")

        self._engine = None
        self._server_task = None

    def _setup_nixl_side_channel(self, base_port: int) -> None:
        """
        Setup NIXL side channel for PD disaggregation.

        Each server instance needs a unique side channel port for KV transfer handshake.
        """
        import json

        side_channel_port = base_port + self._server_idx
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_channel_port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self._ip

        engine_id = f"server-{self._server_idx}-{self._ip}-{side_channel_port}"

        if hasattr(self._engine_args, "kv_transfer_config") and self._engine_args.kv_transfer_config:
            try:
                kv_config = json.loads(self._engine_args.kv_transfer_config)
                kv_config["engine_id"] = engine_id
                self._engine_args.kv_transfer_config = json.dumps(kv_config)
            except (json.JSONDecodeError, TypeError):
                pass

        logger.info(
            f"Server {self._server_idx}: NIXL side channel configured - "
            f"host={self._ip}, port={side_channel_port}, engine_id={engine_id}"
        )

    def get_server_info(self) -> ServerInfo:
        """Get the server's IP and port info."""
        return ServerInfo(ip=self._ip, port=self._port)

    def get_extended_server_info(self) -> Dict[str, Any]:
        """Return extended server info including parallelism settings."""
        return {
            "ip": self._ip,
            "port": self._port,
            "url": f"http://{self._ip}:{self._port}",
            "server_idx": self._server_idx,
            "tp_size": self._engine_args.tensor_parallel_size,
            "pp_size": self._engine_args.pipeline_parallel_size,
            "dp_size": getattr(self._engine_args, "data_parallel_size", 1),
            "world_size": self._num_gpus_per_server,
        }

    def get_dp_info(self) -> tuple:
        """Get the DP master address and RPC port (for server 0 to share with others)."""
        dp_rpc_port = self._port + 500
        return (self._ip, dp_rpc_port)

    async def start(self) -> ServerInfo:
        """Start the vLLM server. Blocks until server is healthy."""
        from vllm.utils.system_utils import set_ulimit

        set_ulimit()
        logger.info(f"Starting server on {self._ip}:{self._port}...")

        # Start HTTP server as background asyncio task
        self._server_task = asyncio.create_task(self._run_server())

        # Wait until the server is actually healthy
        await self._wait_until_healthy()

        return self.get_server_info()

    async def _wait_until_healthy(self, timeout: float = 600, interval: float = 1.0) -> None:
        """Poll the /health endpoint until it responds OK."""
        url = f"http://{self._ip}:{self._port}/health"
        start_time = time.time()

        async with httpx.AsyncClient() as client:
            while True:
                # Check if server task failed
                if self._server_task.done():
                    exc = self._server_task.exception()
                    if exc:
                        raise exc
                    raise RuntimeError("Server task exited unexpectedly")

                try:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        logger.info(f"Server {self._ip}:{self._port} is healthy")
                        return
                except httpx.RequestError:
                    pass

                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Server failed to become healthy within {timeout}s")

                await asyncio.sleep(interval)

    async def _run_server(self) -> None:
        """Internal method to run the HTTP server."""
        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import build_app, create_server_socket, init_app_state
        from vllm.usage.usage_lib import UsageContext
        import vllm.envs as envs

        sock_addr = (self._engine_args.host, self._engine_args.port)
        sock = create_server_socket(sock_addr)
        app = build_app(self._engine_args)

        # Initialize the engine (this loads the model - takes time)
        engine_args = AsyncEngineArgs.from_cli_args(self._engine_args)
        self._engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        logger.info(f"Engine initialized on {self._ip}:{self._port}, adding custom endpoints...")

        # Add custom SkyRL endpoints
        self._add_custom_endpoints(app)

        await init_app_state(self._engine, app.state, self._engine_args)

        # Use uvicorn directly (serve_http tries to add signal handlers which fails in Ray actors)
        config = uvicorn.Config(
            app,
            host=self._engine_args.host,
            port=self._engine_args.port,
            log_level=self._engine_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self._engine_args.ssl_keyfile,
            ssl_certfile=self._engine_args.ssl_certfile,
            ssl_ca_certs=self._engine_args.ssl_ca_certs,
            ssl_cert_reqs=self._engine_args.ssl_cert_reqs,
            access_log=not getattr(self._engine_args, "disable_uvicorn_access_log", False),
        )
        server = uvicorn.Server(config)
        await server.serve(sockets=[sock])

    def _add_custom_endpoints(self, app) -> None:
        """Add custom SkyRL endpoints to the FastAPI app."""
        engine = self._engine

        @app.get("/get_server_info")
        async def _get_server_info():
            """Return server parallelism info."""
            return self.get_extended_server_info()

        @app.post("/init_weight_update_communicator")
        async def _init_weight_update_communicator(request: Request):
            """Initialize weight sync process group."""
            from skyrl_train.weight_sync import BroadcastInitInfo

            data = await request.json()
            init_info = BroadcastInitInfo(**data)
            pickled_init_info = pickle.dumps(init_info)

            await engine.collective_rpc(
                "init_weight_update_communicator",
                args=(pickled_init_info,),
            )
            return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            """Update model weights via NCCL broadcast."""
            from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

            data = await request.json()
            weight_request = BroadcastWeightUpdateRequest(**data)
            pickled_request = pickle.dumps(weight_request)

            await engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
            return {"status": "ok"}

        @app.post("/finalize_weight_update")
        async def _finalize_weight_update(request: Request):
            """
            Finalize weight update - post-processing hook.

            Currently a no-op, reserved for future use (cache invalidation, etc).
            """
            # No-op for now - placeholder for future post-processing
            return {"status": "ok"}

        @app.post("/destroy_weights_update_group")
        async def _destroy_weights_update_group(request: Request):
            """Teardown weight sync process group."""
            await engine.collective_rpc(
                "teardown_weight_receiver",
                args=(),
            )
            return {"status": "ok"}

        @app.post("/sleep")
        async def _sleep(request: Request):
            """Offload weights to CPU."""
            data = await request.json()
            level = data.get("level", 1)

            # Reset prefix cache before sleep to avoid gibberish on wake
            # See: https://github.com/vllm-project/vllm/issues/17103
            await engine.reset_prefix_cache()
            await engine.sleep(level)
            return {"status": "ok"}

        @app.post("/wake_up")
        async def _wake_up(request: Request):
            """Load weights back to GPU."""
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/wakeup")
        async def _wakeup(request: Request):
            """Alias for /wake_up."""
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            """Clear KV cache."""
            data = await request.json()
            reset_running = data.get("reset_running_requests", False)
            if reset_running:
                # If reset_running_requests is True, we need to abort first
                await engine.abort_all_requests()
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        @app.post("/pause")
        async def _pause(request: Request):
            """Pause generation."""
            data = await request.json()
            wait_for_inflight = data.get("wait_for_inflight_request", False)
            # vLLM's pause API - implementation depends on vLLM version
            if hasattr(engine, "pause"):
                await engine.pause(wait_for_inflight_request=wait_for_inflight)
            else:
                # Fallback: abort all if pause not available
                if not wait_for_inflight:
                    await engine.abort_all_requests()
            return {"status": "ok"}

        @app.post("/resume")
        async def _resume(request: Request):
            """Resume generation."""
            if hasattr(engine, "resume"):
                await engine.resume()
            return {"status": "ok"}

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
