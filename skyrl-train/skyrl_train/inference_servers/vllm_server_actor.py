"""
vLLM Server Actor - Ray actor running a vLLM OpenAI-compatible API server.
"""

import asyncio
import logging
import os
import pickle
import time
from argparse import Namespace
from typing import Any, Dict, Optional, Tuple

import httpx
import uvicorn
import vllm.envs as envs
from fastapi import Request, HTTPException
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.usage.usage_lib import UsageContext
from vllm.utils.system_utils import set_ulimit

from skyrl_train.env_vars import (
    SKYRL_VLLM_DP_PORT_OFFSET,
    SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S,
)
from skyrl_train.inference_servers.common import ServerInfo, get_node_ip, get_open_port
from skyrl_train.inference_servers.protocols import ServerActorProtocol
from skyrl_train.inference_servers.vllm_worker import VLLM_WORKER_EXTENSION_CLS

logger = logging.getLogger(__name__)


class VLLMServerActor(ServerActorProtocol):
    """
    Ray actor that runs a vLLM OpenAI-compatible API server.

    Implements ServerActorProtocol for use with ServerGroup.

    The server runs in the actor and exposes an HTTP endpoint that can be
    called from anywhere (other actors, driver, external processes).

    Custom endpoints added for SkyRL:
    - /get_server_info: Return parallelism info

    - (vLLM RFC: https://github.com/vllm-project/vllm/issues/31848)
    - /init_weight_transfer: Initialize weight sync process group
    - /update_weights: Update model weights via NCCL broadcast
    - /finalize_weight_update: Post-processing after weight sync
    """

    @staticmethod
    def compute_num_gpus_per_server(vllm_cli_args: Namespace) -> int:
        """Compute the number of GPUs needed per server based on TP * PP.

        This logic might need adjustment if we want to support other
        parallelism schemes. If we get to this point, we should add a
        vllm-specific utility for it and keep the logic inside the engine.
        """
        return vllm_cli_args.tensor_parallel_size * vllm_cli_args.pipeline_parallel_size

    def __init__(
        self,
        vllm_cli_args: Namespace,
        start_port: int = 8000,
        server_idx: int = 0,
        start_bundle_idx: int = 0,
        dp_size: int = -1,
        dp_master_address: Optional[str] = None,
        dp_rpc_port: Optional[int] = None,
        # PD disaggregation settings
        enable_pd: bool = False,
        nixl_side_channel_base: int = 5600,
        colocated_training: bool = False,
    ):
        """
        Initialize the vLLM server actor.

        Args:
            vllm_cli_args: vLLM CLI arguments.
                Required attributes: tensor_parallel_size, pipeline_parallel_size.
                Optional: uvicorn_log_level, ssl_*, disable_uvicorn_access_log, kv_transfer_config.
            start_port: Base port to start searching for free port
            server_idx: Index of this server in the group
            start_bundle_idx: Starting bundle index in the placement group for this server's workers
            dp_size: Data parallel size (-1 to disable)
            dp_master_address: DP master address (for non-rank-0 servers)
            dp_rpc_port: DP RPC port (for non-rank-0 servers)
            enable_pd: Enable prefill-decode disaggregation
            nixl_side_channel_base: Base port for NIXL side channel
            colocated_training: Whether the server is colocated with training workers
        """
        self._cli_args = vllm_cli_args
        self._ip = get_node_ip()
        self._port = get_open_port(start_port)
        self._server_idx = server_idx
        self._num_gpus_per_server = self.compute_num_gpus_per_server(vllm_cli_args)

        # Ensure SkyRL's custom worker extension is used for weight sync
        self._ensure_worker_extension()

        # Ensure vLLM sleep endpoints are enabled by using dev mode
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(0.2 if colocated_training else 1.0)

        # Ensure Ray executor is used (required for GPU inheritance in placement groups)
        self._ensure_ray_executor()

        # Update args with our assigned host/port
        self._cli_args.host = "0.0.0.0"
        self._cli_args.port = self._port

        # PD disaggregation: setup NIXL side channel for KV transfer
        if enable_pd:
            self._setup_nixl_side_channel(nixl_side_channel_base)

        # Each engine needs to know its dp_rank and dp_size so DP process groups are formed
        if dp_size > 0:
            self._cli_args.data_parallel_size = dp_size
            self._cli_args.data_parallel_rank = server_idx

            # DP0 will be the master sharing its ip and port with others.
            # So if we are not DP0, we need to pass master_ip and port from
            # outside. otherwise, we can use the local ip and port.
            if server_idx == 0:
                dp_master_address, dp_rpc_port = self.get_dp_info()

            if dp_master_address is None or dp_rpc_port is None:
                raise ValueError("DP address and RPC port must be set for non-server 0")

            self._cli_args.data_parallel_address = dp_master_address
            self._cli_args.data_parallel_rpc_port = dp_rpc_port
            logger.info(
                f"Server {server_idx}: DP enabled - dp_size={dp_size}, dp_rank={server_idx}, "
                f"dp_master_address={dp_master_address}, dp_rpc_port={dp_rpc_port}"
            )

        # Set bundle indices for this server's TP/PP workers in the placement group
        bundle_indices = list(range(start_bundle_idx, start_bundle_idx + self._num_gpus_per_server))
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"Server {server_idx}: using bundle indices {bundle_indices}")

        # Initialized lazily to not block the actor initialization.
        self._engine: Optional[AsyncLLMEngine] = None
        self._server_task: Optional[asyncio.Task] = None

    def _ensure_worker_extension(self) -> None:
        """
        Ensure the SkyRL worker extension is configured.

        The worker extension (WorkerWrap) provides the RPC methods needed for
        weight synchronization (init_weight_update_communicator, load_weights).
        """
        if not hasattr(self._cli_args, "worker_extension_cls") or not self._cli_args.worker_extension_cls:
            self._cli_args.worker_extension_cls = VLLM_WORKER_EXTENSION_CLS
            logger.info(f"Using default worker extension: {VLLM_WORKER_EXTENSION_CLS}")
        else:
            logger.info(f"Using provided worker extension: {self._cli_args.worker_extension_cls}")

    def _ensure_ray_executor(self) -> None:
        """
        Ensure Ray is used as the distributed executor backend.

        When running inside a Ray actor, we must use the Ray executor so that
        workers are spawned and properly inherit GPU allocation from the
        placement group.
        """
        if (
            not hasattr(self._cli_args, "distributed_executor_backend")
            or self._cli_args.distributed_executor_backend != "ray"
        ):
            self._cli_args.distributed_executor_backend = "ray"

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

        if hasattr(self._cli_args, "kv_transfer_config") and self._cli_args.kv_transfer_config:
            try:
                kv_config = json.loads(self._cli_args.kv_transfer_config)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(
                    f"Invalid kv_transfer_config: expected valid JSON string, "
                    f"got {type(self._cli_args.kv_transfer_config).__name__}: {e}"
                ) from e
            kv_config["engine_id"] = engine_id
            self._cli_args.kv_transfer_config = json.dumps(kv_config)

        logger.info(
            f"Server {self._server_idx}: NIXL side channel configured - "
            f"host={self._ip}, port={side_channel_port}, engine_id={engine_id}"
        )

    def get_server_info(self) -> ServerInfo:
        """Get the server's IP and port info."""
        return ServerInfo(ip=self._ip, port=self._port)

    def _get_extended_server_info(self) -> Dict[str, Any]:
        """Return extended server info including parallelism settings."""
        return {
            "ip": self._ip,
            "port": self._port,
            "url": f"http://{self._ip}:{self._port}",
            "server_idx": self._server_idx,
            "world_size": self._num_gpus_per_server,
        }

    def get_dp_info(self) -> Tuple[str, int]:
        """Get the DP master address and RPC port (for server 0 to share with others)."""
        dp_rpc_port = self._port + SKYRL_VLLM_DP_PORT_OFFSET
        return (self._ip, dp_rpc_port)

    async def start(self) -> ServerInfo:
        """Start the vLLM server. Blocks until server is healthy."""

        set_ulimit()
        logger.info(f"Starting server on {self._ip}:{self._port}...")

        # Start HTTP server as background asyncio task
        self._server_task = asyncio.create_task(self._run_server())

        # Wait until the server is actually healthy
        await self._wait_until_healthy()

        return self.get_server_info()

    async def _wait_until_healthy(self, timeout: float = SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S) -> None:
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

                await asyncio.sleep(1.0)

    async def _run_server(self) -> None:
        """Internal method to run the HTTP server."""
        sock_addr = (self._cli_args.host, self._cli_args.port)
        sock = create_server_socket(sock_addr)
        app = build_app(self._cli_args)

        # Initialize the engine (this loads the model - takes time)
        engine_args = AsyncEngineArgs.from_cli_args(self._cli_args)
        self._engine = AsyncLLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
        logger.info(f"Engine initialized on {self._ip}:{self._port}, adding custom endpoints...")

        # Add custom SkyRL endpoints
        self._add_custom_endpoints(app)

        await init_app_state(self._engine, app.state, self._cli_args)

        # Use uvicorn directly (serve_http tries to add signal handlers which fails in Ray actors)
        config = uvicorn.Config(
            app,
            host=self._cli_args.host,
            port=self._cli_args.port,
            log_level=self._cli_args.uvicorn_log_level,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=self._cli_args.ssl_keyfile,
            ssl_certfile=self._cli_args.ssl_certfile,
            ssl_ca_certs=self._cli_args.ssl_ca_certs,
            ssl_cert_reqs=self._cli_args.ssl_cert_reqs,
            access_log=not getattr(self._cli_args, "disable_uvicorn_access_log", False),
        )
        server = uvicorn.Server(config)
        await server.serve(sockets=[sock])

    def _add_custom_endpoints(self, app) -> None:
        """Add custom SkyRL endpoints to the FastAPI app."""
        engine = self._engine

        @app.get("/get_server_info")
        async def _get_server_info():
            """Return server parallelism info."""
            return self._get_extended_server_info()

        # TODO (Kourosh): After https://github.com/vllm-project/vllm/pull/
        # 31943/ is merged, use the native API.
        @app.post("/init_weight_transfer")
        async def _init_weight_transfer(request: Request):
            """Initialize weight sync process group."""
            from skyrl_train.weight_sync import BroadcastInitInfo, CudaIpcInitInfo

            data = await request.json()
            # simple way to figure out the strategy type: try to load with BroadcastInitInfo else fallback to CudaIpcInitInfo
            # Can be derived from vllm cli args once https://github.com/vllm-project/vllm/pull/31943/ is merged.
            try:
                init_info = BroadcastInitInfo(**data)
            except Exception:
                try:
                    init_info = CudaIpcInitInfo(**data)
                except Exception:
                    raise HTTPException(status_code=400, detail="Received invalid init info")

            init_info = init_info.for_engine(
                engine_index=self._server_idx,
                tp_size=self._cli_args.tensor_parallel_size,
                pp_size=self._cli_args.pipeline_parallel_size,
            )
            pickled_init_info = pickle.dumps(init_info)

            await engine.collective_rpc(
                "init_weight_update_communicator",
                args=(pickled_init_info,),
            )
            return {"status": "ok"}

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            """Update model weights via NCCL broadcast."""
            from skyrl_train.weight_sync import BroadcastWeightUpdateRequest, CudaIpcWeightUpdateRequest

            data = await request.json()
            try:
                weight_request = BroadcastWeightUpdateRequest.from_json_dict(data)
            except Exception:
                try:
                    weight_request = CudaIpcWeightUpdateRequest.from_json_dict(data)
                except Exception:
                    raise HTTPException(status_code=400, detail="Received invalid weight update request")

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

            Currently a no-op, reserved for future use e.g. Quantization
            See https://github.com/vllm-project/vllm/issues/31848 for more
            details.
            """
            # No-op for now - placeholder for future post-processing
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            """Reset the prefix cache."""
            await engine.reset_prefix_cache()
            return {"status": "ok"}

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
