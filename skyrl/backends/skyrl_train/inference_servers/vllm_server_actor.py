"""
vLLM Server Actor - Ray actor running a vLLM OpenAI-compatible API server.
"""

import asyncio
import logging
import os
import time
from argparse import Namespace
from typing import List, Optional, Tuple

import httpx
import uvicorn
import vllm.envs as envs
from fastapi import HTTPException, Request
from ray.util.placement_group import PlacementGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams as VLLMSamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm.utils.system_utils import set_ulimit

from skyrl.backends.skyrl_train.inference_servers.common import (
    ServerInfo,
    find_and_reserve_port,
    get_node_ip,
)
from skyrl.backends.skyrl_train.inference_servers.protocols import ServerActorProtocol
from skyrl.env_vars import (
    SKYRL_HTTP_CONNECTION_LIMIT,
    SKYRL_VLLM_DP_PORT_OFFSET,
    SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


class VLLMServerActor(ServerActorProtocol):
    """
    Ray actor that runs a vLLM OpenAI-compatible API server.

    Implements ServerActorProtocol for use with ServerGroup.

    The server runs in the actor and exposes an HTTP endpoint that can be
    called from anywhere (other actors, driver, external processes).

    Custom endpoints added for SkyRL:
    - /reset_prefix_cache: Reset prefix cache

    Weight sync uses vLLM native endpoints (/init_weight_transfer_engine,
    /update_weights, /get_world_size) from the RLHF router when VLLM_SERVER_DEV_MODE=1.
    """

    @staticmethod
    def compute_num_gpus_per_server(vllm_cli_args: Namespace) -> int:
        """Compute the number of GPUs needed per server based on TP * PP.

        This logic might need adjustment if we want to support other
        parallelism schemes. If we get to this point, we should add a
        vllm-specific utility for it and keep the logic inside the engine.
        """
        return vllm_cli_args.tensor_parallel_size * vllm_cli_args.pipeline_parallel_size

    @staticmethod
    def prepare_server_kwargs(
        pg: PlacementGroup,
        start_bundle_idx: int,
        num_gpus_per_server: int,
        **kwargs,
    ) -> dict:
        # _gpu_ids is passed by ServerGroup from the cached ResolvedPlacementGroup.bundle_gpu_ids.
        gpu_ids = kwargs.pop("_gpu_ids", None)
        if kwargs.get("distributed_executor_backend") == "mp" and gpu_ids is not None:
            kwargs["mp_cuda_visible_devices"] = ",".join(str(g) for g in gpu_ids)
        return kwargs

    def __init__(
        self,
        vllm_cli_args: Namespace,
        start_port: int = 8000,
        server_idx: int = 0,
        bundle_indices: Optional[List[int]] = None,
        dp_size: int = -1,
        dp_master_address: Optional[str] = None,
        dp_rpc_port: Optional[int] = None,
        # PD disaggregation settings
        enable_pd: bool = False,
        nixl_side_channel_base: int = 5600,
        colocated_training: bool = False,
        distributed_executor_backend: str = "ray",
        mp_cuda_visible_devices: Optional[str] = None,
    ):
        """
        Initialize the vLLM server actor.

        Args:
            vllm_cli_args: vLLM CLI arguments.
                Required attributes: tensor_parallel_size, pipeline_parallel_size.
                Optional: uvicorn_log_level, ssl_*, disable_uvicorn_access_log, kv_transfer_config.
            start_port: Base port to start searching for free port
            server_idx: Index of this server in the group
            bundle_indices: Bundle indices in the placement group for this server's workers.
                If None, defaults to [0, 1, ..., num_gpus_per_server - 1].
            dp_size: Data parallel size (-1 to disable)
            dp_master_address: DP master address (for non-rank-0 servers)
            dp_rpc_port: DP RPC port (for non-rank-0 servers)
            enable_pd: Enable prefill-decode disaggregation
            nixl_side_channel_base: Base port for NIXL side channel to start searching for a free port
            colocated_training: Whether the server is colocated with training workers
            distributed_executor_backend: vLLM distributed executor backend.
                ``"ray"`` spawns TP/PP workers as Ray tasks (default).
                ``"mp"`` spawns workers as local processes using
                CUDA_VISIBLE_DEVICES.
            mp_cuda_visible_devices: Comma-separated physical GPU IDs for the
                ``"mp"`` backend. Pre-computed by ServerGroup from the
                per-server placement group. Only used when
                ``distributed_executor_backend="mp"`` and TP*PP > 1.
        """
        from skyrl.train.utils.ray_logging import redirect_actor_output_to_file

        redirect_actor_output_to_file()

        self._cli_args = vllm_cli_args
        self._ip = get_node_ip()
        self._port, self._port_reservation = find_and_reserve_port(start_port)
        self._server_idx = server_idx
        self._num_gpus_per_server = self.compute_num_gpus_per_server(vllm_cli_args)
        self._use_mp_backend = distributed_executor_backend == "mp"

        # Ensure vLLM sleep endpoints are enabled by using dev mode
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        # Enable runtime LoRA loading/unloading via /v1/load_lora_adapter endpoint
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        # TODO (aaron): once native ipc stops needing this, remove
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # Configure the distributed executor backend
        self._cli_args.distributed_executor_backend = distributed_executor_backend

        # Update args with our assigned host/port
        self._cli_args.host = "0.0.0.0"
        self._cli_args.port = self._port

        # PD disaggregation: setup NIXL side channel for KV transfer
        self._nixl_port_reservation = None
        self._nixl_side_channel_base = None
        if enable_pd:
            # use nixl_side_channel_base + server_idx as convention for the start port for this server
            self._nixl_side_channel_base, self._nixl_port_reservation = find_and_reserve_port(
                nixl_side_channel_base + server_idx
            )
            self._setup_nixl_side_channel(self._nixl_side_channel_base)

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

        # Configure GPU visibility for this server's TP/PP workers
        if self._use_mp_backend:
            self._setup_mp_gpu_visibility(mp_cuda_visible_devices)
        else:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(0.2 if colocated_training else 1.0)
            # Set bundle indices for this server's TP/PP workers in the placement group.
            # NOTE: This assumes single-GPU-per-bundle placement groups.
            if bundle_indices is None:
                bundle_indices = list(range(self._num_gpus_per_server))
            assert (
                len(bundle_indices) == self._num_gpus_per_server
            ), f"Expected {self._num_gpus_per_server} bundle indices (one per GPU), got {len(bundle_indices)}"
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            logger.info(f"Server {server_idx}: using bundle indices {bundle_indices}")

        # Initialized lazily to not block the actor initialization.
        self._engine: Optional[AsyncLLMEngine] = None
        self._server_task: Optional[asyncio.Task] = None

    def _setup_mp_gpu_visibility(self, mp_cuda_visible_devices: Optional[str]) -> None:
        """Set CUDA_VISIBLE_DEVICES for the mp backend.

        When using the mp backend, vLLM spawns workers as local processes.
        They discover GPUs via CUDA_VISIBLE_DEVICES rather than inheriting
        from a placement group.
        """
        if mp_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = mp_cuda_visible_devices
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            logger.info(f"Server {self._server_idx}: mp backend, " f"CUDA_VISIBLE_DEVICES={mp_cuda_visible_devices}")
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("ROCR_VISIBLE_DEVICES", None)
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
            logger.info(
                f"Server {self._server_idx}: mp backend, " f"cleared CUDA_VISIBLE_DEVICES (single-GPU or auto-detect)"
            )

    def _setup_nixl_side_channel(self, side_channel_port: int) -> None:
        """
        Setup NIXL side channel for PD disaggregation.

        Each server instance needs a unique side channel port for KV transfer handshake.
        """
        import json

        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(side_channel_port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = self._ip

        engine_id = f"server-{self._server_idx}-{self._ip}-{side_channel_port}"

        if hasattr(self._cli_args, "kv_transfer_config") and self._cli_args.kv_transfer_config:
            kv_config = self._cli_args.kv_transfer_config
            # Handle both dict and JSON string formats
            if isinstance(kv_config, str):
                try:
                    kv_config = json.loads(kv_config)
                except (json.JSONDecodeError, TypeError) as e:
                    raise ValueError(
                        f"Invalid kv_transfer_config: expected valid JSON string or dict, "
                        f"got {type(self._cli_args.kv_transfer_config).__name__}: {e}"
                    ) from e
            kv_config["engine_id"] = engine_id
            self._cli_args.kv_transfer_config = kv_config

        logger.info(
            f"Server {self._server_idx}: NIXL side channel configured - "
            f"host={self._ip}, port={side_channel_port}, engine_id={engine_id}"
        )

    def get_server_info(self) -> ServerInfo:
        """Get the server's IP and port info."""
        return ServerInfo(ip=self._ip, port=self._port)

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
        # Release the port reservation right before vLLM rebinds.
        if self._port_reservation is not None:
            self._port_reservation.close()
            self._port_reservation = None

        if self._nixl_port_reservation is not None:
            self._nixl_port_reservation.close()
            self._nixl_port_reservation = None

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
            backlog=SKYRL_HTTP_CONNECTION_LIMIT,
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
        cli_args = self._cli_args

        # Weight sync uses vLLM native endpoints (/init_weight_transfer_engine,
        # /update_weights, /get_world_size) registered by the RLHF router when
        # VLLM_SERVER_DEV_MODE=1.

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            """Reset the prefix cache."""
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        # NOTE (sumanthrh): We use a custom generate endpoint /skyrl/v1/generate because the native
        # endpoint /inference/v1/generate does not support returning routed expert IDs.
        # TODO (sumanthrh): Migrate back to /inference/v1/generate once this is fixed on the vllm side
        @app.post("/skyrl/v1/generate")
        async def _skyrl_generate(request: Request):
            """SkyRL generate endpoint that returns routed_experts alongside token output."""
            if getattr(cli_args, "enable_lora", False):
                raise HTTPException(status_code=400, detail="/skyrl/v1/generate does not support LoRA.")

            body = await request.json()
            token_ids = body["token_ids"]
            sampling_params_dict = body.get("sampling_params", {})

            sampling_params = VLLMSamplingParams(**sampling_params_dict)
            prompt = TokensPrompt(prompt_token_ids=token_ids)
            request_id = random_uuid()

            final_res = None
            async for res in engine.generate(prompt, sampling_params, request_id=request_id):
                final_res = res

            if final_res is None:
                raise HTTPException(status_code=500, detail="vLLM returned no output")
            resp = final_res.outputs[0]

            token_ids_out = list(resp.token_ids)
            finish_reason = resp.finish_reason

            logprobs = None
            if resp.logprobs is not None:
                content = []
                for tid, lp_dict in zip(token_ids_out, resp.logprobs):
                    if lp_dict and tid in lp_dict:
                        content.append({"logprob": lp_dict[tid].logprob})
                    else:
                        # -9999.0 is the default in vLLM's ChatCompletionLogProb
                        content.append({"logprob": -9999.0})
                logprobs = {"content": content}

            routed_experts = None
            if resp.routed_experts is not None:
                if hasattr(resp.routed_experts, "tolist"):
                    routed_experts = resp.routed_experts.tolist()
                else:
                    routed_experts = resp.routed_experts

            return {
                "choices": [
                    {
                        "token_ids": token_ids_out,
                        "finish_reason": finish_reason,
                        "logprobs": logprobs,
                        "routed_experts": routed_experts,
                    }
                ]
            }

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
