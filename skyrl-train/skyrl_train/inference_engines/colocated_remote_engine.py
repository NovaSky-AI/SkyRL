import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy, PlacementGroup, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
)
from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine
from skyrl_train.distributed.utils import get_free_port

from typing import List, Optional
import asyncio
import os
import socket
import subprocess
import time
import urllib.request
import signal
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig
from loguru import logger


def _server_ready(host: str, port: int, endpoint: str, timeout_seconds: int = 180) -> bool:
    """Ensures that the server is ready by ensure TCP connection
    AND HTTP readiness

    Args:
        host (str)
        port (int)
        endpoint (str): application endpoint for HTTP readiness
        timeout_seconds (int): return False if not ready within timeout_seconds

    Returns:
        bool: True if server is ready
    """

    # first ensure TCP socket connection
    start_time = time.time()
    while True:
        try:
            with socket.socket() as s:
                s.settimeout(1)
                s.connect((host, port))
                break
        except (socket.timeout, ConnectionRefusedError):
            pass
        if time.time() - start_time >= timeout_seconds:
            return False
        time.sleep(0.5)

    # then ensure HTTP readiness
    start_time = time.time()
    while True:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/{endpoint}", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception as e:
            logger.warning(f"Server readiness failed: {e}")
        if time.time() - start_time >= timeout_seconds:
            return False
        time.sleep(0.5)


@ray.remote
class VLLMHTTPServerActor:
    """Ray actor that spawns a vLLM HTTP server as a subprocess."""

    def __init__(self):
        self.pid: Optional[int] = None
        self.host = ray.util.get_node_ip_address().strip("[]")
        self.port: Optional[int] = None

    def start(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        enforce_eager: bool,
        enable_prefix_caching: bool,
        dtype: str,
        distributed_executor_backend: str,
        quiet: bool,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> str:
        self.port = port or get_free_port()
        if host:
            self.host = host

        # Build vLLM server CLI
        cmd = [
            "python",
            "-m",
            "skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            model_path,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--dtype",
            dtype,
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--worker-extension-cls",
            "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
            "--enable-sleep-mode",
            "--distributed-executor-backend",
            distributed_executor_backend,
        ]

        if enforce_eager:
            cmd.append("--enforce-eager")
        if enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        # Set CUDA_VISIBLE_DEVICES based on Ray's GPU allocation for this actor
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Launch server
        stdout = subprocess.DEVNULL if quiet else None
        stderr = subprocess.DEVNULL if quiet else None
        # Put subprocess in its own process group so we can kill the entire vLLM process tree
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, preexec_fn=os.setsid)
        self.pid = proc.pid

        healthy = _server_ready(self.host, self.port, "v1/models")
        if not healthy:
            raise RuntimeError(f"vLLM server not ready {self.host}:{self.port}")

        return f"{self.host}:{self.port}"

    def kill(self):
        # Kill only the vLLM subprocess group (created via setsid in start())
        if self.pid is None:
            # kill might be called before start finishes
            return
        try:
            pgid = os.getpgid(self.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            # process group might already be dead
            pass

    def __ray_shutdown__(self):
        self.kill()


class ColocatedRemoteEngine(RemoteInferenceEngine):
    """RemoteInferenceEngine that retains a reference to its Ray server actor."""

    def __init__(
        self,
        *,
        server_actor,  # ActorHandle
        url: str,
        model_name: str,
        tokenizer: PreTrainedTokenizerBase,
        engine_backend: str,
        tp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
    ):
        super().__init__(
            url=url,
            model_name=model_name,
            engine_backend=engine_backend,
            tokenizer=tokenizer,
            tp_size=tp_size,
            dp_size=dp_size,
            ep_size=ep_size,
        )
        self.server_actor = server_actor

    async def teardown(self):
        await super().teardown()
        try:
            await self.server_actor.kill.remote()
        except Exception:
            pass


def create_colocated_remote_engines(
    cfg: DictConfig, tokenizer: PreTrainedTokenizerBase, shared_pg: PlacementGroup
) -> List[InferenceEngineInterface]:
    """
    Launches remote inference servers in ray actors
    """
    assert cfg.trainer.placement.colocate_all, "colocate_all flag must be true for creating colocated remote engines"
    assert shared_pg is not None, "Must have valid placement group for colocated training"

    num_inference_engines = cfg.generator.num_inference_engines
    tensor_parallel_size = cfg.generator.inference_engine_tensor_parallel_size
    data_parallel_size = cfg.generator.inference_engine_data_parallel_size

    use_shared_pg = tensor_parallel_size == 1
    if not use_shared_pg:
        # Create a big placement group to ensure that all inference engines are packed
        total_servers = num_inference_engines * data_parallel_size
        bundles = [{"GPU": tensor_parallel_size, "CPU": 1} for _ in range(total_servers)]
        pg = placement_group(bundles, strategy="PACK")
        from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout
        from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
    else:
        pg = shared_pg

    # Launch one server per DP replica for each engine index
    server_actors = []
    bundle_index = 0
    for _ in range(num_inference_engines):
        for dp_rank in range(data_parallel_size):
            sched = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )
            # If using the shared PG (TP == 1), reserve fractional GPU/CPU to colocate with trainers.
            # Otherwise, we created bundles sized to TP.
            num_gpus_for_actor = 0.2 if use_shared_pg else tensor_parallel_size
            num_cpus_for_actor = 0.2 if use_shared_pg else 1
            actor_class = VLLMHTTPServerActor
            actor = actor_class.options(
                num_cpus=num_cpus_for_actor,
                num_gpus=num_gpus_for_actor,
                scheduling_strategy=sched,
            ).remote()
            server_actors.append(actor)
            bundle_index += 1

    # start servers
    start_refs = []
    for actor in server_actors:
        start_refs.append(
            actor.start.remote(
                model_path=cfg.trainer.policy.model.path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
                enforce_eager=cfg.generator.enforce_eager,
                enable_prefix_caching=cfg.generator.enable_prefix_caching,
                dtype=cfg.generator.model_dtype,
                distributed_executor_backend="mp" if tensor_parallel_size > 1 else "uni",
                quiet=False,
            )
        )
    urls: list[str] = ray.get(start_refs)

    # create inference engines
    engines: List[InferenceEngineInterface] = []
    for url, actor in zip(urls, server_actors):
        engines.append(
            ColocatedRemoteEngine(
                server_actor=actor,
                url=url,
                model_name=cfg.trainer.policy.model.path,
                tokenizer=tokenizer,
                engine_backend=cfg.generator.backend,
                tp_size=tensor_parallel_size,
                dp_size=data_parallel_size,
                ep_size=cfg.generator.inference_engine_expert_parallel_size,
            )
        )

    # Put servers to sleep
    sleep_level = 1 if getattr(cfg.trainer.policy.model.lora, "rank", 0) > 0 else 2

    async def _sleep_all():
        await asyncio.gather(*[engine.sleep(level=sleep_level) for engine in engines])

    asyncio.run(_sleep_all())

    return engines
