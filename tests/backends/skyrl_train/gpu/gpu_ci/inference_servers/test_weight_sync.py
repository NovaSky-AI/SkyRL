"""
GPU CI tests for weight synchronization from trainer to inference server.

1. Non-colocated (NCCL broadcast), TP=2:
    - Trainer on GPUs 0-1, server (TP=2) on GPUs 2-3 (4 GPUs total)
    - Uses NCCL broadcast for weight sync via HTTP router

2. Colocated (CUDA IPC), TP=1:
    - Trainer and server share GPU 0 (2 GPUs total, 1 shared)
    - Uses CUDA IPC handles for zero-copy weight transfer

3. Legacy `WorkerWrap.load_weights` MoE reload, TP=1:
    - Server on GPU 0 (1 GPU total, no separate trainer process)
    - The NCCL or CUDA-IPC receiver is stubbed with safetensors-from-disk
      to skip trainer-side sender setup

Run:
    uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v -s
"""

import asyncio
import base64
import os
import pickle

import httpx
import pytest
import pytest_asyncio
import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.inference_servers.common import (
    get_node_ip,
    get_open_port,
)
from skyrl.backends.skyrl_train.weight_sync import (
    BroadcastInitInfo,
    CudaIpcInitInfo,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = os.environ.get("SKYRL_RDT_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


@ray.remote
class Trainer:
    """
    Simple trainer emulator that holds the real model weights.

    This is a simplified version of the trainer side for testing weight sync
    via NCCL broadcast in non-colocated scenarios.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.pg = None
        self.model_name = model_name

    def ready(self):
        """Check if the trainer is ready."""
        return True

    def init_weight_sync(self, master_address: str, master_port: int, world_size: int, group_name: str):
        """Initialize the weight sync process group as rank 0 (trainer)."""
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        self.pg = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            )
        )
        return True

    def get_weight_info(self) -> dict:
        """
        Get weight metadata (names, dtypes, shapes) without doing NCCL.

        Returns:
            dict with names, dtypes, shapes for the weight update request.
        """
        names = []
        dtypes = []
        shapes = []

        for name, param in self.model.named_parameters():
            names.append(name)
            dtypes.append(str(param.dtype).split(".")[-1])  # e.g. "bfloat16"
            shapes.append(list(param.shape))

        return {"names": names, "dtypes": dtypes, "shapes": shapes}

    def broadcast_weights(self):
        """
        Broadcast all model weights to inference workers via NCCL.

        This is a blocking operation - server must call receive concurrently.
        """
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        params = list(self.model.named_parameters())
        print(
            f"[Trainer.broadcast_weights] Starting send of {len(params)} params, pg={self.pg}, pg.rank={self.pg.rank}, pg.world_size={self.pg.world_size}"
        )
        try:
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(params),
                trainer_args={"group": self.pg, "packed": True},
            )
            torch.cuda.synchronize()
            print("[Trainer.broadcast_weights] Send complete")
        except Exception as e:
            print(f"[Trainer.broadcast_weights] ERROR: {e}")
            raise


@pytest_asyncio.fixture(
    scope="class",
    params=[
        pytest.param({"enable_pd": False}, id="no_pd"),
        pytest.param(
            {"enable_pd": True, "num_prefill": 1, "num_decode": 1},
            id="pd_1P1D_non_colocated",
        ),
    ],
)
async def weight_update_env(class_scoped_ray_init_fixture, request):
    """
    Create environment for weight update testing (non-colocated, NCCL broadcast).

    - no_pd: TP=2 server on its own GPUs, trainer on separate GPU(s) (4 GPUs).
    - pd_1P1D_non_colocated: 1P1D (2 engines, TP=1), trainer on separate GPU (3 GPUs).
      Exercises non-colocated PD path in create_inference_servers with separate
      prefill/decode placement groups.
    """
    pd_cfg = request.param
    enable_pd = pd_cfg["enable_pd"]
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL

    if enable_pd:
        num_prefill = pd_cfg["num_prefill"]
        num_decode = pd_cfg["num_decode"]
        create_kwargs = dict(
            model=MODEL,
            tp_size=1,
            num_inference_engines=num_prefill + num_decode,
            colocate_all=False,
            gpu_memory_utilization=0.5,
            engine_init_kwargs={
                "load_format": "dummy",
                "kv_transfer_config": {
                    "kv_connector": "NixlConnector",
                },
            },
            enable_pd=True,
            num_prefill=num_prefill,
        )
    else:
        create_kwargs = dict(
            model=MODEL,
            tp_size=2,
            colocate_all=False,
            gpu_memory_utilization=0.5,
            engine_init_kwargs={"load_format": "dummy"},
        )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        trainer = Trainer.options(num_gpus=1.0).remote(MODEL)
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": engines.client,
            "router_url": engines.client.proxy_url,
        }

        await engines.client.teardown()
        ray.kill(trainer)
    # cleanup manually in colocated case
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestWeightUpdateFlow:
    """Tests for weight synchronization from trainer to inference server (non-colocated)."""

    async def test_update_weights_flow(self, weight_update_env):
        """
        Full E2E weight sync test (non-colocated, NCCL broadcast):
        1. Query with dummy weights → gibberish
        2. Init weight transfer (both sides concurrently via client)
        3. Broadcast weights from trainer (concurrent with server receive)
        4. Finalize weight update
        5. Query again → correct output
        """
        router_url = weight_update_env["router_url"]
        trainer = weight_update_env["trainer"]
        client = weight_update_env["client"]

        print("\n[TEST] Running non-colocated weight sync test")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
            # ===== Step 1: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")

            # Dummy weights should NOT produce coherent output about Paris
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: Init weight transfer (both sides concurrently) =====
            master_address = get_node_ip()
            master_port = get_open_port()

            # Query all servers for world_size via client (fans out to all backends)
            inference_world_size, _ = await client.get_world_size()
            world_size = 1 + inference_world_size  # 1 trainer + all inference workers
            group_name = f"weight_sync_test_{master_port}"

            print(f"[Step 2] Init weight transfer: master={master_address}:{master_port}, world_size={world_size}")

            init_info = BroadcastInitInfo(
                master_addr=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
                override_existing_receiver=True,
            )

            # Both sides must init concurrently (NCCL blocks until all ranks join)
            # Start trainer init (returns immediately, runs in Ray actor)
            trainer_init_ref = trainer.init_weight_sync.remote(master_address, master_port, world_size, group_name)

            # Await server init via client (fans out to all backends)
            result = await client.init_weight_update_communicator(init_info)
            for server_url, resp in result.items():
                assert resp["status"] == 200, f"Server {server_url} init failed: {resp}"

            # Trainer should be done now (NCCL group formed)
            ray.get(trainer_init_ref)
            print("[Step 2] Both sides init complete")

            # ===== Step 3: Broadcast weights (concurrent send/receive) =====
            print("[Step 3] Broadcasting weights from trainer to server...")

            # Get weight metadata first (no NCCL yet)
            weight_info = ray.get(trainer.get_weight_info.remote())
            print(f"[Step 3] Weight info: {len(weight_info['names'])} parameters")

            # Start trainer broadcast (returns immediately, runs in Ray actor)
            print("[Step 3] Launching trainer broadcast_weights.remote()...")
            trainer_broadcast_ref = trainer.broadcast_weights.remote()

            # Await server receive via client (fans out to all backends)
            dtype_names = [(d.split(".")[-1] if "." in d else d) for d in weight_info["dtypes"]]
            update_info = {
                "names": weight_info["names"],
                "dtype_names": dtype_names,
                "shapes": weight_info["shapes"],
                "packed": True,
            }
            print(
                f"[Step 3] Calling update_weights_nccl with {len(update_info['names'])} names, packed={update_info['packed']}"
            )
            # Use SkyRL's chunked weight-sync API (skyrl_start_weight_update ->
            # update_weights_nccl -> skyrl_finish_weight_update) rather than vLLM's
            # native /update_weights endpoint, which in vLLM 0.22.0+ requires
            # vLLM's own native start_weight_update to be called first.
            # skyrl_start_weight_update is local (layerwise-reload init), so it is
            # safe to call while the trainer is blocked on the NCCL send; the
            # actual receive happens in update_weights_nccl.
            await client.start_weight_update()
            result = await client.update_weights_nccl(update_info)
            print(f"[Step 3] update_weights_nccl returned: {list(result.keys())}")
            for server_url, resp in result.items():
                assert resp["status"] == 200, f"Server {server_url} update weights failed: {resp}"
            await client.finish_weight_update()

            # Trainer should be done now (NCCL broadcast complete)
            ray.get(trainer_broadcast_ref)
            print("[Step 3] Weight sync complete")

            # ===== Step 4: Query again - should produce correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 5] Real weights output: {text_after!r}")

            assert "Paris" in text_after, f"Weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] Non-colocated weight sync test passed!")


# -----------------------------------------------------------------
# Colocated CUDA IPC Weight Sync Test
# -----------------------------------------------------------------


@ray.remote
class IpcTrainer:
    """
    Trainer emulator that creates CUDA IPC handles for weight transfer.

    Unlike the NCCL Trainer, this does not create a process group.
    Instead it creates per-tensor IPC handles that the colocated
    inference server opens to read weights directly from GPU memory.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self._tensor_refs: list = []

    def ready(self):
        return True

    def create_ipc_update_info(self) -> dict:
        """Create a single packed CUDA-IPC buffer for all model parameters.

        Matches SkyRL's ``update_weights_ipc`` contract (the packed format
        produced by ``CudaIpcTransferStrategy``): all parameters are copied into
        one contiguous CUDA buffer, a single IPC handle is created for that
        buffer, and per-parameter ``sizes`` let the receiver slice it back out.
        This differs from vLLM's native ``/update_weights`` (one handle per
        parameter), which we no longer use.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        params = list(self.model.named_parameters())
        # The model is loaded in a single dtype (bfloat16), so element offsets
        # into one packed buffer are well-defined across all parameters.
        dtype = params[0][1].dtype
        total_numel = sum(p.numel() for _, p in params)
        packed_tensor = torch.empty(total_numel, device=self.device, dtype=dtype)

        names, dtype_names, shapes, sizes = [], [], [], []
        offset = 0
        for name, param in params:
            size = param.numel()
            packed_tensor[offset : offset + size].copy_(param.detach().reshape(-1))
            offset += size
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))
            sizes.append(size)

        # Keep the packed buffer alive so the IPC handle stays valid on the receiver.
        self._tensor_refs = [packed_tensor]

        ipc_handle = reduce_tensor(packed_tensor)
        pickled = base64.b64encode(pickle.dumps({gpu_uuid: ipc_handle})).decode("utf-8")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "sizes": sizes,
            "ipc_handles_pickled": pickled,
        }


@pytest_asyncio.fixture(scope="class")
async def ipc_weight_update_env(class_scoped_ray_init_fixture):
    """Create environment for colocated IPC weight update testing."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    create_kwargs = dict(
        model=MODEL,
        tp_size=1,
        colocate_all=True,
        gpu_memory_utilization=0.5,
        engine_init_kwargs={"load_format": "dummy"},
    )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        # Trainer on same PG bundle as server (colocated) with fractional GPU
        trainer = IpcTrainer.options(
            num_gpus=0.2,
            num_cpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=engines.pg,
                placement_group_bundle_index=0,
            ),
        ).remote(MODEL)
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": engines.client,
            "router_url": engines.client.proxy_url,
        }

        await engines.client.teardown()
        ray.kill(trainer)
    # cleanup manually in colocated case
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestColocatedIpcWeightUpdateFlow:
    """Tests for weight synchronization via CUDA IPC (colocated, TP=1)."""

    async def test_update_weights_ipc(self, ipc_weight_update_env):
        """
        Full E2E weight sync test (colocated, CUDA IPC):
        1. Query with dummy weights → gibberish
        2. Init IPC weight transfer engine (no-op for IPC)
        3. Create IPC handles from trainer weights and send to server
        4. Query again → correct output
        """
        router_url = ipc_weight_update_env["router_url"]
        trainer = ipc_weight_update_env["trainer"]
        client = ipc_weight_update_env["client"]

        print("\n[TEST] Running colocated IPC weight sync test")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as http_client:
            # ===== Step 1: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: Init IPC engine (no-op but verifies endpoint) =====
            init_info = CudaIpcInitInfo(
                model_dtype_str="bfloat16",
                override_existing_receiver=True,
            )
            result = await client.init_weight_update_communicator(init_info)
            for server_url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {server_url} IPC init failed: {resp_data}"
            print("[Step 2] IPC engine init complete (no-op)")

            # ===== Step 3: Create IPC handles and send to server =====
            print("[Step 3] Creating IPC handles from trainer weights...")
            update_info = ray.get(trainer.create_ipc_update_info.remote())
            print(f"[Step 3] Created handles for {len(update_info['names'])} parameters")

            # Use SkyRL's chunked weight-sync API (skyrl_start_weight_update ->
            # update_weights_ipc -> skyrl_finish_weight_update) rather than vLLM's
            # native /update_weights endpoint.
            await client.start_weight_update()
            result = await client.update_weights_ipc(update_info)
            for server_url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {server_url} IPC update failed: {resp_data}"
            await client.finish_weight_update()
            print("[Step 3] IPC weight update complete")

            # ===== Step 4: Query again — should produce correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 4] Real weights output: {text_after!r}")
            assert "Paris" in text_after, f"IPC weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] Colocated IPC weight sync test passed!")


# -----------------------------------------------------------------
# Sharded RDT (NIXL pull) Weight Sync Test
# -----------------------------------------------------------------

# The trainer side is the VENDORED vLLM RDT sidecar engine
# (ShardedRDTTrainerWeightTransferEngine): trainer_init spawns a per-rank
# _RDTProducerServer actor (the NIXL serve surface) and shares gathered weights
# into it over CUDA IPC; send_weights drives the concurrent start/update/finish
# handshake. This test drives that engine from a single-GPU (no-FSDP) Ray actor
# via the SkyRL adapter (_SyncInferenceClient + _FsdpWeightSource) — the same
# code path FSDPPolicyWorkerBase uses in production, minus FSDP sharding.
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (  # noqa: E402
    RemoteInferenceClient,
)


class _ShimExtractor:
    """Minimal weight-extractor for a single-GPU (no-FSDP) trainer: params are
    already whole, so gather is identity and there is no name prefix. Matches the
    surface ``_FsdpWeightSource`` reads (model / weight_prefix / _gather_tensor /
    get_weight_metadata)."""

    weight_prefix = ""

    def __init__(self, model):
        self.model = model

    def _gather_tensor(self, param):
        return param

    def get_weight_metadata(self, dtype):
        names, dtype_names, shapes = [], [], []
        dtype_name = str(dtype).split(".")[-1]
        for name, param in self.model.state_dict().items():
            names.append(name)
            dtype_names.append(dtype_name)
            shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}


@ray.remote(num_gpus=1, max_concurrency=4)
class RdtTrainerActor:
    """Single-GPU (no-FSDP) trainer that drives the vendored sidecar RDT engine.

    Holds the model on its own GPU and owns a background asyncio loop so the
    engine's synchronous control-plane calls (via ``_SyncInferenceClient``) can
    drive SkyRL's async ``RemoteInferenceClient`` off the loop. ``num_gpus=1``
    (a real GPU assignment) is what lets the trainer engine pin its spawned
    ``_RDTProducerServer`` to this GPU for CUDA IPC.
    """

    def __init__(self, model_name, proxy_url, server_urls, data_parallel_size, model_id, namespace, num_consumers):
        import threading

        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to("cuda")
        self._extractor = _ShimExtractor(self._model)
        self._namespace = namespace
        self._num_consumers = num_consumers
        # Background event loop for the sync client (the engine calls it from a
        # worker thread, never from this loop, so run_coroutine_threadsafe is safe).
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._client = RemoteInferenceClient(
            proxy_url=proxy_url,
            server_urls=server_urls,
            data_parallel_size=data_parallel_size,
            model_name=model_id,
        )
        self._engine = None

    def ready(self):
        return True

    def sync_once(self):
        """Rendezvous (first call: spawn server + bake the inference side) and run
        one full weight sync through the vendored engine."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_strategy import (
            _FsdpWeightSource,
            _SyncInferenceClient,
        )
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerInitInfo,
            ShardedRDTTrainerWeightTransferEngine,
        )

        sync_client = _SyncInferenceClient(self._client, self._loop)
        source = _FsdpWeightSource(self._extractor, torch.bfloat16)
        if self._engine is None:
            init_info = ShardedRDTTrainerInitInfo(
                rank=0,
                num_consumers=self._num_consumers,
                trainer_actor_namespace=self._namespace,
            )
            self._engine = ShardedRDTTrainerWeightTransferEngine.trainer_init(
                init_info,
                client=sync_client,
                source=source,
            )
        self._engine.send_weights()

    def shutdown(self):
        if self._engine is not None:
            self._engine.shutdown()
            self._engine = None


@pytest_asyncio.fixture(scope="class")
async def rdt_weight_update_env(class_scoped_ray_init_fixture):
    """Non-colocated sharded_rdt (NIXL pull) environment, TP=1.

    The trainer actor (1 GPU) drives the vendored sidecar engine, which spawns
    its own producer server; the vLLM server (TP=1,
    distributed_executor_backend=ray) runs on another GPU. 2 GPUs + the sidecar
    (shares the trainer's GPU).
    """
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    # Select the sharded_rdt weight-sync backend (build_vllm_cli_args reads this
    # and sets WeightTransferConfig(backend="sharded_rdt") + executor=ray).
    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"

    create_kwargs = dict(
        model=MODEL,
        tp_size=1,
        colocate_all=False,
        gpu_memory_utilization=0.5,
        engine_init_kwargs={"load_format": "dummy"},
    )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        client = engines.client
        namespace = ray.get_runtime_context().namespace or None
        trainer = RdtTrainerActor.remote(
            MODEL,
            client.proxy_url,
            client.server_urls,
            client.data_parallel_size,
            client.model_name,
            namespace,
            1,  # num_consumers (TP=1, single engine)
        )
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": client,
            "router_url": client.proxy_url,
        }

        ray.get(trainer.shutdown.remote())
        await engines.client.teardown()
        ray.kill(trainer)
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestShardedRdtWeightUpdateFlow:
    """Weight sync via the sharded_rdt (NIXL pull) backend (non-colocated, TP=1)."""

    async def test_update_weights_rdt(self, rdt_weight_update_env):
        """
        Full E2E weight sync test (non-colocated, sharded RDT / NIXL pull) via
        the VENDORED sidecar trainer engine:
        1. Query with dummy weights -> gibberish.
        2. trainer.sync_once(): the vendored engine spawns its producer server,
           bakes the plan on the inference side (init_weight_transfer_engine_rdt),
           then drives start_weight_update -> concurrent gather/publish +
           update_weights_rdt (workers pull their slices over NIXL) ->
           finish_weight_update.
        3. Query again -> correct output.
        """
        router_url = rdt_weight_update_env["router_url"]
        trainer = rdt_weight_update_env["trainer"]

        print("\n[TEST] Running sharded_rdt (NIXL pull) weight sync test (sidecar)")

        async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as http_client:
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            # ===== Step 1: dummy weights -> gibberish =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200
            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: drive the vendored sidecar engine end-to-end =====
            # sync_once rendezvouses (spawn server + bake) on the first call and
            # runs the full concurrent gather/pull handshake.
            print("[Step 2] trainer.sync_once() — bake + NIXL pull weight sync")
            await asyncio.to_thread(lambda: ray.get(trainer.sync_once.remote()))
            print("[Step 2] Weight sync complete")

            # ===== Step 3: real weights -> correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200
            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 3] Real weights output: {text_after!r}")
            assert "Paris" in text_after, f"RDT weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] sharded_rdt (sidecar) weight sync test passed!")
