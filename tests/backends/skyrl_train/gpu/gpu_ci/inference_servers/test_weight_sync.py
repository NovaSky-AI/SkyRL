"""
GPU CI tests for weight synchronization from trainer to inference server.

1. Non-colocated (NCCL broadcast), TP=2:
    - Trainer on GPUs 0-1, server (TP=2) on GPUs 2-3 (4 GPUs total)
    - Uses NCCL broadcast for weight sync via HTTP router

2. Colocated (CUDA IPC), TP=1:
    - Trainer and server share GPU 0 (2 GPUs total, 1 shared)
    - Uses CUDA IPC handles for zero-copy weight transfer

Run:
    uv run pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_weight_sync.py -v -s
"""

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
from skyrl.backends.skyrl_train.weight_sync import BroadcastInitInfo, CudaIpcInitInfo
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
            use_new_inference_servers=True,
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
            use_new_inference_servers=True,
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
                group_name=group_name,
                backend="nccl",
                model_dtype_str="bfloat16",
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
                f"[Step 3] Calling update_named_weights with {len(update_info['names'])} names, packed={update_info['packed']}"
            )
            result = await client.update_named_weights(update_info)
            print(f"[Step 3] update_named_weights returned: {list(result.keys())}")
            for server_url, resp in result.items():
                assert resp["status"] == 200, f"Server {server_url} update weights failed: {resp}"

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
        """Create IPC handles for all model parameters.

        Returns a dict matching the /update_weights API contract:
        names, dtype_names, shapes, and ipc_handles_pickled (base64).
        """
        from torch.multiprocessing.reductions import reduce_tensor

        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)

        names, dtype_names, shapes = [], [], []
        ipc_handles = []
        tensor_refs = []

        for name, param in self.model.named_parameters():
            weight = param.detach().contiguous()
            tensor_refs.append(weight)
            handle = reduce_tensor(weight)
            ipc_handles.append({gpu_uuid: handle})
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))

        # Prevent GC so IPC handles remain valid
        self._tensor_refs = tensor_refs

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        return {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
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
        use_new_inference_servers=True,
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

            result = await client.update_named_weights(update_info)
            for server_url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {server_url} IPC update failed: {resp_data}"
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

# Allowlist mirror for the producer-side op-chain replay (see
# weight_sync/sharded_rdt_engine.py LazyRDTTensor and the reference example).
_RDT_TEST_ALLOWED_OPS = frozenset(
    {
        "narrow",
        "view",
        "reshape",
        "__getitem__",
        "unsqueeze",
        "squeeze",
        "transpose",
        "t",
        "permute",
        "flatten",
        "contiguous",
        "chunk",
    }
)


@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class RdtTrainer:
    """Named NIXL trainer actor for the sharded_rdt backend (TP=1, no FSDP).

    Mirrors FSDPTrainWorker in the vllm-rdt-weight-sync reference example
    (examples/rl/rlhf_sharded_rdt_fsdp_ep.py): the vLLM inference workers pull
    only the slice each one consumes from this actor over NIXL, driven
    layer-by-layer. With FSDP world size 1 there is no sharding, so gather_layer
    simply caches the (whole) named params; produce replays each op chain and
    clones the resulting slice for NIXL.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        self._param_lookup = dict(self.model.named_parameters())
        self._cache = {}

    def ready(self):
        return True

    def get_weight_metadata(self) -> dict:
        names, dtype_names, shapes = [], [], []
        for name, param in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(param.dtype).split(".")[-1])
            shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}

    def gather_layer(self, names: list) -> None:
        for name in names:
            self._cache[name] = self._param_lookup[name].detach().contiguous()

    def free_group(self, names: list) -> None:
        for name in names:
            self._cache.pop(name, None)

    @ray.method(tensor_transport="nixl")
    def rdt_warmup(self):
        return torch.zeros(1, device=self.device)

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs):
        out = []
        for name, chain in specs:
            tensor = self._cache[name]
            for op_name, args, kwargs_items in chain:
                assert op_name in _RDT_TEST_ALLOWED_OPS, f"disallowed op {op_name!r}"
                tensor = getattr(tensor, op_name)(*args, **dict(kwargs_items))
            out.append(tensor.clone(memory_format=torch.contiguous_format))
        torch.accelerator.synchronize()
        return out


@pytest_asyncio.fixture(scope="class")
async def rdt_weight_update_env(class_scoped_ray_init_fixture):
    """Non-colocated sharded_rdt (NIXL pull) environment, TP=1.

    Trainer (named, tensor-transport) on its own GPU; vLLM server (TP=1,
    distributed_executor_backend=ray) on another GPU. 2 GPUs total.
    """
    from skyrl.backends.skyrl_train.weight_sync import RDT_TRAINER_ACTOR_NAME

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
        use_new_inference_servers=True,
        engine_init_kwargs={"load_format": "dummy"},
    )

    async with InferenceEngineState.create(cfg, **create_kwargs) as engines:
        trainer = RdtTrainer.options(name=RDT_TRAINER_ACTOR_NAME).remote(MODEL)
        ray.get(trainer.ready.remote())

        yield {
            "engines": engines,
            "trainer": trainer,
            "client": engines.client,
            "router_url": engines.client.proxy_url,
        }

        await engines.client.teardown()
        ray.kill(trainer)
    if engines.pg:
        ray.util.remove_placement_group(engines.pg)


@pytest.mark.asyncio(loop_scope="class")
class TestShardedRdtWeightUpdateFlow:
    """Weight sync via the sharded_rdt (NIXL pull) backend (non-colocated, TP=1)."""

    async def test_update_weights_rdt(self, rdt_weight_update_env):
        """
        Full E2E weight sync test (non-colocated, sharded RDT / NIXL pull):
        1. Query with dummy weights -> gibberish
        2. init_weight_transfer_engine_rdt (resolves trainer actor + bakes plan)
        3. start_weight_update -> per layer-group: gather + update_weights_rdt
           (workers pull their slices from the trainer over NIXL) + free
           -> finish_weight_update
        4. Query again -> correct output
        """
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_strategy import (
            RDT_TRAINER_ACTOR_NAME,
            layerwise_groups,
        )

        router_url = rdt_weight_update_env["router_url"]
        trainer = rdt_weight_update_env["trainer"]
        client = rdt_weight_update_env["client"]

        print("\n[TEST] Running sharded_rdt (NIXL pull) weight sync test")

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

            # ===== Step 2: init engine + bake on the inference side =====
            meta = ray.get(trainer.get_weight_metadata.remote())
            namespace = ray.get_runtime_context().namespace or None
            init_payload = {
                "trainer_actor_name": RDT_TRAINER_ACTOR_NAME,
                "trainer_actor_namespace": namespace,
                "produce_method_name": "rdt_produce_weights_batched",
                "names": meta["names"],
                "dtype_names": meta["dtype_names"],
                "shapes": meta["shapes"],
                "warmup_method_name": "rdt_warmup",
            }
            print(f"[Step 2] init_weight_transfer_engine_rdt: {len(meta['names'])} params")
            result = await client.init_weight_transfer_engine_rdt(init_payload)
            for url, resp_data in result.items():
                assert resp_data["status"] == 200, f"Server {url} RDT init failed: {resp_data}"

            # ===== Step 3: per-layer-group gather + pull =====
            groups = layerwise_groups(meta["names"])
            print(f"[Step 3] {len(groups)} layer-aligned groups")
            await client.start_weight_update(is_checkpoint_format=True)
            for group_names in groups:
                ray.get(trainer.gather_layer.remote(group_names))
                result = await client.update_weights_rdt({"names": group_names})
                for url, resp_data in result.items():
                    assert resp_data["status"] == 200, f"Server {url} RDT update failed: {resp_data}"
                ray.get(trainer.free_group.remote(group_names))
            await client.finish_weight_update()
            print("[Step 3] Weight sync complete")

            # ===== Step 4: real weights -> correct output =====
            resp = await http_client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200
            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 4] Real weights output: {text_after!r}")
            assert "Paris" in text_after, f"RDT weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] sharded_rdt weight sync test passed!")
