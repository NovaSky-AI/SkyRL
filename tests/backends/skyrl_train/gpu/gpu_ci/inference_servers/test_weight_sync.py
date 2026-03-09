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
import pickle
import time

import httpx
import pytest
import ray
import torch
import argparse

import pytest_asyncio
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from skyrl.backends.skyrl_train.inference_servers.common import get_node_ip, get_open_port
from skyrl.backends.skyrl_train.inference_servers.router import InferenceRouter
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
from skyrl.backends.skyrl_train.weight_sync import BroadcastInitInfo, CudaIpcInitInfo

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_vllm_cli_args(
    model: str,
    tp_size: int = 2,
    load_format: str = "auto",
    gpu_memory_utilization: float = 0.5,
) -> argparse.Namespace:
    """Create CLI args for vLLM server using official parser."""
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.config import WeightTransferConfig

    parser = FlexibleArgumentParser(description="vLLM server")
    parser = make_arg_parser(parser)
    args = parser.parse_args(
        [
            "--model",
            model,
            "--tensor-parallel-size",
            str(tp_size),
            "--enforce-eager",
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            "2048",
            "--load-format",
            load_format,
        ]
    )
    args.weight_transfer_config = WeightTransferConfig(backend="nccl")
    return args


def wait_for_url(url: str, timeout: float = 180.0) -> bool:
    """Wait for a URL to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(2.0)
    return False


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
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

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
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        params = list(self.model.named_parameters())
        print(
            f"[Trainer.broadcast_weights] Starting send of {len(params)} params, pg={self.pg}, pg.rank={self.pg.rank}, pg.world_size={self.pg.world_size}"
        )
        try:
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(params),
                group=self.pg,
                packed=True,
            )
            torch.cuda.synchronize()
            print("[Trainer.broadcast_weights] Send complete")
        except Exception as e:
            print(f"[Trainer.broadcast_weights] ERROR: {e}")
            raise


@pytest_asyncio.fixture(scope="class")
async def weight_update_env(ray_init_fixture):
    """
    Create environment for weight update testing.

    Non-colocated setup with TP=2 for both trainer and inference server:
    - 4 GPUs total: trainer on GPUs 0-1, server on GPUs 2-3
    - Uses NCCL broadcast for weight sync
    """

    # Create server with dummy weights (TP=2)
    cli_args = make_vllm_cli_args(
        MODEL,
        tp_size=2,
        load_format="dummy",
        gpu_memory_utilization=0.5,
    )
    start_port = get_open_port()

    # 4 bundles: trainer on 0-1, server on 2-3
    pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(4)])
    ray.get(pg.ready())

    # Trainer on bundle 0 (uses GPU 0-1 with TP=2 via the model itself)
    trainer = Trainer.options(
        num_gpus=1.0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(MODEL)

    ray.get(trainer.ready.remote())

    # Server on bundles 2-3 (separate from trainer)
    group = ServerGroup(
        cli_args=cli_args,
        num_servers=1,
        start_port=start_port,
        placement_group=pg,
        placement_group_bundle_offset=2,
    )
    server_infos = group.start()
    server_urls = [info.url for info in server_infos]

    for url in server_urls:
        assert wait_for_url(url), f"Server {url} failed to start"

    # Create router
    router_port = get_open_port()
    router = InferenceRouter(server_urls, host="0.0.0.0", port=router_port)
    router_url = router.start()
    assert wait_for_url(router_url), "Router failed to start"

    # Create RemoteInferenceClient for control plane operations
    client = RemoteInferenceClient(
        proxy_url=router_url,
        server_urls=server_urls,
        model_name=MODEL,
    )

    yield {
        "group": group,
        "server_urls": server_urls,
        "router": router,
        "router_url": router_url,
        "trainer": trainer,
        "client": client,
    }

    await client.teardown()
    router.shutdown()


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
async def ipc_weight_update_env(ray_init_fixture):
    """
    Create environment for colocated IPC weight update testing.

    Colocated setup with TP=1:
    - 2 GPU bundles, trainer and server share bundle 0 (GPU 0)
    - Server uses CUDA IPC backend for weight sync
    """
    from skyrl.backends.skyrl_train.weight_sync.vllm_ipc_engine import _patch_weight_transfer_config
    from vllm.config import WeightTransferConfig

    _patch_weight_transfer_config()

    cli_args = make_vllm_cli_args(
        MODEL,
        tp_size=1,
        load_format="dummy",
        gpu_memory_utilization=0.5,
    )
    cli_args.weight_transfer_config = WeightTransferConfig(backend="ipc")
    cli_args.enable_sleep_mode = True

    start_port = get_open_port()

    pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(2)])
    ray.get(pg.ready())

    # Trainer on bundle 0 — fractional GPU so it coexists with the server
    trainer = IpcTrainer.options(
        num_gpus=0.2,
        num_cpus=0.2,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(MODEL)
    ray.get(trainer.ready.remote())

    # Server on bundle 0 (colocated — same GPU as trainer)
    group = ServerGroup(
        cli_args=cli_args,
        num_servers=1,
        start_port=start_port,
        placement_group=pg,
        placement_group_bundle_offset=0,
    )
    server_infos = group.start()
    server_urls = [info.url for info in server_infos]

    for url in server_urls:
        assert wait_for_url(url), f"Server {url} failed to start"

    router_port = get_open_port()
    router = InferenceRouter(server_urls, host="0.0.0.0", port=router_port)
    router_url = router.start()
    assert wait_for_url(router_url), "Router failed to start"

    client = RemoteInferenceClient(
        proxy_url=router_url,
        server_urls=server_urls,
        model_name=MODEL,
    )

    yield {
        "group": group,
        "server_urls": server_urls,
        "router": router,
        "router_url": router_url,
        "trainer": trainer,
        "client": client,
    }

    await client.teardown()
    router.shutdown()


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
