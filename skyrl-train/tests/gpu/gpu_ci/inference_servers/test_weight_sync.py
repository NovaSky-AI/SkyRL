"""
GPU CI tests for weight synchronization from trainer to inference server.

Tests non-colocated scenario with TP=2 on both sides:
    - Trainer (TP=2) on GPUs 0-1, server (TP=2) on GPUs 2-3 (4 GPUs total)
    - Uses NCCL broadcast for weight sync via HTTP router

Colocated test (CUDA IPC strategy) is deferred until vLLM weight sync endpoints are available.
See: https://github.com/vllm-project/vllm/issues/31848

Run:
    uv run pytest tests/gpu/gpu_ci/inference_servers/test_weight_sync.py -v -s
"""

import time

import httpx
import pytest
import ray
import torch
import argparse

from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from skyrl_train.inference_servers.common import get_node_ip, get_open_port
from skyrl_train.inference_servers.router import InferenceRouter
from skyrl_train.inference_servers.server_group import ServerGroup

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

    parser = FlexibleArgumentParser(description="vLLM server")
    parser = make_arg_parser(parser)
    return parser.parse_args(
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
        from skyrl_train.distributed.utils import init_custom_process_group
        from skyrl_train.utils import get_tcp_url

        self.pg = init_custom_process_group(
            backend="nccl",
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=0,  # Trainer is always rank 0
            group_name=group_name,
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
        for name, param in self.model.named_parameters():
            torch.distributed.broadcast(param.data, src=0, group=self.pg)
        torch.cuda.synchronize()

    def teardown(self):
        """Clean up the process group."""
        if self.pg is not None:
            torch.distributed.destroy_process_group(self.pg)
            self.pg = None


@pytest.fixture(scope="class")
def weight_update_env(ray_init_fixture):
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

    yield {
        "group": group,
        "server_urls": server_urls,
        "router": router,
        "router_url": router_url,
        "trainer": trainer,
    }

    router.shutdown()
    group.shutdown()
    ray.get(trainer.teardown.remote())


@pytest.mark.vllm
class TestWeightUpdateFlow:
    """Tests for weight synchronization from trainer to inference server (non-colocated)."""

    @pytest.mark.asyncio
    async def test_update_weights_flow(self, weight_update_env):
        """
        Full E2E weight sync test via router (non-colocated, NCCL broadcast):
        1. Query with dummy weights → gibberish
        2. Init weight transfer (both sides concurrently via router)
        3. Broadcast weights from trainer (concurrent with server receive)
        4. Finalize weight update
        5. Query again → correct output
        """
        router_url = weight_update_env["router_url"]
        trainer = weight_update_env["trainer"]

        print("\n[TEST] Running non-colocated weight sync test")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            # ===== Step 1: Verify dummy weights produce gibberish =====
            payload = {
                "model": MODEL,
                "prompt": "What is the capital of France?",
                "max_tokens": 32,
                "temperature": 0.0,
            }

            resp = await client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_before = resp.json()["choices"][0]["text"]
            print(f"[Step 1] Dummy weights output: {text_before!r}")

            # Dummy weights should NOT produce coherent output about Paris
            assert "Paris" not in text_before, "Dummy weights unexpectedly produced correct answer"

            # ===== Step 2: Init weight transfer (both sides concurrently) =====
            master_address = get_node_ip()
            master_port = get_open_port()

            # Query all servers for world_size (TP * PP) via router
            resp = await client.get(f"{router_url}/get_server_info")
            assert resp.status_code == 200
            server_info_map = resp.json()
            # Sum world_size across all servers
            inference_world_size = sum(info["world_size"] for info in server_info_map.values())
            world_size = 1 + inference_world_size  # 1 trainer + all inference workers
            group_name = f"weight_sync_test_{master_port}"

            print(f"[Step 2] Init weight transfer: master={master_address}:{master_port}, world_size={world_size}")

            init_info = {
                "master_addr": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": world_size,
                "group_name": group_name,
                "backend": "nccl",
                "model_dtype_str": "bfloat16",
                "override_existing_receiver": True,
            }

            # Both sides must init concurrently (NCCL blocks until all ranks join)
            # Start trainer init (returns immediately, runs in Ray actor)
            trainer_init_ref = trainer.init_weight_sync.remote(master_address, master_port, world_size, group_name)

            # Await server init (triggers NCCL join on server side)
            server_resp = await client.post(f"{router_url}/init_weight_transfer", json=init_info)
            assert server_resp.status_code == 200, f"Server init failed: {server_resp.text}"

            # Trainer should be done now (NCCL group formed)
            ray.get(trainer_init_ref)
            print("[Step 2] Both sides init complete")

            # ===== Step 3: Broadcast weights (concurrent send/receive) =====
            print("[Step 3] Broadcasting weights from trainer to server...")

            # Get weight metadata first (no NCCL yet)
            weight_info = ray.get(trainer.get_weight_info.remote())
            print(f"[Step 3] Weight info: {len(weight_info['names'])} parameters")

            # Start trainer broadcast (returns immediately, runs in Ray actor)
            trainer_broadcast_ref = trainer.broadcast_weights.remote()

            # Await server receive (triggers NCCL receive on server side)
            server_resp = await client.post(f"{router_url}/update_weights", json=weight_info)
            assert server_resp.status_code == 200, f"Update weights failed: {server_resp.text}"

            # Trainer should be done now (NCCL broadcast complete)
            ray.get(trainer_broadcast_ref)
            print("[Step 3] Weight sync complete")

            # ===== Step 4: Finalize weight update =====
            resp = await client.post(f"{router_url}/finalize_weight_update", json={})
            assert resp.status_code == 200
            print("[Step 4] Weight update finalized")

            # ===== Step 5: Query again - should produce correct output =====
            resp = await client.post(f"{router_url}/v1/completions", json=payload)
            assert resp.status_code == 200

            text_after = resp.json()["choices"][0]["text"]
            print(f"[Step 5] Real weights output: {text_after!r}")

            assert "Paris" in text_after, f"Weight sync failed - expected 'Paris' but got: {text_after!r}"

            print("[SUCCESS] Non-colocated weight sync test passed!")
