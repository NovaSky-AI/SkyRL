"""
GPU CI tests for inference server infrastructure.

Test Suite 1: ServerGroup + InferenceRouter
    - 2 vLLM servers with TP=2 (4 GPUs total)
    - Router with load balancing and control plane fan-out
    - Tests: health, completions, get_server_info, session affinity

Test Suite 2: Weight Update Flow
    - 1 vLLM server with TP=2 + dummy weights
    - Trainer emulation with real weights
    - Tests: pause/resume, init_weight_transfer, weight sync effectiveness

Run:
    uv run pytest tests/gpu/gpu_ci/test_inference_server_group.py -v -s
"""

import asyncio
import time

import httpx
import pytest
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import torch
from transformers import AutoModelForCausalLM
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser

from skyrl_train.inference_servers.router import InferenceRouter
from skyrl_train.inference_servers.server_group import ServerGroup
from skyrl_train.inference_servers.common import get_open_port, get_node_ip

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# =============================================================================
# Utility: Skip tests if not enough GPUs
# =============================================================================

# Skip entire module if not enough GPUs
_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
if _gpu_count < 4:
    pytest.skip(f"Need 4 GPUs for full test suite, found {_gpu_count}", allow_module_level=True)



def make_vllm_cli_args(
    model: str,
    tp_size: int = 2,
    load_format: str = "auto",
) -> FlexibleArgumentParser:
    """Create CLI args for vLLM server using official parser."""
    parser = FlexibleArgumentParser(description="vLLM server")
    parser = make_arg_parser(parser)
    return parser.parse_args([
        "--model", model,
        "--tensor-parallel-size", str(tp_size),
        "--enforce-eager",
        "--gpu-memory-utilization", "0.5",
        "--max-model-len", "2048",
        "--load-format", load_format,
    ])


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


# =============================================================================
# Test Suite 1: ServerGroup + Router (2 servers, TP=2 each, 4 GPUs)
# =============================================================================

@pytest.fixture(scope="class")
def server_group_and_router(ray_init_fixture):
    """Create 2 vLLM servers (TP=2 each) + router."""
    cli_args = make_vllm_cli_args(MODEL, tp_size=2)
    start_port = get_open_port()

    # Create server group with 2 servers
    group = ServerGroup(
        cli_args=cli_args,
        num_servers=2,
        start_port=start_port,
    )
    server_infos = group.start()
    server_urls = [info.url for info in server_infos]

    # Wait for servers
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
    }

    router.shutdown()
    group.shutdown()
    del group
    del router


class TestServerGroupAndRouter:
    """Tests for ServerGroup + InferenceRouter with 2 TP=2 servers."""

    def test_health_check(self, server_group_and_router):
        """Health endpoint works through router."""
        router_url = server_group_and_router["router_url"]
        resp = httpx.get(f"{router_url}/health", timeout=10.0)
        assert resp.status_code == 200

    def test_list_servers(self, server_group_and_router):
        """/servers returns all backends."""
        router_url = server_group_and_router["router_url"]
        resp = httpx.get(f"{router_url}/servers", timeout=10.0)
        assert resp.status_code == 200
        assert len(resp.json()["servers"]) == 2

    def test_get_server_info(self, server_group_and_router):
        """/get_server_info returns mapping of server_url -> info for all servers."""
        router_url = server_group_and_router["router_url"]
        server_urls = server_group_and_router["server_urls"]
        
        resp = httpx.get(f"{router_url}/get_server_info", timeout=10.0)
        assert resp.status_code == 200
        info_map = resp.json()
        print(f"Server info map: {info_map}")
        
        # Should have info for each server
        assert len(info_map) == 2
        for url in server_urls:
            assert url in info_map
            server_info = info_map[url]
            # Each server has TP=2, so per-server world_size=2
            assert server_info["world_size"] == 2

    def test_completion_request(self, server_group_and_router):
        """Completion requests work through router."""
        router_url = server_group_and_router["router_url"]

        payload = {
            "model": MODEL,
            "prompt": "What is 2 + 2? Answer:",
            "max_tokens": 16,
            "temperature": 0.0,
        }

        resp = httpx.post(f"{router_url}/v1/completions", json=payload, timeout=60.0)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]
        print(f"Completion: {data['choices'][0]['text']}")

    @pytest.mark.asyncio
    async def test_pause_resume(self, server_group_and_router):
        """Pause/resume control plane routes work."""
        router_url = server_group_and_router["router_url"]

        async with httpx.AsyncClient() as client:
            # Pause
            resp = await client.post(f"{router_url}/pause", json={"wait_for_inflight_request": False}, timeout=30.0)
            assert resp.status_code == 200
            
            # Check is paused
            resp = await client.get(f"{router_url}/is_paused", timeout=30.0)
            assert resp.status_code == 200
            assert resp.json()["is_paused"] == True
            
            # Send a request while paused (should block)
            async def send_request():
                r = await client.post(f"{router_url}/v1/completions", json={"model": MODEL, "prompt": "Test", "max_tokens": 4}, timeout=60.0)
                assert r.status_code == 200
                return r.json()
            
            task = asyncio.create_task(send_request())
            await asyncio.sleep(1)
            
            # Task should not be done here (request blocked by pause)
            assert not task.done()
            
            # Resume
            resp = await client.post(f"{router_url}/resume", json={}, timeout=30.0)
            assert resp.status_code == 200
            
            # Verify that after resume, the request is completed
            result = await task
            assert result["choices"][0]["text"] is not None



# =============================================================================
# Test Suite 2: Weight Update Flow (1 server TP=2 + trainer emulation)
# =============================================================================

class Trainer:
    """
    Simple trainer emulator that holds the real model weights.
    
    This is a simplified version of the trainer side for testing weight sync.
    Non-colocated: runs on a separate GPU from the inference server.
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
    Create environment for weight update testing:
    - Trainer with real weights on GPU 0
    - 1 vLLM server with TP=2 and DUMMY weights (uses GPU 1,2)
    - Router to proxy requests
    """
    # Create server with dummy weights
    cli_args = make_vllm_cli_args(MODEL, tp_size=2, load_format="dummy")
    start_port = get_open_port()
    
    pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(3)])
    ray.get(pg.ready())
    
    trainer = ray.remote(Trainer).options(
        num_gpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        ),
    ).remote(MODEL)
    
    ray.get(trainer.ready.remote())

    group = ServerGroup(
        cli_args=cli_args,
        num_servers=1,
        start_port=start_port,
        placement_group=pg,
        placement_group_bundle_offset=1,
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
    del router
    del group
    del trainer
    del pg

class TestWeightUpdateFlow:
    """Tests for weight synchronization from trainer to inference server."""

    @pytest.mark.asyncio
    async def test_update_weights_flow(self, weight_update_env):
        """
        Full E2E weight sync test via router:
        1. Query with dummy weights → gibberish
        2. Init weight transfer (both sides concurrently via router)
        3. Broadcast weights from trainer (concurrent with server receive)
        4. Finalize weight update
        5. Query again → correct output
        """
        router_url = weight_update_env["router_url"]
        trainer = weight_update_env["trainer"]

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
            
            print("[SUCCESS] Weight sync test passed!")

