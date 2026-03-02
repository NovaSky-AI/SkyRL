"""
GPU CI tests for ServerGroup + InferenceRouter.

Tests:
    - 2 vLLM servers with TP=2 (4 GPUs total)
    - Router with load balancing (data plane only)
    - RemoteInferenceClient for control plane operations (fan-out)
    - Health, completions, get_server_info, session affinity, pause/resume

Run:
    uv run pytest tests/gpu/gpu_ci/test_inference_server_group.py -v -s
"""

import asyncio
import time

import httpx
import pytest
import argparse

from skyrl_train.inference_servers.common import get_open_port
from skyrl_train.inference_servers.router import InferenceRouter
from skyrl_train.inference_servers.server_group import ServerGroup
from skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_vllm_cli_args(
    model: str,
    tp_size: int = 2,
    load_format: str = "auto",
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
            "0.5",
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
        "client": client,
    }

    asyncio.get_event_loop().run_until_complete(client.teardown())
    router.shutdown()
    group.shutdown()


@pytest.mark.vllm
@pytest.mark.asyncio(loop_scope="class")
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

    async def test_get_world_size(self, server_group_and_router):
        """get_world_size returns total world size across all servers."""
        client = server_group_and_router["client"]

        # Each server has TP=2, we have 2 servers = total world_size of 4
        world_size = await client.get_world_size()
        print(f"Total world_size: {world_size}")
        assert world_size == 4

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

    async def test_pause_resume(self, server_group_and_router):
        """Pause/resume control plane operations work via RemoteInferenceClient."""
        router_url = server_group_and_router["router_url"]
        client = server_group_and_router["client"]

        # Pause using client (fans out to all servers)
        result = await client.pause()
        # All servers should report success
        for server_url, resp in result.items():
            assert resp["status"] == 200, f"Server {server_url} failed to pause: {resp}"

        # Send a request while paused (should block)
        async def send_request():
            async with httpx.AsyncClient() as http_client:
                r = await http_client.post(
                    f"{router_url}/v1/completions",
                    json={"model": MODEL, "prompt": "Test", "max_tokens": 4},
                    timeout=60.0,
                )
                assert r.status_code == 200
                return r.json()

        task = asyncio.create_task(send_request())
        await asyncio.sleep(1)

        # Task should not be done here (request blocked by pause)
        assert not task.done()

        # Resume using client (fans out to all servers)
        result = await client.resume()
        for server_url, resp in result.items():
            assert resp["status"] == 200, f"Server {server_url} failed to resume"

        # Verify that after resume, the request is completed
        result = await task
        assert result["choices"][0]["text"] is not None
