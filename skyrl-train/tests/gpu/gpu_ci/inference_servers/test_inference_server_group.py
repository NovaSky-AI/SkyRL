"""
GPU CI tests for ServerGroup + InferenceRouter.

Tests:
    - 2 vLLM servers with TP=2 (4 GPUs total)
    - Router with load balancing and control plane fan-out
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

    yield {
        "group": group,
        "server_urls": server_urls,
        "router": router,
        "router_url": router_url,
    }

    router.shutdown()
    group.shutdown()


@pytest.mark.vllm
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
            resp = await client.post(
                f"{router_url}/pause",
                json={"wait_for_inflight_request": False},
                timeout=30.0,
            )
            assert resp.status_code == 200

            # Check is paused - router returns aggregated responses from all servers
            resp = await client.get(f"{router_url}/is_paused", timeout=30.0)
            assert resp.status_code == 200
            # Response format: {server_url: {"status": 200, "body": {...}}}
            server_responses = resp.json()
            # All servers should report is_paused=True
            for server_url, server_resp in server_responses.items():
                assert server_resp["status"] == 200, f"Server {server_url} failed"
                assert server_resp["body"]["is_paused"] is True, f"Server {server_url} not paused"

            # Send a request while paused (should block)
            async def send_request():
                r = await client.post(
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

            # Resume
            resp = await client.post(f"{router_url}/resume", json={}, timeout=30.0)
            assert resp.status_code == 200

            # Verify that after resume, the request is completed
            result = await task
            assert result["choices"][0]["text"] is not None
