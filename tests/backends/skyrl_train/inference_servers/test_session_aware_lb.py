"""End-to-end integration tests for the ``sticky_least_loaded`` policy.

These spin up a *real* ``vllm-router`` in-process (via the ``vllm_router`` Python
package, which wraps the Rust router) pointing at several mock vLLM backends, and
drive it through a ``RemoteInferenceClient``. They verify:

- New sessions are balanced across replicas (least-loaded assignment).
- Requests for the same session are sticky (routed to the same replica).
- ``finish_session`` releases replica capacity so subsequent new sessions avoid
  the still-busy replica.

Run:
    uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_session_aware_lb.py

If the installed ``vllm_router`` does not support the policy, the whole module is
skipped.
"""

import multiprocessing
import time
from typing import List, Set

import httpx
import pytest
import pytest_asyncio

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter

# --- Skip the module unless the custom router (with the new policy) is available.
vllm_router = pytest.importorskip("vllm_router", reason="vllm_router package not installed")
try:
    from vllm_router.router_args import RouterArgs
    from vllm_router_rs import PolicyType

    _POLICY_SUPPORTED = hasattr(PolicyType, "StickyLeastLoaded")
except Exception:  # pragma: no cover - import-time capability probe
    _POLICY_SUPPORTED = False

pytestmark = pytest.mark.skipif(
    not _POLICY_SUPPORTED,
    reason="installed vllm_router does not support sticky_least_loaded",
)

NUM_BACKENDS = 3


def _run_backend_process(server_id: int, port: int) -> None:
    """Entry point for a mock vLLM backend running in its own process.

    Records the ``X-Session-ID`` of generate requests and exposes them via ``/test/seen``.

    NOTE: This runs in a separate process. Running in a separate thread is not recommended
    as the router's ``start()`` call will hold the GIL.
    """
    import uvicorn
    from fastapi import FastAPI, Request

    app = FastAPI()
    seen_sessions: Set[str] = set()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        session_id = request.headers.get("x-session-id")
        if session_id:
            seen_sessions.add(session_id)
        await request.json()  # drain body
        # Minimal response compatible with RemoteInferenceClient._generate_single
        return {
            "choices": [
                {
                    "request_id": "mock",
                    "token_ids": [server_id, server_id + 1],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -0.1}]},
                }
            ]
        }

    @app.post("/detokenize")
    async def detokenize(request: Request):
        await request.json()
        return {"prompt": "text"}

    @app.get("/test/seen")
    async def seen():
        return {"server_id": server_id, "seen": sorted(seen_sessions)}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


def _wait_ready(url: str, path: str = "/health", timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(f"{url}{path}", timeout=1.0).status_code == 200:
                return True
        except httpx.RequestError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def backends():
    """Start ``NUM_BACKENDS`` mock vLLM backends, each in its own process.

    Backends run in separate processes (not threads) so the router's blocking,
    GIL-holding ``start()`` cannot starve them, which would fail the router's
    worker health checks. Each backend encodes its ``server_id`` into the
    response ``token_ids`` so the test can attribute each request to a replica.
    """
    # "spawn" avoids forking the Ray-initialized parent (ray.init runs in the
    # autouse session fixture), which can leave inherited locks/threads wedged.
    ctx = multiprocessing.get_context("spawn")

    backend_ports = [get_open_port() for _ in range(NUM_BACKENDS)]
    backend_urls = [f"http://127.0.0.1:{p}" for p in backend_ports]

    procs: List[multiprocessing.Process] = []
    for i, port in enumerate(backend_ports):
        p = ctx.Process(target=_run_backend_process, args=(i, port), daemon=True)
        p.start()
        procs.append(p)
    for url in backend_urls:
        assert _wait_ready(url), f"backend {url} failed to start"

    yield backend_urls

    for p in procs:
        p.terminate()
        p.join(timeout=5)


@pytest.fixture
def router_url(backends):
    """A fresh router (and thus fresh session-tracking state) for each test.

    Recreating the router per test keeps the active-session bookkeeping
    isolated so per-test load-balancing assertions are deterministic.
    """
    args = RouterArgs(
        worker_urls=list(backends),
        host="0.0.0.0",
        port=get_open_port(),
        policy="sticky_least_loaded",
        worker_startup_timeout_secs=30,
        worker_startup_check_interval=1,
        health_check_interval_secs=1,
    )
    router = VLLMRouter(args)
    url = router.start()
    yield url
    router.shutdown()


@pytest_asyncio.fixture
async def client(router_url, backends):
    c = RemoteInferenceClient(
        proxy_url=router_url,
        server_urls=list(backends),
        data_parallel_size=1,
    )
    yield c
    await c.teardown()


async def _server_for_session(client: RemoteInferenceClient, session_id: str) -> int:
    """Send a generate request for ``session_id``; return the serving backend id.

    Each mock backend encodes its ``server_id`` as the first response token, so
    the routed replica can be identified from the client's output without
    relying on the router forwarding the ``X-Session-ID`` header downstream.
    """
    out = await client.generate(
        {"prompt_token_ids": [[1, 2, 3]], "session_ids": [session_id], "sampling_params": {"max_tokens": 4}},
    )
    return out["response_ids"][0][0]


@pytest.mark.asyncio
async def test_new_sessions_balanced_one_per_replica(client):
    """Distinct new sessions are balanced across distinct replicas."""
    sessions = [f"balance-{i}" for i in range(NUM_BACKENDS)]

    served_by = {s: await _server_for_session(client, s) for s in sessions}

    # Each new session goes to a distinct, least-loaded replica.
    assert sorted(served_by.values()) == list(range(NUM_BACKENDS)), f"expected one session per replica, got {served_by}"


@pytest.mark.asyncio
async def test_same_session_is_sticky(client):
    """Repeated requests for the same session reuse the same replica."""
    session_id = "sticky-session"

    served = [await _server_for_session(client, session_id) for _ in range(5)]

    assert len(set(served)) == 1, f"sticky session bounced across replicas: {served}"


@pytest.mark.asyncio
async def test_finish_session_frees_capacity(client):
    """After finishing a replica's sessions, new sessions avoid the busy replica."""
    # Assign one session per replica (balanced across all backends).
    base = [f"free-{i}" for i in range(NUM_BACKENDS)]
    served_by = {s: await _server_for_session(client, s) for s in base}
    assert sorted(served_by.values()) == list(
        range(NUM_BACKENDS)
    ), f"setup expected one session per replica, got {served_by}"

    # Keep the first session active; finish all the others.
    kept_session = base[0]
    kept_backend = served_by[kept_session]
    for s in base[1:]:
        await client.finish_session(s)

    # New sessions should go to the now-idle replicas, never the still-busy one.
    for s in ["after-free-0", "after-free-1"]:
        server_id = await _server_for_session(client, s)
        assert server_id != kept_backend, f"new session {s} wrongly routed to still-busy replica {kept_backend}"
