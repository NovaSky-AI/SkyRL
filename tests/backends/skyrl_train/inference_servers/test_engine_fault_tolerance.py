"""Part 1 of inference-engine fault tolerance: surviving an engine death.

Covers the client-side half — membership state, the reactive `/health` probe and
its debounce, the hardened `_post` retry set, degraded control-plane fan-out, and
the router worker-management calls — plus the config gates that decide when any of
it is allowed to run.

The vendored half (live-set free targets, the producer stall watchdog, `set_live`)
is covered in `tests/backends/skyrl_train/weight_sync/`.

Everything here is CPU-only. Backends are real uvicorn servers in threads so a
"death" is a genuine connection refusal rather than a mock's raise; the two tests
that need a real Rust router spin one up and are skipped if `vllm_router` is
missing.

Run:
    uv run --extra dev --extra fsdp pytest \
        tests/backends/skyrl_train/inference_servers/test_engine_fault_tolerance.py
"""

import asyncio
import multiprocessing
import pickle
import threading
import time
from typing import List, Optional

import aiohttp
import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from skyrl.backends.skyrl_train.inference_servers.common import get_open_port
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    EngineFleetError,
    RemoteInferenceClient,
)
from skyrl.train.config.config import InferenceFaultToleranceConfig

# --------------------------------------------------------------------------
# Mock backends
# --------------------------------------------------------------------------


class _Backend:
    """A killable mock vLLM backend.

    Three failure shapes, because they reach the client through different paths:
      * ``kill()``    -> the socket closes; new connections are refused (a crash).
      * ``wedge()``   -> connections are accepted, requests never answered (a hang).
      * ``sick()``    -> ``/health`` fails but the data plane still answers, which
                         is what the probe has to notice on its own.
    """

    def __init__(self, server_id: int):
        self.server_id = server_id
        self.port = get_open_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.health_calls = 0
        self.post_calls: List[str] = []
        self._wedged = False
        self._sick = False
        self._server: Optional[uvicorn.Server] = None
        self._app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health():
            self.health_calls += 1
            if self._sick:
                return JSONResponse({"status": "unhealthy"}, status_code=500)
            if self._wedged:
                await asyncio.sleep(3600)
            return {"status": "ok"}

        @app.post("/echo")
        async def echo():
            self.post_calls.append("/echo")
            if self._wedged:
                await asyncio.sleep(3600)
            return {"server_id": self.server_id}

        @app.post("/pause")
        async def pause():
            self.post_calls.append("/pause")
            return {"status": "paused"}

        @app.get("/get_world_size")
        async def world_size():
            return {"world_size": 1}

        return app

    def start(self) -> "_Backend":
        config = uvicorn.Config(self._app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        threading.Thread(target=lambda: asyncio.run(self._server.serve()), daemon=True).start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=1.0).status_code == 200:
                    self.health_calls = 0
                    return self
            except httpx.RequestError:
                time.sleep(0.05)
        raise RuntimeError(f"backend {self.url} failed to start")

    def wedge(self) -> None:
        self._wedged = True

    def sick(self) -> None:
        self._sick = True

    def kill(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    httpx.get(f"{self.url}/health", timeout=0.5)
                except httpx.RequestError:
                    return
                time.sleep(0.05)
            raise RuntimeError(f"backend {self.url} did not stop")


@pytest.fixture
def backends():
    made: List[_Backend] = []

    def _make(n: int) -> List[_Backend]:
        for i in range(n):
            made.append(_Backend(i).start())
        return made

    yield _make
    for b in made:
        b.kill()


def _ft(**overrides) -> InferenceFaultToleranceConfig:
    cfg = InferenceFaultToleranceConfig(enabled=True, health_probe_timeout_s=2.0, request_timeout_s=3.0)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _client(urls: List[str], proxy: Optional[str] = None, ft=None) -> RemoteInferenceClient:
    return RemoteInferenceClient(
        proxy_url=proxy if proxy is not None else urls[0],
        server_urls=list(urls),
        data_parallel_size=1,
        fault_tolerance=ft,
    )


# --------------------------------------------------------------------------
# Membership state
# --------------------------------------------------------------------------


class TestMembershipViews:
    """``server_urls`` is identity; ``active_server_urls`` is reachability.

    Conflating them is the dangerous mistake on this path: ``replica_rank`` is a
    position in ``server_urls``, so compacting or rebuilding that list would
    silently re-map every surviving consumer's weight slice.
    """

    def test_active_equals_provisioned_before_any_failure(self):
        urls = ["http://a", "http://b", "http://c"]
        c = _client(urls, ft=_ft())
        assert c.active_server_urls == urls
        assert c.disabled_server_urls == []
        assert c.membership_generation == 0

    def test_disabling_preserves_order_and_never_compacts(self):
        urls = ["http://a", "http://b", "http://c"]
        c = _client(urls, ft=_ft())
        c._disabled_server_urls.add("http://b")
        assert c.active_server_urls == ["http://a", "http://c"]
        assert c.disabled_server_urls == ["http://b"]
        # The identity list is untouched, so index 2 is still "c".
        assert c.server_urls == urls and c.server_urls.index("http://c") == 2

    def test_ft_disabled_leaves_the_views_inert(self):
        c = _client(["http://a", "http://b"], ft=None)
        assert c._ft_enabled is False
        assert c.active_server_urls == c.server_urls

    def test_sync_membership_adopts_a_view_without_probing(self):
        """A worker's cached copy is pickled at init and can never reconcile, so the
        driver hands it the answer. Local bookkeeping only — no HTTP at all."""
        urls = ["http://a", "http://b", "http://c"]
        c = _client(urls, ft=_ft())
        c.sync_membership(["http://a", "http://c"])
        assert c.active_server_urls == ["http://a", "http://c"]
        assert c.membership_generation == 1

    def test_sync_membership_none_means_the_whole_fleet(self):
        urls = ["http://a", "http://b"]
        c = _client(urls, ft=_ft())
        c.sync_membership(["http://a"])
        c.sync_membership(None)
        assert c.active_server_urls == urls
        assert c.membership_generation == 2

    def test_sync_membership_is_idempotent(self):
        """It runs on every sync; repeating the same view must not churn the
        generation, which other callers use to decide whether to re-probe."""
        c = _client(["http://a", "http://b"], ft=_ft())
        c.sync_membership(["http://a"])
        gen = c.membership_generation
        c.sync_membership(["http://a"])
        assert c.membership_generation == gen

    def test_sync_membership_rejects_an_unprovisioned_url(self):
        c = _client(["http://a", "http://b"], ft=_ft())
        with pytest.raises(ValueError, match="outside the provisioned set"):
            c.sync_membership(["http://a", "http://zzz"])

    def test_membership_survives_pickling_but_the_probe_task_does_not(self):
        """Worker copies carry a snapshot (they can never reconcile), so the live
        set has to be passed to the weight sync explicitly rather than read here."""
        c = _client(["http://a", "http://b"], ft=_ft())
        c._disabled_server_urls.add("http://b")
        c._membership_generation = 4
        c._reconcile_task = object()
        c._reconcile_loop = object()

        restored = pickle.loads(pickle.dumps(c))
        assert restored.active_server_urls == ["http://a"]
        assert restored.membership_generation == 4
        assert restored._reconcile_task is None and restored._reconcile_loop is None


# --------------------------------------------------------------------------
# Reactive detection
# --------------------------------------------------------------------------


class TestReconcileFleet:
    @pytest.mark.asyncio
    async def test_a_dead_backend_is_probed_out_of_the_active_set(self, backends):
        b0, b1, b2 = backends(3)
        c = _client([b.url for b in (b0, b1, b2)], ft=_ft())
        try:
            b1.kill()
            await c._reconcile_fleet()
            assert c.active_server_urls == [b0.url, b2.url]
            assert c.membership_generation == 1
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_an_unhealthy_but_reachable_backend_is_still_dead(self, backends):
        """The probe reads ``/health``, not reachability: a backend whose engine has
        fallen over while its HTTP server survives must not keep taking traffic."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            b1.sick()
            await c._reconcile_fleet()
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_wedged_backend_is_dead_within_the_probe_timeout(self, backends):
        """A backend that accepts and never answers is the case an unbounded client
        cannot see at all. The probe must bound it."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft(health_probe_timeout_s=0.5))
        try:
            b1.wedge()
            t0 = time.monotonic()
            await c._reconcile_fleet()
            assert time.monotonic() - t0 < 5.0
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_healthy_fleet_changes_nothing(self, backends):
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            await c._reconcile_fleet()
            assert c.active_server_urls == [b0.url, b1.url]
            assert c.membership_generation == 0
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_burst_of_callers_triggers_exactly_one_probe(self, backends):
        """The debounce. A batch of 512 trajectories failing at once must cost one
        fleet probe, not 512 — each backend answers /health once per probe, so the
        health-call count is the probe count."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            b1.kill()
            await asyncio.gather(*[c._reconcile_fleet(seen_generation=0) for _ in range(32)])
            assert b0.health_calls == 1, f"expected one probe, saw {b0.health_calls}"
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_caller_behind_the_generation_skips_probing(self, backends):
        """Someone else already reconciled since this caller last looked, so their
        probe covers it and this one returns without touching the fleet."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            c._membership_generation = 7
            await c._reconcile_fleet(seen_generation=3)
            assert b0.health_calls == 0
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_ft_disabled_never_probes(self, backends):
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=None)
        try:
            b1.kill()
            await c._reconcile_fleet()
            assert b0.health_calls == 0
            assert c.active_server_urls == [b0.url, b1.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_dropping_below_the_floor_raises(self, backends):
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft(min_live_engines=2))
        try:
            b1.kill()
            with pytest.raises(EngineFleetError, match="min_live_engines"):
                await c._reconcile_fleet()
            # The URL is still recorded dead — the state is true regardless.
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_every_waiter_sees_the_floor_error(self, backends):
        """One shared probe, so its failure has to reach all of its awaiters rather
        than only the caller that happened to create the task."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft(min_live_engines=2))
        try:
            b1.kill()
            results = await asyncio.gather(
                *[c._reconcile_fleet(seen_generation=0) for _ in range(8)],
                return_exceptions=True,
            )
            assert all(isinstance(r, EngineFleetError) for r in results)
        finally:
            await c.teardown()


# --------------------------------------------------------------------------
# _post hardening
# --------------------------------------------------------------------------


class _StatusServer:
    """A one-endpoint server that returns a scripted sequence of statuses."""

    def __init__(self, statuses: List[int]):
        self.port = get_open_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.statuses = list(statuses)
        self.calls = 0
        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/echo")
        async def echo():
            i = min(self.calls, len(self.statuses) - 1)
            self.calls += 1
            status = self.statuses[i]
            if status == 200:
                return {"ok": True}
            return JSONResponse({"error": {"message": "backend down"}}, status_code=status)

        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)

    def start(self) -> "_StatusServer":
        threading.Thread(target=lambda: asyncio.run(self._server.serve()), daemon=True).start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.url}/health", timeout=1.0).status_code == 200:
                    return self
            except httpx.RequestError:
                time.sleep(0.05)
        raise RuntimeError("status server failed to start")

    def stop(self) -> None:
        self._server.should_exit = True


@pytest.fixture
def status_server():
    made: List[_StatusServer] = []

    def _make(statuses):
        made.append(_StatusServer(statuses).start())
        return made[-1]

    yield _make
    for s in made:
        s.stop()


class TestPostHardening:
    @pytest.mark.asyncio
    async def test_router_5xx_is_retried_under_ft(self, status_server):
        """A router 502/503/504 is the actual signature of a dead backend. Without
        this it raises straight out of ``raise_for_status`` and kills the trajectory."""
        srv = status_server([503, 200])
        c = _client([srv.url], proxy=srv.url, ft=_ft())
        try:
            body = await c._post(f"{srv.url}/echo", {})
            assert body == {"ok": True}
            assert srv.calls == 2
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_router_5xx_still_raises_without_ft(self, status_server):
        """The retry changes error semantics (a genuine server error now looks
        transient), so it stays behind the switch."""
        srv = status_server([503, 200])
        c = _client([srv.url], proxy=srv.url, ft=None)
        try:
            with pytest.raises(aiohttp.ClientResponseError) as ei:
                await c._post(f"{srv.url}/echo", {})
            assert ei.value.status == 503
            assert srv.calls == 1
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_client_errors_are_never_retried(self, status_server):
        """4xx means the request is wrong; retrying 30 times helps nobody."""
        srv = status_server([422, 200])
        c = _client([srv.url], proxy=srv.url, ft=_ft())
        try:
            with pytest.raises(aiohttp.ClientResponseError) as ei:
                await c._post(f"{srv.url}/echo", {})
            assert ei.value.status == 422
            assert srv.calls == 1
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_wedged_backend_times_out_instead_of_hanging(self, monkeypatch, backends):
        """The session has no timeout, so without ``request_timeout_s`` this call
        never returns. Bounded, it becomes a failure the retry path can act on.

        Retries are cut to 1 so the test is about the bound, not about waiting out
        30 x (timeout + backoff)."""
        from skyrl.backends.skyrl_train.inference_servers import (
            remote_inference_client as ric,
        )

        monkeypatch.setattr(ric, "_DATA_PLANE_RETRIES", 1)
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], proxy=b1.url, ft=_ft(request_timeout_s=0.5))
        try:
            b1.wedge()
            t0 = time.monotonic()
            with pytest.raises(asyncio.TimeoutError):
                await c._post(f"{b1.url}/echo", {})
            assert time.monotonic() - t0 < 10
            # The probe classified the wedged backend as dead on the way through.
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_the_fleet_floor_error_propagates_out_of_a_request(self, monkeypatch, backends):
        """A retry that reconciles below the floor must surface ``EngineFleetError``
        rather than grinding through 30 attempts against a fleet that cannot serve."""
        from skyrl.backends.skyrl_train.inference_servers import (
            remote_inference_client as ric,
        )

        monkeypatch.setattr(ric, "_DATA_PLANE_RETRIES", 5)
        b0 = backends(1)[0]
        c = _client([b0.url], proxy=b0.url, ft=_ft())
        try:
            b0.kill()
            with pytest.raises(EngineFleetError):
                await c._post(f"{b0.url}/echo", {})
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_failure_reconciles_before_retrying(self, backends):
        """The retry has to be issued AFTER the dead backend has left the router,
        or consistent_hash sends it straight back to the same corpse."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], proxy=b0.url, ft=_ft())
        try:
            b1.kill()
            b0.health_calls = 0
            with pytest.raises(Exception):
                # b1 is dead: connecting to it fails, retries, and gives up.
                await asyncio.wait_for(c._post(f"{b1.url}/echo", {}), timeout=15)
            assert b0.health_calls >= 1, "the first retryable failure must reconcile"
            assert c.active_server_urls == [b0.url]
        finally:
            await c.teardown()


# --------------------------------------------------------------------------
# Control-plane fan-out
# --------------------------------------------------------------------------


class TestControlPlaneDegrades:
    @pytest.mark.asyncio
    async def test_pause_succeeds_on_the_survivors(self, backends):
        """pause/resume bracket every non-colocated weight sync, so a bare gather
        here means one dead engine breaks the sync path before RDT is involved."""
        b0, b1, b2 = backends(3)
        c = _client([b.url for b in (b0, b1, b2)], ft=_ft())
        try:
            b1.kill()
            results = await c._call_all_servers("/pause")
            assert set(results) == {b0.url, b2.url}
            assert c.active_server_urls == [b0.url, b2.url], "the failure must reconcile"
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_dead_server_aborts_the_fanout_without_ft(self, backends):
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=None)
        try:
            b1.kill()
            with pytest.raises(Exception):
                await c._call_all_servers("/pause")
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_totally_dead_fleet_raises_rather_than_returning_empty(self, backends):
        """``{}`` would read as "fan-out succeeded, no servers" to every caller.

        The reconcile runs first, so the error is the probe's verdict — the whole
        fleet is gone — rather than one server's transport exception."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            b0.kill()
            b1.kill()
            with pytest.raises(EngineFleetError):
                await c._call_all_servers("/pause")
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_an_already_empty_active_set_raises(self, backends):
        """Reachable after a reconcile that breached the floor: it disables the URLs
        before raising, so a later fan-out finds nothing to call."""
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            c._disabled_server_urls.update({b0.url, b1.url})
            with pytest.raises(EngineFleetError, match="every inference server is marked dead"):
                await c._call_all_servers("/pause")
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_a_disabled_server_is_not_called_at_all(self, backends):
        b0, b1 = backends(2)
        c = _client([b0.url, b1.url], ft=_ft())
        try:
            c._disabled_server_urls.add(b1.url)
            b1.post_calls.clear()
            results = await c._call_all_servers("/pause")
            assert set(results) == {b0.url}
            assert b1.post_calls == []
        finally:
            await c.teardown()


# --------------------------------------------------------------------------
# Router worker management (real Rust router)
# --------------------------------------------------------------------------

vllm_router = pytest.importorskip("vllm_router", reason="vllm_router package not installed")


def _router_backend_process(port: int) -> None:
    import uvicorn as _uvicorn
    from fastapi import FastAPI as _FastAPI

    app = _FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    _uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")


@pytest.fixture(scope="module")
def router_fleet():
    """A real vllm-router over three mock backends, two registered up front.

    Backends run in separate processes: the router's ``start()`` blocks holding the
    GIL, which would starve in-process backends and fail its startup health checks.
    """
    from vllm_router.router_args import RouterArgs

    from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter

    ctx = multiprocessing.get_context("spawn")
    ports = [get_open_port() for _ in range(3)]
    urls = [f"http://127.0.0.1:{p}" for p in ports]
    procs = [ctx.Process(target=_router_backend_process, args=(p,), daemon=True) for p in ports]
    for p in procs:
        p.start()
    for u in urls:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(f"{u}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.RequestError:
                time.sleep(0.05)
        else:
            raise RuntimeError(f"router backend {u} failed to start")

    router = VLLMRouter(
        RouterArgs(
            worker_urls=urls[:2],
            host="0.0.0.0",
            port=get_open_port(),
            policy="consistent_hash",
            worker_startup_timeout_secs=30,
            worker_startup_check_interval=1,
            # High enough that the router's own health checker does not race the
            # assertions -- these tests are about OUR removal, not its eviction.
            health_check_interval_secs=300,
        )
    )
    router_url = router.start()
    yield {"router": router, "url": router_url, "urls": urls}
    router.shutdown()
    for p in procs:
        p.terminate()
        p.join(timeout=5)


class TestRouterWorkerManagement:
    """The three routes the design leans on, against the real binary.

    The spellings are inherited from sglang-router and had never been exercised
    here, so these pin them: ``url`` is the query parameter (anything else is a 400
    ``missing field``), removal is idempotent, and addition is not.
    """

    def test_driver_side_round_trip(self, router_fleet):
        router, urls = router_fleet["router"], router_fleet["urls"]
        assert set(router.list_workers()) == set(urls[:2])

        router.add_worker(urls[2])
        assert set(router.list_workers()) == set(urls)

        router.remove_worker(urls[2])
        assert set(router.list_workers()) == set(urls[:2])

    def test_removal_is_idempotent(self, router_fleet):
        """It runs on the failure path, possibly more than once for the same URL;
        a second removal must not be an error."""
        router, urls = router_fleet["router"], router_fleet["urls"]
        router.remove_worker(urls[2])  # never added
        router.remove_worker(urls[2])
        assert set(router.list_workers()) == set(urls[:2])

    def test_adding_a_known_worker_is_not_an_error(self, router_fleet):
        """The router answers 400 ``already exists``; the post-condition holds, so
        the wrapper normalizes it."""
        router, urls = router_fleet["router"], router_fleet["urls"]
        router.add_worker(urls[0])
        assert set(router.list_workers()) == set(urls[:2])

    @pytest.mark.asyncio
    async def test_client_side_round_trip(self, router_fleet):
        urls = router_fleet["urls"]
        c = _client(urls[:2], proxy=router_fleet["url"], ft=_ft())
        try:
            assert set(await c.router_list_workers()) == set(urls[:2])
            assert await c._router_add_worker(urls[2]) is True
            assert set(await c.router_list_workers()) == set(urls)
            assert await c._router_remove_worker(urls[2]) is True
            assert set(await c.router_list_workers()) == set(urls[:2])
            # Idempotent / already-present, both reported as success.
            assert await c._router_remove_worker(urls[2]) is True
            assert await c._router_add_worker(urls[0]) is True
        finally:
            await c.teardown()

    @pytest.mark.asyncio
    async def test_router_calls_never_raise(self, router_fleet):
        """De-routing is best-effort: the URL is already locally disabled and the
        router's own health check is the backstop, so a failure here must not
        propagate out of the reconcile and kill the run."""
        c = _client(router_fleet["urls"][:2], proxy="http://127.0.0.1:1", ft=_ft())
        try:
            assert await c._router_remove_worker("http://127.0.0.1:2") is False
            assert await c._router_add_worker("http://127.0.0.1:2") is False
        finally:
            await c.teardown()

    def test_add_worker_requires_the_backend_to_be_up(self, router_fleet):
        """``/add_worker`` health-checks the new backend before accepting it and
        blocks until it answers, so it cannot be used to pre-register a URL.

        Pinned because it constrains Part 2: a restarted engine must be healthy
        before it is re-admitted, and the admit call is not instantaneous."""
        router = router_fleet["router"]
        nothing_there = f"http://127.0.0.1:{get_open_port()}"
        with pytest.raises((httpx.TimeoutException, RuntimeError)):
            router.add_worker(nothing_there, timeout=3.0)
        assert nothing_there not in router.list_workers()

    @pytest.mark.asyncio
    async def test_reconcile_de_routes_the_dead_backend(self, router_fleet):
        """End to end: a registered backend dies, the probe finds it, and the router
        stops routing to it — without waiting out the router's own health interval
        (300s in this fixture, 60s x 3 failures in production)."""
        ctx = multiprocessing.get_context("spawn")
        port = get_open_port()
        victim_url = f"http://127.0.0.1:{port}"
        victim = ctx.Process(target=_router_backend_process, args=(port,), daemon=True)
        victim.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(f"{victim_url}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.RequestError:
                time.sleep(0.05)
        else:
            raise RuntimeError("victim backend failed to start")

        urls = router_fleet["urls"]
        router = router_fleet["router"]
        router.add_worker(victim_url)
        c = _client(urls[:2] + [victim_url], proxy=router_fleet["url"], ft=_ft(health_probe_timeout_s=1.0))
        try:
            assert victim_url in router.list_workers()
            victim.terminate()
            victim.join(timeout=10)

            await c._reconcile_fleet()
            assert c.active_server_urls == urls[:2]
            assert victim_url not in router.list_workers()
        finally:
            await c.teardown()
            router.remove_worker(victim_url)
            if victim.is_alive():
                victim.kill()
                victim.join(timeout=5)


# --------------------------------------------------------------------------
# Config gates
# --------------------------------------------------------------------------


class TestConfigGates:
    """FT is only sound on a topology whose failure unit is one engine and whose
    weight sync has no fixed communicator. These pin each refusal."""

    @staticmethod
    def _cfg(**ie_overrides):
        from skyrl.train.config.config import SkyRLTrainConfig

        cfg = SkyRLTrainConfig()
        ie = cfg.generator.inference_engine
        ie.weight_sync_backend = "sharded_rdt"
        ie.num_engines = 2
        ie.data_parallel_size = 1
        ie.run_engines_locally = True
        ie.enable_pd = False
        cfg.trainer.placement.colocate_all = False
        ie.fault_tolerance = InferenceFaultToleranceConfig(enabled=True)
        for k, v in ie_overrides.items():
            setattr(ie, k, v)
        return cfg

    @staticmethod
    def _validate(cfg):
        from skyrl.train.utils.utils import _validate_inference_fault_tolerance_cfg

        _validate_inference_fault_tolerance_cfg(cfg)

    def test_the_supported_topology_passes(self):
        self._validate(self._cfg())

    def test_disabled_is_never_validated(self):
        """Every other setting is irrelevant when the switch is off — an FT block
        left in a config must not break an unrelated run."""
        cfg = self._cfg(weight_sync_backend="nccl", num_engines=1, data_parallel_size=4)
        cfg.generator.inference_engine.fault_tolerance.enabled = False
        self._validate(cfg)

    @pytest.mark.parametrize("backend", ["nccl", "cuda_ipc"])
    def test_non_rdt_backends_are_rejected(self, backend):
        with pytest.raises(ValueError, match="sharded_rdt"):
            self._validate(self._cfg(weight_sync_backend=backend))

    def test_colocation_is_rejected(self):
        cfg = self._cfg()
        cfg.trainer.placement.colocate_all = True
        with pytest.raises(ValueError, match="colocate_all"):
            self._validate(cfg)

    def test_data_parallel_is_rejected(self):
        with pytest.raises(ValueError, match="data_parallel_size"):
            self._validate(self._cfg(data_parallel_size=2))

    def test_pd_disaggregation_is_rejected(self):
        with pytest.raises(ValueError, match="enable_pd"):
            self._validate(self._cfg(enable_pd=True))

    def test_external_engines_are_rejected(self):
        with pytest.raises(ValueError, match="run_engines_locally"):
            self._validate(self._cfg(run_engines_locally=False))

    def test_a_single_engine_is_rejected(self):
        with pytest.raises(ValueError, match="num_engines"):
            self._validate(self._cfg(num_engines=1))

    def test_min_live_engines_must_fit_the_fleet(self):
        cfg = self._cfg(num_engines=2)
        cfg.generator.inference_engine.fault_tolerance.min_live_engines = 3
        with pytest.raises(ValueError, match="min_live_engines"):
            self._validate(cfg)

    @pytest.mark.parametrize("field,value", [("health_probe_timeout_s", 0.0), ("stall_timeout_s", -1.0)])
    def test_non_positive_timeouts_are_rejected(self, field, value):
        cfg = self._cfg()
        setattr(cfg.generator.inference_engine.fault_tolerance, field, value)
        with pytest.raises(ValueError, match=field):
            self._validate(cfg)

    def test_request_timeout_may_be_none(self):
        cfg = self._cfg()
        cfg.generator.inference_engine.fault_tolerance.request_timeout_s = None
        self._validate(cfg)

    def test_the_block_survives_a_hydra_round_trip(self):
        """Configs reach the trainer through ``from_dict_config``, so a nested
        dataclass that does not rebuild would silently come back as a dict and
        every ``ft.enabled`` read would be wrong."""
        from omegaconf import OmegaConf

        from skyrl.train.config.config import SkyRLTrainConfig

        raw = OmegaConf.structured(SkyRLTrainConfig())
        raw.generator.inference_engine.fault_tolerance.enabled = True
        raw.generator.inference_engine.fault_tolerance.min_live_engines = 2
        raw.generator.inference_engine.fault_tolerance.request_timeout_s = None

        cfg = SkyRLTrainConfig.from_dict_config(raw)
        ft = cfg.generator.inference_engine.fault_tolerance
        assert isinstance(ft, InferenceFaultToleranceConfig)
        assert ft.enabled is True
        assert ft.min_live_engines == 2
        assert ft.request_timeout_s is None
        assert ft.stall_timeout_s == 300.0  # untouched default survives
