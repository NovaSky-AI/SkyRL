"""CaptureService: a server in this process, built from options, on its own thread."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from skycap import CapturePool, CaptureService, record
from skycap.service import build_backend
from tests.fake_renderer import FakeRenderer
from tests.mock_engine import MockEngine
from tests.mock_openai import API_KEY, MockOpenAI
from tests.test_tokens import client, converse


async def test_a_service_serves_a_trajectory_and_writes_it_when_stopped(tmp_path: Path) -> None:
    upstream = TestServer(MockOpenAI().app())
    await upstream.start_server()
    service = CaptureService(str(upstream.make_url("/v1")), api_key=API_KEY, record_dir=str(tmp_path), host="127.0.0.1")
    url = await asyncio.to_thread(service.start)
    try:
        async with CapturePool([url]) as pool:
            trajectory = await pool.create({"task": "t"})
            await converse(client(trajectory.base_url), "q")
    finally:
        # Left open on purpose: stopping writes it.
        assert await asyncio.to_thread(service.stop)
        await upstream.close()
    (trajectory_id,) = record.list_ids(tmp_path)
    assert trajectory_id == trajectory.id
    assert len(record.load(tmp_path, trajectory_id).graph) == 2


async def test_the_harness_listener_serves_the_harness_routes_alone(tmp_path: Path) -> None:
    upstream = TestServer(MockOpenAI().app())
    await upstream.start_server()
    service = CaptureService(
        str(upstream.make_url("/v1")),
        api_key=API_KEY,
        record_dir=str(tmp_path),
        host="127.0.0.1",
        harness_host="0.0.0.0",
    )
    url = await asyncio.to_thread(service.start)
    try:
        # Bound on every interface, and reached from this node on loopback.
        assert service.harness_url is not None and service.harness_url.startswith("http://127.0.0.1:")
        assert service.harness_url != url
        async with CapturePool([url]) as pool, aiohttp.ClientSession() as http:
            trajectory = await pool.create({"task": "t"})
            # The harness calls the same trajectory on the harness listener ...
            path = trajectory.base_url.removeprefix(url)
            await converse(client(service.harness_url + path), "q", "r")
            # ... where nothing of the control plane is routed.
            for method, route in [
                ("POST", "/trajectories"),
                ("GET", f"/trajectories/{trajectory.id}"),
                ("GET", "/healthz"),
            ]:
                async with http.request(method, service.harness_url + route) as response:
                    assert response.status == 404, route
            result = await trajectory.finish({"reward": 1.0})
    finally:
        assert await asyncio.to_thread(service.stop)
        await upstream.close()
    assert result.status == "finished" and len(record.load(tmp_path, trajectory.id).graph) == 4
    # It stopped with the server.
    async with aiohttp.ClientSession() as http:
        with pytest.raises(aiohttp.ClientConnectionError):
            await http.get(service.harness_url + "/healthz")


async def test_token_mode_is_built_from_options(tmp_path: Path) -> None:
    engine = TestServer(MockEngine().app())
    await engine.start_server()
    service = CaptureService(
        str(engine.make_url("")).rstrip("/"),
        mode="tokens",
        renderer=FakeRenderer(),
        use_raw_content=True,
        record_dir=str(tmp_path),
        host="127.0.0.1",
    )
    url = await asyncio.to_thread(service.start)
    try:
        async with CapturePool([url]) as pool, pool.trajectory({}) as trajectory:
            await converse(client(trajectory.base_url), "q", "r")
            result = await trajectory.finish({"reward": 1.0})
    finally:
        await asyncio.to_thread(service.stop)
        await engine.close()
    assert result.status == "finished" and len(result.samples) == 1
    assert service.server.backend.describe()["use_raw_content"] is True


def test_bad_options_are_refused_before_anything_starts() -> None:
    with pytest.raises(ValueError, match="mode"):
        build_backend("http://x", mode="bytes")
    with pytest.raises(ValueError, match="exactly one of tokenizer and renderer"):
        build_backend("http://x", mode="tokens")
    with pytest.raises(ValueError, match="exactly one of tokenizer and renderer"):
        build_backend("http://x", mode="tokens", tokenizer="t", renderer=FakeRenderer())
    with pytest.raises(TypeError):
        CaptureService("http://x", unknown_option=1)
