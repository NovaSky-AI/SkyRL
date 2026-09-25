"""The client pool: round-robin creates, the URL as the routing, and always finishing."""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skycap import CaptureError, CapturePool, Sample
from skycap.server import CaptureServer
from skycap.text import TextBackend
from tests.conftest import openai_client
from tests.mock_openai import API_KEY, MockOpenAI
from tests.test_tokens import client, converse, token_stack


@pytest.fixture
async def servers() -> AsyncIterator[tuple[list[str], list[CaptureServer]]]:
    async with AsyncExitStack() as stack:
        upstream = TestServer(MockOpenAI().app())
        await upstream.start_server()
        stack.push_async_callback(upstream.close)
        urls, captures = [], []
        for _ in range(3):
            capture = CaptureServer(TextBackend(str(upstream.make_url("/v1")), api_key=API_KEY))
            server = TestServer(capture.app())
            await server.start_server()
            stack.push_async_callback(server.close)
            urls.append(str(server.make_url("")).rstrip("/"))
            captures.append(capture)
        yield urls, captures


async def test_creates_round_robin_and_each_url_names_its_server(servers) -> None:
    urls, captures = servers
    async with CapturePool(urls) as pool:
        trajectories = [await pool.create({"i": i}) for i in range(6)]
        for trajectory in trajectories:
            await client(trajectory.base_url).chat.completions.create(
                model="policy", messages=[{"role": "user", "content": trajectory.id}]
            )

    assert Counter(t.server for t in trajectories) == Counter({url: 2 for url in urls})
    for trajectory in trajectories:
        assert trajectory.base_url.startswith(trajectory.server)
        owner = captures[urls.index(trajectory.server)]
        assert len(owner.trajectories[trajectory.id].graph) == 2


async def test_the_context_manager_finishes_and_returns_samples(servers) -> None:
    urls, _ = servers
    async with CapturePool(urls) as pool, pool.trajectory({"task": "t"}) as trajectory:
        await client(trajectory.base_url).chat.completions.create(
            model="policy", messages=[{"role": "user", "content": "q"}]
        )
        result = await trajectory.finish({"reward": 1.0})
        document = await trajectory.document()

    assert result.status == "finished"
    (sample,) = result.samples
    assert isinstance(sample, Sample)
    assert [m["content"] for m in sample.messages] == ["q", "re: q"]
    assert document["annotations"] == {"reward": 1.0}


async def test_a_block_that_raises_still_finishes_the_trajectory(servers) -> None:
    urls, _ = servers
    async with CapturePool(urls) as pool:
        with pytest.raises(RuntimeError):
            async with pool.trajectory() as trajectory:
                raise RuntimeError("harness crashed")
        document = await trajectory.document()

    assert document["status"] == "finished"
    assert document["annotations"] == {"error": "RuntimeError"}


async def test_a_block_that_forgets_to_finish_is_finished_on_exit(servers) -> None:
    urls, _ = servers
    async with CapturePool(urls) as pool:
        async with pool.trajectory() as trajectory:
            pass
        assert trajectory.result is not None and trajectory.result.status == "finished"


async def test_an_unreachable_server_is_skipped(servers) -> None:
    urls, _ = servers
    async with CapturePool(["http://127.0.0.1:1", urls[0]]) as pool:
        trajectories = [await pool.create() for _ in range(3)]
    assert {t.server for t in trajectories} == {urls[0]}


async def test_a_server_answering_5xx_is_skipped(servers) -> None:
    urls, captures = servers

    async def unavailable(request: web.Request) -> web.Response:
        return web.Response(text="<html>503 Service Unavailable</html>", status=503, content_type="text/html")

    broken = web.Application()
    broken.router.add_post("/trajectories", unavailable)
    server = TestServer(broken)
    await server.start_server()
    try:
        async with CapturePool([str(server.make_url("")).rstrip("/"), urls[0]]) as pool:
            trajectories = [await pool.create() for _ in range(3)]
    finally:
        await server.close()
    assert {t.server for t in trajectories} == {urls[0]}


async def test_a_4xx_on_create_is_raised(servers) -> None:
    urls, _ = servers
    async with CapturePool([f"{urls[0]}/nowhere"]) as pool:
        with pytest.raises(CaptureError) as raised:
            await pool.create()
    assert raised.value.status == 404


async def test_a_failed_finish_is_sent_again_with_its_annotations(servers) -> None:
    urls, captures = servers
    async with CapturePool(urls) as pool:
        with pytest.raises(CaptureError):
            async with pool.trajectory() as trajectory:
                trajectory.id, real = "tr_missing", trajectory.id
                try:
                    await trajectory.finish({"reward": 1.0})
                finally:
                    trajectory.id = real
        document = await trajectory.document()

    assert document["status"] == "finished"
    assert document["annotations"] == {"reward": 1.0}


async def test_a_closed_pool_refuses_calls(servers) -> None:
    urls, _ = servers
    async with CapturePool(urls) as pool:
        trajectory = await pool.create()
    with pytest.raises(RuntimeError):
        await trajectory.document()


async def test_no_reachable_server_is_an_error() -> None:
    async with CapturePool(["http://127.0.0.1:1"]) as pool:
        with pytest.raises(CaptureError):
            await pool.create()


async def test_token_samples_arrive_decoded() -> None:
    async with token_stack(sampling_mask=True) as stack, CapturePool([stack.url]) as pool:
        async with pool.trajectory() as trajectory:
            await converse(client(trajectory.base_url), "hi", "more")
            result = await trajectory.finish()

    (sample,) = result.samples
    assert isinstance(sample.routed_experts, np.ndarray)
    assert sample.routed_experts.shape == (len(sample.input_ids), 2, 2)
    assert sample.sampling_mask is not None and len(sample.sampling_mask) == len(sample.input_ids)


def test_a_pool_needs_servers() -> None:
    with pytest.raises(ValueError):
        CapturePool([])


async def test_the_openai_client_is_all_a_caller_needs(servers) -> None:
    """No harness: an ordinary OpenAI client pointed at the trajectory is the interface."""
    urls, _ = servers
    async with CapturePool(urls) as pool, pool.trajectory() as trajectory:
        llm = openai_client(trajectory.base_url, api_key="anything")
        reply = await llm.chat.completions.create(model="policy", messages=[{"role": "user", "content": "x"}])
        assert reply.choices[0].message.content == "re: x"
