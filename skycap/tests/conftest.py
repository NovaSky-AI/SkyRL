from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import aiohttp
import openai
import pytest
from aiohttp.test_utils import TestServer

from skycap.server import CaptureServer
from skycap.text import TextBackend
from tests.mock_openai import API_KEY, MockOpenAI


@dataclass
class Stack:
    upstream: MockOpenAI
    server: CaptureServer
    url: str
    http: aiohttp.ClientSession

    async def create(self, meta: dict | None = None) -> dict:
        async with self.http.post(f"{self.url}/trajectories", json={"meta": meta or {}}) as response:
            assert response.status == 200
            return await response.json()

    async def finish(self, trajectory_id: str, annotations: dict | None = None, *, expect: int = 200) -> dict:
        async with self.http.post(
            f"{self.url}/trajectories/{trajectory_id}/finish", json={"annotations": annotations or {}}
        ) as response:
            assert response.status == expect
            return await response.json()

    async def document(self, trajectory_id: str) -> dict:
        async with self.http.get(f"{self.url}/trajectories/{trajectory_id}") as response:
            assert response.status == 200
            return await response.json()


_openai_clients: list[openai.AsyncOpenAI] = []


def openai_client(base_url: str, *, api_key: str = "k", **options: Any) -> openai.AsyncOpenAI:
    """An OpenAI client that is closed when the test ends.

    Left to the garbage collector, an unclosed client's finalizer can close a
    socket descriptor a newer connection has already reused, which shows up
    much later as a connect timeout in an unrelated test.
    """
    options.setdefault("max_retries", 0)
    llm = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, **options)
    _openai_clients.append(llm)
    return llm


@pytest.fixture(autouse=True)
async def _close_openai_clients() -> AsyncIterator[None]:
    yield
    while _openai_clients:
        await _openai_clients.pop().close()


@pytest.fixture
async def stack() -> AsyncIterator[Stack]:
    upstream = MockOpenAI()
    upstream_server = TestServer(upstream.app())
    await upstream_server.start_server()
    server = CaptureServer(TextBackend(str(upstream_server.make_url("/v1")), api_key=API_KEY))
    capture_server = TestServer(server.app())
    await capture_server.start_server()
    async with aiohttp.ClientSession() as http:
        yield Stack(upstream, server, str(capture_server.make_url("")).rstrip("/"), http)
    await capture_server.close()
    await upstream_server.close()
