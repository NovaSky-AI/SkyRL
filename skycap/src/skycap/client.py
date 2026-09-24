"""The client side: a round-robin pool over capture servers, and a handle per trajectory.

    pool = CapturePool(["http://capture-0:8080", "http://capture-1:8080"])
    async with pool.trajectory({"task": "t1", "step": 3}) as trajectory:
        run_harness(base_url=trajectory.base_url)       # an unchanged OpenAI client
        result = await trajectory.finish({"reward": 1.0})
    result.status, result.samples

Each trajectory lives on one server, and its ``base_url`` names that server,
so no router or load balancer is involved: the URL is the routing. Picking a
server is plain round-robin from a random starting point, which keeps several
independent pools (one per generator process) balanced without coordinating.
A server that can't be reached is skipped for that create.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp

from skycap.samples import Sample


class CaptureError(Exception):
    """A capture server refused or failed a control-plane call."""


@dataclass(slots=True)
class FinishResult:
    id: str
    status: str
    samples: list[Sample]


class Trajectory:
    def __init__(self, pool: CapturePool, server: str, trajectory_id: str, base_url: str) -> None:
        self._pool = pool
        self.server = server
        self.id = trajectory_id
        #: Point the harness's OpenAI client here.
        self.base_url = base_url
        self.result: FinishResult | None = None

    async def finish(self, annotations: dict[str, Any] | None = None) -> FinishResult:
        """Seal the trajectory and get its samples. Safe to call more than once."""
        body = await self._pool._post(
            f"{self.server}/trajectories/{self.id}/finish", {"annotations": annotations or {}}
        )
        self.result = FinishResult(
            id=body["id"], status=body["status"], samples=[Sample.from_json(s) for s in body["samples"]]
        )
        return self.result

    async def document(self) -> dict[str, Any]:
        return await self._pool._get(f"{self.server}/trajectories/{self.id}")

    def __repr__(self) -> str:
        return f"Trajectory({self.id!r}, base_url={self.base_url!r})"


class CapturePool:
    def __init__(
        self,
        urls: Sequence[str],
        *,
        session: aiohttp.ClientSession | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not urls:
            raise ValueError("a pool needs at least one capture server")
        self.urls = [url.rstrip("/") for url in urls]
        start = random.randrange(len(self.urls))
        self._next = itertools.cycle(self.urls[start:] + self.urls[:start])
        self._session = session
        self._owns_session = session is None
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> CapturePool:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def create(self, meta: dict[str, Any] | None = None) -> Trajectory:
        """A new trajectory on the next reachable server."""
        errors = []
        for _ in range(len(self.urls)):
            server = next(self._next)
            try:
                body = await self._post(f"{server}/trajectories", {"meta": meta or {}})
            except (aiohttp.ClientConnectionError, TimeoutError) as error:
                errors.append(f"{server}: {error}")
                continue
            return Trajectory(self, server, body["id"], body["base_url"])
        raise CaptureError(f"no capture server reachable: {'; '.join(errors)}")

    @asynccontextmanager
    async def trajectory(self, meta: dict[str, Any] | None = None) -> AsyncIterator[Trajectory]:
        """A trajectory that is always finished: with ``{"error": ...}`` if the block raised."""
        trajectory = await self.create(meta)
        try:
            yield trajectory
        except BaseException as error:
            if trajectory.result is None:
                try:
                    await trajectory.finish({"error": type(error).__name__})
                except Exception:  # noqa: BLE001 - the block's own error is the one to raise
                    pass
            raise
        if trajectory.result is None:
            await trajectory.finish()

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        http = await self._http()
        async with http.post(url, json=payload) as response:
            return await _body(response)

    async def _get(self, url: str) -> dict[str, Any]:
        http = await self._http()
        async with http.get(url) as response:
            return await _body(response)


async def _body(response: aiohttp.ClientResponse) -> dict[str, Any]:
    body = await response.json(content_type=None)
    if response.status != 200:
        raise CaptureError(f"{response.method} {response.url}: HTTP {response.status}: {body}")
    return body
