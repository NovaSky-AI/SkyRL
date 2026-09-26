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
A server that can't be reached, or answers 5xx, is skipped for that create.
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

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        #: The HTTP status, when the server answered.
        self.status = status


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
        #: What the last ``finish`` sent, so a failed one can be sent again unchanged.
        self.finishing: dict[str, Any] | None = None

    async def finish(self, annotations: dict[str, Any] | None = None) -> FinishResult:
        """Seal the trajectory and get its samples. Safe to call more than once."""
        self.finishing = annotations or {}
        body = await self._pool._post(f"{self.server}/trajectories/{self.id}/finish", {"annotations": self.finishing})
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
        self._closed = False

    async def _http(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("the CapturePool is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        self._closed = True
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
            except CaptureError as error:
                if error.status is None or error.status < 500:
                    raise
                errors.append(str(error))
                continue
            return Trajectory(self, server, body["id"], body["base_url"])
        raise CaptureError(f"no capture server reachable: {'; '.join(errors)}")

    @asynccontextmanager
    async def trajectory(self, meta: dict[str, Any] | None = None) -> AsyncIterator[Trajectory]:
        """A trajectory that is always finished.

        If the block raised before finishing, the trajectory is finished with
        ``{"error": ...}``. If its own ``finish`` failed, that finish is sent
        again, so the caller's annotations (a reward) aren't replaced.
        """
        trajectory = await self.create(meta)
        try:
            yield trajectory
        except BaseException as error:
            if trajectory.result is None:
                annotations = trajectory.finishing
                if annotations is None:
                    annotations = {"error": type(error).__name__}
                try:
                    await trajectory.finish(annotations)
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
    if response.status != 200:
        # An error body may not be JSON (a proxy's HTML page, aiohttp's plain 404).
        detail = (await response.text(errors="replace"))[:2000]
        raise CaptureError(f"{response.method} {response.url}: HTTP {response.status}: {detail}", response.status)
    return await response.json(content_type=None)
