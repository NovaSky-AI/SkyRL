"""The client side: a round-robin pool over capture servers, and a handle per trajectory.

    pool = CapturePool(["http://capture-0:8080", "http://capture-1:8080"])
    async with pool.trajectory({"task": "t1", "step": 3}) as trajectory:
        run_harness(base_url=trajectory.base_url, api_key=trajectory.api_key)  # an unchanged OpenAI client
        result = await trajectory.finish({"reward": 1.0})
    result.status, result.samples

Each trajectory lives on one server, and its ``base_url`` names that server,
so no router or load balancer is involved: the URL is the routing. Picking a
server is plain round-robin from a random starting point, which keeps several
independent pools (one per generator process) balanced without coordinating.
A server that can't be reached, or answers 5xx, is skipped for that create.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import random
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp

from skycap.paths import PATH_RULE_FAILED
from skycap.samples import Sample


class CaptureError(Exception):
    """A capture server refused or failed a control-plane call."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        #: The HTTP status, when the server answered.
        self.status = status


class PathRuleError(CaptureError):
    """``finish``'s path rule raised on the server. The trajectory is ended, without samples."""


@dataclass(frozen=True, slots=True)
class RecordLocation:
    """Where a finished trajectory's record is. See ``docs/format.md``."""

    #: The document's path on the capture server's own disk, in its ``record_dir``.
    path: str
    #: The document's URI in the server's record mirror, or None without one. The copy is made in the
    #: background and fails open, so it may not be there yet, or at all.
    mirror: str | None = None
    #: The record's file names, sidecars then the document (e.g. ``tr_ab12.tokens.zst``, ``tr_ab12.json.zst``):
    #: as the mirror holds them when there is one (so without the kinds its ``exclude`` leaves out), else as
    #: the record directory does. They sit beside the document, in the mirror and on disk.
    files: tuple[str, ...] = ()

    @property
    def uri(self) -> str:
        """The mirror URI when there is one, else the local path."""
        return self.mirror or self.path

    @classmethod
    def from_json(cls, body: dict[str, Any] | None) -> RecordLocation | None:
        if body is None:
            return None
        return cls(path=body["path"], mirror=body.get("mirror"), files=tuple(body.get("files", ())))


@dataclass(slots=True)
class FinishResult:
    id: str
    status: str
    samples: list[Sample]
    #: Token-mode calls whose prompt had to be rendered rather than extended (``CallInfo.bridged``).
    #: Zero for a harness that keeps its history append-only.
    unbridged_calls: int = 0
    #: Where the record is, or None when the server has no ``record_dir`` (or couldn't write it).
    record: RecordLocation | None = None


class Trajectory:
    def __init__(
        self,
        pool: CapturePool,
        server: str,
        trajectory_id: str,
        base_url: str,
        paths: str = "all",
        exposed_base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._pool = pool
        self.server = server
        self.id = trajectory_id
        #: Point the harness's OpenAI client here.
        self.base_url = base_url
        #: Or here, for a harness outside this network, when the server is exposed (``skycap.exposure``).
        self.exposed_base_url = exposed_base_url
        #: And give it this as its ``api_key``: a server started with ``require_api_key`` answers only it.
        self.api_key = api_key
        self.result: FinishResult | None = None
        #: What the last ``finish`` sent, so a failed one can be sent again unchanged.
        self.finishing: dict[str, Any] | None = None
        #: The path rule ``finish`` uses when it names none, including the one a failed block is finished with.
        self.paths = paths

    async def finish(self, annotations: dict[str, Any] | None = None, *, paths: str | None = None) -> FinishResult:
        """Seal the trajectory and get its samples. Safe to call more than once.

        ``paths`` names the path rule that picks the samples (``skycap.paths``): ``all``, a sample per
        root-to-leaf path; ``final``, only the path to the last model call's reply; or a custom rule the
        server has. It defaults to the trajectory's ``paths``. Raises ``PathRuleError`` if the rule raised.
        """
        self.finishing = annotations or {}
        if paths is not None:
            self.paths = paths
        body = await self._pool._post(
            f"{self.server}/trajectories/{self.id}/finish", {"annotations": self.finishing, "paths": self.paths}
        )
        self.result = FinishResult(
            id=body["id"],
            status=body["status"],
            samples=[Sample.from_json(s) for s in body["samples"]],
            unbridged_calls=body.get("unbridged_calls", 0),
            record=RecordLocation.from_json(body.get("record")),
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
        timeout: float = 60.0,
    ) -> None:
        if not urls:
            raise ValueError("a pool needs at least one capture server")
        self.urls = [url.rstrip("/") for url in urls]
        start = random.randrange(len(self.urls))
        self._next = itertools.cycle(self.urls[start:] + self.urls[:start])
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._closed = False

    async def _http(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("the CapturePool is closed")
        if self._session is None:
            self._session, self._session_loop = aiohttp.ClientSession(timeout=self._timeout), asyncio.get_running_loop()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session, also once the loop it was used on has ended.

        A session can only be awaited on its own loop. When that loop is gone, so
        are its connections: the session is detached and its connector marked
        closed instead, which keeps aiohttp from reporting them as leaked.
        """
        self._closed = True
        session, loop = self._session, self._session_loop
        self._session = self._session_loop = None
        if session is None:
            return
        if loop is asyncio.get_running_loop():
            await session.close()
            return
        connector = session.connector
        session.detach()
        if connector is not None:
            with contextlib.suppress(Exception):
                connector._close()  # the synchronous close aiohttp itself uses on a dead loop

    async def __aenter__(self) -> CapturePool:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def create(self, meta: dict[str, Any] | None = None, *, paths: str = "all") -> Trajectory:
        """A new trajectory on the next reachable server, to be finished with the path rule ``paths``."""
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
            return Trajectory(
                self,
                server=server,
                trajectory_id=body["id"],
                base_url=body["base_url"],
                paths=paths,
                exposed_base_url=body.get("exposed_base_url"),
                api_key=body.get("api_key"),
            )
        raise CaptureError(f"no capture server reachable: {'; '.join(errors)}")

    @asynccontextmanager
    async def trajectory(self, meta: dict[str, Any] | None = None, *, paths: str = "all") -> AsyncIterator[Trajectory]:
        """A trajectory that is always finished, with the path rule ``paths`` unless its ``finish`` names another.

        If the block raised before finishing, the trajectory is finished with
        ``{"error": ...}``. If its own ``finish`` failed, that finish is sent
        again, so the caller's annotations (a reward) aren't replaced.
        """
        trajectory = await self.create(meta, paths=paths)
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
        message = f"{response.method} {response.url}: HTTP {response.status}: {detail}"
        if _code(detail) == PATH_RULE_FAILED:
            raise PathRuleError(message, response.status)
        raise CaptureError(message, response.status)
    return await response.json(content_type=None)


def _code(detail: str) -> str | None:
    """The ``code`` of a JSON error body, if it has one."""
    try:
        body = json.loads(detail)
    except ValueError:
        return None
    return body.get("code") if isinstance(body, dict) else None
