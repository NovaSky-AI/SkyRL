"""The capture server: a control plane plus one OpenAI-compatible route per trajectory.

    POST /trajectories                    {meta}         -> {id, base_url}
    POST /trajectories/{id}/finish        {annotations}  -> {status, samples}
    GET  /trajectories/{id}                               -> the trajectory document
    POST /t/{id}/v1/chat/completions                      (the harness)
    GET  /t/{id}/v1/models                                (passed through)
    GET  /healthz

A trajectory's route is its URL: the harness needs no header and no SDK patch.
``finish`` seals the trajectory: in-flight calls are cancelled and never
committed, so its samples are final the moment they are returned.

With a ``record_dir``, a trajectory is written when it ends -- by ``finish``,
by the idle TTL (as ``abandoned``), or by a graceful shutdown (as ``open``) --
and then dropped from memory; reads of it are served from disk. Without one,
ended trajectories stay in memory, which is only for tests and development.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import orjson
from aiohttp import web

from skycap import record, retry
from skycap.openai_chat import ChatRequest, RequestError, error_body, parse_request
from skycap.samples import build_samples
from skycap.trajectory import Status, Trajectory, new_trajectory_id

logger = logging.getLogger(__name__)


class Backend(Protocol):
    """How a call reaches a model. One per server process: text or tokens."""

    def describe(self) -> dict[str, Any]: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def release(self, trajectory: Trajectory) -> None: ...
    async def finalize(self, trajectory: Trajectory) -> None: ...
    async def models(self, request: web.Request) -> web.Response: ...
    async def chat(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> web.StreamResponse: ...


def _json(payload: Any, status: int = 200) -> web.Response:
    return web.Response(body=orjson.dumps(payload), status=status, content_type="application/json")


def _openai_error(message: str, status: int, *, code: str | None = None) -> web.Response:
    return web.Response(body=error_body(message, code=code), status=status, content_type="application/json")


class CaptureServer:
    def __init__(
        self,
        backend: Backend,
        *,
        record_dir: str | Path | None = None,
        ttl: float = 3600.0,
        sweep_interval: float = 60.0,
    ) -> None:
        self.backend = backend
        self.record_dir = Path(record_dir) if record_dir is not None else None
        #: Seconds an open trajectory may go without a request before it is abandoned.
        self.ttl = ttl
        self.sweep_interval = sweep_interval
        self.trajectories: dict[str, Trajectory] = {}
        self._sweeper: asyncio.Task[None] | None = None

    def app(self) -> web.Application:
        app = web.Application(client_max_size=1024**3)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_post("/trajectories", self.create)
        app.router.add_post("/trajectories/{id}/finish", self.finish)
        app.router.add_get("/trajectories/{id}", self.get)
        app.router.add_post("/t/{id}/v1/chat/completions", self.chat)
        app.router.add_get("/t/{id}/v1/models", self.models)
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        return app

    async def _on_startup(self, app: web.Application) -> None:
        await self.backend.start()
        # Without a record, ending a trajectory would neither record it nor free it, so nothing expires.
        if self.record_dir is not None:
            self._sweeper = asyncio.create_task(self._sweep_forever())

    async def _on_cleanup(self, app: web.Application) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
        # Graceful shutdown: everything still in memory is unwritten, and is written as it stands.
        # Sessions of trajectories that never ended are released: nothing here will continue them.
        for trajectory in list(self.trajectories.values()):
            await self._persist(trajectory)
            if not trajectory.ended:
                try:
                    await self.backend.release(trajectory)
                except Exception:
                    logger.exception("releasing %s failed", trajectory.id)
        await self.backend.close()

    # -- ending a trajectory -----------------------------------------------------
    async def end(self, trajectory: Trajectory, status: Status, annotations: dict[str, Any] | None = None) -> None:
        """Seal, release the upstream session, write, and drop from memory.

        A trajectory that already failed keeps its status; ``annotations`` are
        still recorded on it. It is written when this call ended it (including
        one read back from a shutdown-time record) or when it is still in
        memory, which means an earlier write failed and is retried.
        """
        ended_now = not trajectory.ended
        if trajectory.is_open:
            trajectory.seal(status, annotations)
        elif not trajectory.ended:
            trajectory.annotations.update(annotations or {})
        if not trajectory.ended:
            trajectory.ended = True
            try:
                await self.backend.release(trajectory)
            except Exception:
                logger.exception("releasing %s failed", trajectory.id)
        unwritten = self.trajectories.get(trajectory.id) is trajectory
        if (ended_now or unwritten) and await self._persist(trajectory):
            self.trajectories.pop(trajectory.id, None)

    async def _persist(self, trajectory: Trajectory) -> bool:
        """Best-effort write. Returns whether the trajectory is on disk."""
        if self.record_dir is None:
            return False
        try:
            await self.backend.finalize(trajectory)
        except Exception:
            logger.exception("finalizing %s failed; writing it without token text", trajectory.id)
        try:
            await asyncio.to_thread(record.write, self.record_dir, trajectory)
        except Exception:
            logger.exception("writing %s failed", trajectory.id)
            return False
        return True

    async def sweep(self) -> list[str]:
        """End trajectories nobody finished within the TTL; open ones as abandoned. Returns their ids."""
        cutoff = time.monotonic() - self.ttl

        def idle(trajectory: Trajectory) -> bool:
            return not trajectory.ended and not trajectory.inflight and trajectory.last_active < cutoff

        abandoned = []
        for trajectory in [t for t in self.trajectories.values() if idle(t)]:
            # Checked again: a request may have arrived while an earlier one was being written.
            if not idle(trajectory):
                continue
            logger.info("abandoning %s after %.0fs idle", trajectory.id, self.ttl)
            await self.end(trajectory, "abandoned")
            abandoned.append(trajectory.id)
        return abandoned

    async def _sweep_forever(self) -> None:
        while True:
            await asyncio.sleep(self.sweep_interval)
            try:
                await self.sweep()
            except Exception:
                logger.exception("sweep failed")

    async def _lookup(self, trajectory_id: str) -> Trajectory | None:
        """A live trajectory, or an ended one read back from disk."""
        trajectory = self.trajectories.get(trajectory_id)
        if trajectory is not None or self.record_dir is None:
            return trajectory
        try:
            return await asyncio.to_thread(record.load, self.record_dir, trajectory_id)
        except FileNotFoundError:
            return None

    def _ended_or_unknown(self, trajectory_id: str) -> web.Response:
        """The answer to a harness call for a trajectory not in memory: 410 if it was recorded, else 404."""
        if self.record_dir is not None and record.document_path(self.record_dir, trajectory_id).exists():
            return _openai_error("trajectory has ended", 410, code="trajectory_closed")
        return _openai_error("unknown trajectory", 404)

    # -- control plane --------------------------------------------------------
    async def healthz(self, request: web.Request) -> web.Response:
        open_count = sum(1 for t in self.trajectories.values() if t.is_open)
        return _json({"ok": True, "open_trajectories": open_count, "capture": self.backend.describe()})

    async def create(self, request: web.Request) -> web.Response:
        body = await _read_json(request, default={})
        meta = body.get("meta") if isinstance(body, dict) else None
        if meta is not None and not isinstance(meta, dict):
            return _json({"error": "`meta` must be an object"}, 400)
        trajectory = Trajectory(id=new_trajectory_id(), meta=meta or {}, capture=self.backend.describe())
        self.trajectories[trajectory.id] = trajectory
        # The route is on the host the pool reached us at: harnesses reach it the same way.
        base = f"{request.scheme}://{request.host}"
        return _json({"id": trajectory.id, "base_url": f"{base}/t/{trajectory.id}/v1"})

    async def finish(self, request: web.Request) -> web.Response:
        body = await _read_json(request, default={})
        annotations = body.get("annotations") if isinstance(body, dict) else None
        if annotations is not None and not isinstance(annotations, dict):
            return _json({"error": "`annotations` must be an object"}, 400)
        trajectory = await self._lookup(request.match_info["id"])
        if trajectory is None:
            return _json({"error": "unknown trajectory"}, 404)
        if trajectory.ended and _changes(trajectory.annotations, annotations):
            # A repeat is answered as the first finish was. New annotations on it would be
            # silently lost, so they are refused instead.
            return _json({"error": "trajectory already finished; its annotations can't change"}, 409)
        await self.end(trajectory, "finished", annotations)
        return _json(
            {
                "id": trajectory.id,
                "status": trajectory.status,
                "samples": [s.to_json() for s in build_samples(trajectory.graph)],
            }
        )

    async def get(self, request: web.Request) -> web.Response:
        trajectory_id = request.match_info["id"]
        trajectory = self.trajectories.get(trajectory_id)
        if trajectory is not None:
            return _json(trajectory.document())
        if self.record_dir is not None:
            try:
                # The stored document, as written: no sidecar is opened.
                return _json(await asyncio.to_thread(record.read_document, self.record_dir, trajectory_id))
            except FileNotFoundError:
                pass
        return _json({"error": "unknown trajectory"}, 404)

    # -- data plane -------------------------------------------------------------
    async def models(self, request: web.Request) -> web.Response:
        trajectory_id = request.match_info["id"]
        if trajectory_id not in self.trajectories:
            return self._ended_or_unknown(trajectory_id)
        return await self.backend.models(request)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        trajectory_id = request.match_info["id"]
        trajectory = self.trajectories.get(trajectory_id)
        if trajectory is None:
            return self._ended_or_unknown(trajectory_id)
        if not trajectory.is_open:
            return _openai_error(f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
        # In flight from here on, so a finish that lands during the body read cancels this call.
        task = asyncio.current_task()
        assert task is not None
        trajectory.inflight.add(task)
        trajectory.touch()
        try:
            raw = await request.read()
            try:
                chat = parse_request(orjson.loads(raw))
            except (orjson.JSONDecodeError, RequestError) as error:
                return _openai_error(str(error), 400)
            return await self._call(trajectory, request, chat, raw)
        finally:
            trajectory.inflight.discard(task)
            trajectory.touch()

    async def _call(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> web.StreamResponse:
        """One call, or the replay of the call it retries (see ``skycap.retry``)."""
        cache = trajectory.replay
        digest = retry.body_digest(raw)
        explicit = request.headers.get(retry.IDEMPOTENCY_KEY_HEADER)
        key = f"key:{explicit}" if explicit else f"body:{digest}"
        previous = cache.get(key) if explicit or retry.is_retry(request.headers) else None
        if previous is not None:
            if previous.digest != digest:
                return _openai_error("Idempotency-Key was reused with a different request", 400)
            if not previous.future.done():
                cache.coalesced += 1
            reply = await asyncio.shield(previous.future)
            if reply is not None:
                cache.replayed += 1
                return web.Response(body=reply.body, status=reply.status, headers=reply.headers)
        entry = cache.start(key, digest)
        # The call runs in its own task, so a client that disconnects (an SDK
        # timing out, about to retry) doesn't cancel it: it finishes and leaves
        # its reply for the retry. `finish` still cancels it, as in-flight work.
        work = asyncio.ensure_future(self.backend.chat(trajectory, request, chat, raw))
        trajectory.inflight.add(work)

        def settle(task: asyncio.Future[web.StreamResponse]) -> None:
            trajectory.inflight.discard(task)  # type: ignore[arg-type]
            cache.complete(key, entry, _replayable(task))

        work.add_done_callback(settle)
        return await asyncio.shield(work)


def _replayable(task: asyncio.Future[web.StreamResponse]) -> retry.Replay | None:
    """The finished call's reply, if a retry may be answered with it."""
    if task.cancelled() or task.exception() is not None:
        return None
    response = task.result()
    if type(response) is not web.Response or response.status != 200 or not isinstance(response.body, bytes):
        return None
    headers = {k: v for k, v in response.headers.items() if k.lower() not in {"content-length", "date", "server"}}
    return retry.Replay(body=response.body, status=response.status, headers=headers)


def _changes(current: dict[str, Any], update: dict[str, Any] | None) -> bool:
    """Whether ``update`` would change any annotation already recorded."""
    return any(key not in current or current[key] != value for key, value in (update or {}).items())


async def _read_json(request: web.Request, *, default: Any) -> Any:
    """The request's JSON body, ``default`` when it is empty, or a 400 when it doesn't parse."""
    raw = await request.read()
    if not raw:
        return default
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as error:
        raise web.HTTPBadRequest(
            text=orjson.dumps({"error": f"request body is not JSON: {error}"}).decode(), content_type="application/json"
        ) from error
