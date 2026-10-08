"""The capture server: a control plane plus one OpenAI-compatible route per trajectory.

    POST /trajectories                    {meta}                -> {id, base_url, api_key[, exposed_base_url]}
    POST /trajectories/{id}/finish        {annotations, paths}  -> {status, samples, record}
    GET  /trajectories/{id}                                      -> the trajectory document
    POST /t/{id}/v1/chat/completions                             (the harness)
    GET  /t/{id}/v1/models                                       (passed through)
    GET  /healthz

A trajectory's route is its URL: the harness needs no header and no SDK patch.
Each trajectory also gets an API key of its own, minted at ``create``. With
``require_api_key``, its harness routes answer only a caller that sends it the
way any OpenAI client sends its key (``Authorization: Bearer <api_key>``), so a
route reachable from outside (a sandbox, a tunnel) can't be written to by anyone
who merely learns the URL. The key is never recorded.
``finish`` seals the trajectory: in-flight calls are cancelled and never
committed, so its samples are final the moment they are returned. ``paths``
names the path rule that picks them (``skycap.paths``): ``all`` (default),
``final``, or a custom rule the server was built with.

``harness_app`` serves the harness routes alone. A server with an exposure
(``skycap.exposure``) listens a second time with it, for harnesses outside this
network, and ``create`` adds the trajectory's route on the exposed URL,
``exposed_base_url``. The control plane is never routed there.

With a ``record_dir``, a trajectory is written when it ends -- by ``finish``,
by the idle TTL (as ``abandoned``), or by a graceful shutdown (as ``open``) --
and then dropped from memory; reads of it are served from disk. Without one,
ended trajectories stay in memory, which is only for tests and development,
unless ``keep_unrecorded=False``: then a trajectory is dropped once it ends,
and the reply to its ``finish`` is the only copy of its samples. That is for a
trainer that keeps no record.

With a ``record_mirror`` as well, each written record is then copied to that
URL in the background (``skycap.mirror``). The copy fails open: a slow or
failing store never fails a trajectory, and its losses are counted on
``/healthz``. ``finish`` answers with where the record is: its path on this
server's disk, its mirror URI, and its file names.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import orjson
from aiohttp import web

from skycap import record
from skycap.mirror import RecordMirror
from skycap.openai_chat import ChatRequest, RequestError, error_body, parse_request
from skycap.paths import PATH_RULE_FAILED, PathRule, Row, rule_registry
from skycap.samples import Sample, build_samples, samples_for
from skycap.trajectory import Status, Trajectory, new_api_key, new_trajectory_id

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


def node_address() -> str:
    """This machine's primary IP address: the one its default route leaves from. Falls back to its hostname.

    Connecting a UDP socket sends nothing; it only makes the kernel pick the outgoing interface.
    """
    for family, probe in ((socket.AF_INET, ("8.8.8.8", 53)), (socket.AF_INET6, ("2001:4860:4860::8888", 53))):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as probe_socket:
                probe_socket.connect(probe)
                return probe_socket.getsockname()[0]
        except OSError:
            continue
    return socket.gethostname()


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
        record_mirror: str | RecordMirror | None = None,
        record_mirror_config: Mapping[str, Any] | None = None,
        record_host: str | None = None,
        ttl: float = 3600.0,
        sweep_interval: float = 60.0,
        path_rules: Mapping[str, PathRule | str] | None = None,
        keep_unrecorded: bool = True,
        require_api_key: bool = False,
    ) -> None:
        if record_dir is not None and not keep_unrecorded:
            raise ValueError("keep_unrecorded=False is for a server with no record_dir")
        self.backend = backend
        self.keep_unrecorded = keep_unrecorded
        #: Whether the harness routes check the trajectory's own key.
        self.require_api_key = require_api_key
        #: The rules ``finish`` accepts, by name: the built-in ones plus ``path_rules``, each given as a
        #: function or as its ``"pkg.module:function"`` import path.
        self.path_rules = rule_registry(path_rules)
        self.record_dir = Path(record_dir).absolute() if record_dir is not None else None
        #: The machine ``record_dir`` is on, as ``finish``'s ``record.host``: so another node (a trainer's
        #: head node, say) knows where to reach a record that only exists here.
        self.record_host = record_host or (node_address() if self.record_dir is not None else None)
        if record_mirror is not None and self.record_dir is None:
            raise ValueError("record_mirror copies what is written to record_dir, so it needs a record_dir")
        #: The remote copy of the record directory, if any.
        if record_mirror_config and not isinstance(record_mirror, str):
            raise ValueError("record_mirror_config configures a mirror given by URL; set it on a RecordMirror directly")
        self.mirror = (
            RecordMirror.from_config(record_mirror, record_mirror_config)
            if isinstance(record_mirror, str)
            else record_mirror
        )
        #: Seconds an open trajectory may go without a request before it is abandoned.
        self.ttl = ttl
        self.sweep_interval = sweep_interval
        self.trajectories: dict[str, Trajectory] = {}
        self._sweeper: asyncio.Task[None] | None = None
        #: Where the harness listener is reached from outside, once an exposure has opened it.
        self.exposed_url: str | None = None

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

    def harness_app(self) -> web.Application:
        """The harness routes alone, served alongside ``app`` by the same server.

        It has no control plane to reach and no lifecycle of its own: ``app`` starts and stops the backend.
        """
        app = web.Application(client_max_size=1024**3)
        app.router.add_post("/t/{id}/v1/chat/completions", self.chat)
        app.router.add_get("/t/{id}/v1/models", self.models)
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
        if self.mirror is not None:
            # Bounded by the mirror's shutdown deadline; what is left is dropped with a warning.
            await asyncio.to_thread(self.mirror.close)

    # -- ending a trajectory -----------------------------------------------------
    async def end(
        self,
        trajectory: Trajectory,
        status: Status,
        annotations: dict[str, Any] | None = None,
        paths: str | None = None,
    ) -> list[Sample]:
        """Seal, release the upstream session, write, and drop from memory.

        A trajectory that already failed keeps its status; ``annotations`` are
        still recorded on it. It is written when this call ended it (including
        one read back from a shutdown-time record) or when it is still in
        memory, which means an earlier write failed and is retried.

        With ``paths`` (a ``finish``), the sealed graph's samples are built by
        that rule and recorded, so the record says what trained, and returned.
        A rule that raises leaves the trajectory ended and written without
        samples, and its error is raised; a later ``finish`` runs it again, and
        the trajectory is written again if that one records samples.
        """
        ended_now = not trajectory.ended
        recorded = trajectory.samples is not None
        if trajectory.is_open:
            trajectory.seal(status, annotations)
        elif not trajectory.ended:
            trajectory.annotations.update(annotations or {})
        samples: list[Sample] = []
        error: Exception | None = None
        if paths is not None:
            try:
                samples = self._samples(trajectory, paths)
            except Exception as caught:  # noqa: BLE001 - raised once the trajectory is written
                error = caught
        if not trajectory.ended:
            trajectory.ended = True
            try:
                await self.backend.release(trajectory)
            except Exception:
                logger.exception("releasing %s failed", trajectory.id)
        unwritten = self.trajectories.get(trajectory.id) is trajectory
        recorded_now = not recorded and trajectory.samples is not None
        if (ended_now or unwritten or recorded_now) and await self._persist(trajectory):
            self.trajectories.pop(trajectory.id, None)
        elif not self.keep_unrecorded and error is None:
            # A rule that raised keeps the trajectory, so a later finish can run it again.
            self.trajectories.pop(trajectory.id, None)
        if error is not None:
            raise error
        return samples

    def _samples(self, trajectory: Trajectory, paths: str) -> list[Sample]:
        """The samples of the rule named ``paths``, recorded on the trajectory the first time."""
        graph = trajectory.graph
        if trajectory.samples is not None:
            # A repeat: the rows the first finish recorded, rather than the rule run again.
            rows = [Row(graph.path_to(row["leaf"]), row["targets"]) for row in trajectory.samples["rows"]]
            return samples_for(graph, rows)
        samples = build_samples(graph, self.path_rules[paths])
        trajectory.samples = {
            "paths": paths,
            "rows": [{"leaf": sample.leaf, "targets": sample.targets} for sample in samples],
        }
        return samples

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
        if self.mirror is not None:
            self.mirror.submit(self.record_dir, trajectory.id)
        return True

    def location(self, trajectory: Trajectory) -> dict[str, Any] | None:
        """Where a trajectory's record is: ``{"host", "path", "mirror", "files"}``, or None when it isn't written.

        ``files`` names the record's files, sidecars then document: as the mirror holds them when there is
        one (without the sidecar kinds it excludes), else as the record directory does.
        """
        if self.record_dir is None or self.trajectories.get(trajectory.id) is trajectory:
            return None
        name = record.document_path(self.record_dir, trajectory.id).name
        if self.mirror is None:
            files = [path.name for path in record.record_files(self.record_dir, trajectory.id)]
        else:
            files = self.mirror.names(self.record_dir, trajectory.id)
        return {
            "host": self.record_host,
            "path": str(self.record_dir / name),
            "mirror": None if self.mirror is None else self.mirror.uri(name),
            "files": files,
        }

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
        health = {"ok": True, "open_trajectories": open_count, "capture": self.backend.describe()}
        if self.mirror is not None:
            health["record_mirror"] = {"url": self.mirror.url, **self.mirror.stats()}
        return _json(health)

    async def create(self, request: web.Request) -> web.Response:
        body = await _read_json(request, default={})
        meta = body.get("meta") if isinstance(body, dict) else None
        if meta is not None and not isinstance(meta, dict):
            return _json({"error": "`meta` must be an object"}, 400)
        trajectory = Trajectory(
            id=new_trajectory_id(), meta=meta or {}, capture=self.backend.describe(), api_key=new_api_key()
        )
        self.trajectories[trajectory.id] = trajectory
        # The route is on the host the pool reached us at: harnesses reach it the same way.
        base = f"{request.scheme}://{request.host}"
        created = {"id": trajectory.id, "base_url": f"{base}/t/{trajectory.id}/v1", "api_key": trajectory.api_key}
        if self.exposed_url is not None:
            created["exposed_base_url"] = f"{self.exposed_url}/t/{trajectory.id}/v1"
        return _json(created)

    async def finish(self, request: web.Request) -> web.Response:
        body = await _read_json(request, default={})
        annotations = body.get("annotations") if isinstance(body, dict) else None
        if annotations is not None and not isinstance(annotations, dict):
            return _json({"error": "`annotations` must be an object"}, 400)
        paths = body.get("paths", "all") if isinstance(body, dict) else "all"
        if not isinstance(paths, str):
            return _json({"error": "`paths` must be a string"}, 400)
        trajectory = await self._lookup(request.match_info["id"])
        if trajectory is None:
            return _json({"error": "unknown trajectory"}, 404)
        if trajectory.ended and _changes(trajectory.annotations, annotations):
            # A repeat is answered as the first finish was. New annotations on it would be
            # silently lost, so they are refused instead.
            return _json({"error": "trajectory already finished; its annotations can't change"}, 409)
        if trajectory.samples is not None:
            # Likewise, the record says which rule trained it, so a repeat can't ask for another. A repeat
            # naming that rule is answered from the recorded rows, so the server needn't still have it.
            if trajectory.samples["paths"] != paths:
                return _json({"error": f"trajectory already finished with paths={trajectory.samples['paths']!r}"}, 409)
        elif paths not in self.path_rules:
            return _json({"error": f"`paths` must be one of {sorted(self.path_rules)}"}, 400)
        try:
            samples = await self.end(trajectory, "finished", annotations, paths=paths)
        except Exception as error:  # noqa: BLE001 - a custom rule's failure, reported to the caller
            logger.exception("path rule %r failed on %s", paths, trajectory.id)
            message = f"path rule {paths!r} failed: {type(error).__name__}: {error}"
            return _json({"error": message, "code": PATH_RULE_FAILED}, 500)
        return _json(
            {
                "id": trajectory.id,
                "status": trajectory.status,
                "samples": [s.to_json() for s in samples],
                "unbridged_calls": trajectory.graph.unbridged_calls(),
                "record": self.location(trajectory),
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
        trajectory = self.trajectories.get(trajectory_id)
        if trajectory is None:
            return self._ended_or_unknown(trajectory_id)
        # Ended first: an ended trajectory answers 410 whatever key is sent, and its key opens nothing more.
        if not trajectory.is_open:
            return _openai_error(f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
        if not self._authorized(trajectory, request):
            return _unauthorized()
        return await self.backend.models(request)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        trajectory_id = request.match_info["id"]
        trajectory = self.trajectories.get(trajectory_id)
        if trajectory is None:
            return self._ended_or_unknown(trajectory_id)
        if not trajectory.is_open:
            return _openai_error(f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
        if not self._authorized(trajectory, request):
            return _unauthorized()
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
            return await trajectory.replay.call(
                request.headers, raw, lambda: self._start(trajectory, request, chat, raw)
            )
        finally:
            trajectory.inflight.discard(task)
            trajectory.touch()

    def _authorized(self, trajectory: Trajectory, request: web.Request) -> bool:
        """Whether the call carries this trajectory's key, or no key is required."""
        if not self.require_api_key:
            return True
        if trajectory.api_key is None:
            return False
        parts = request.headers.get("Authorization", "").split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        # Bytes: compare_digest refuses str with non-ASCII characters, which a caller controls. aiohttp keeps
        # header bytes that aren't UTF-8 as surrogates; surrogateescape turns them back into those bytes.
        given = parts[1].strip().encode("utf-8", "surrogateescape")
        return hmac.compare_digest(given, trajectory.api_key.encode())

    def _start(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> asyncio.Future[web.StreamResponse]:
        """Start the call as its own task. ``finish`` still cancels it, as in-flight work."""
        work = asyncio.ensure_future(self.backend.chat(trajectory, request, chat, raw))
        trajectory.inflight.add(work)
        work.add_done_callback(trajectory.inflight.discard)  # type: ignore[arg-type]
        return work


def _unauthorized() -> web.Response:
    return _openai_error("this trajectory's api_key is required", 401, code="invalid_api_key")


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
