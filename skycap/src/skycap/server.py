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
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import orjson
from aiohttp import web

from skycap.openai_chat import ChatRequest, RequestError, error_body, parse_request
from skycap.samples import build_samples
from skycap.trajectory import Trajectory, new_trajectory_id


class Backend(Protocol):
    """How a call reaches a model. One per server process: text or tokens."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def release(self, trajectory: Trajectory) -> None: ...
    async def models(self, request: web.Request) -> web.Response: ...
    async def chat(
        self, trajectory: Trajectory, request: web.Request, chat: ChatRequest, raw: bytes
    ) -> web.StreamResponse: ...


def _json(payload: Any, status: int = 200) -> web.Response:
    return web.Response(body=orjson.dumps(payload), status=status, content_type="application/json")


def _openai_error(message: str, status: int, *, code: str | None = None) -> web.Response:
    return web.Response(body=error_body(message, code=code), status=status, content_type="application/json")


class CaptureServer:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.trajectories: dict[str, Trajectory] = {}

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

    async def _on_cleanup(self, app: web.Application) -> None:
        await self.backend.close()

    # -- control plane --------------------------------------------------------
    async def healthz(self, request: web.Request) -> web.Response:
        open_count = sum(1 for t in self.trajectories.values() if t.is_open)
        return _json({"ok": True, "open_trajectories": open_count})

    async def create(self, request: web.Request) -> web.Response:
        body = await _read_json(request, default={})
        meta = body.get("meta") if isinstance(body, dict) else None
        if meta is not None and not isinstance(meta, dict):
            return _json({"error": "`meta` must be an object"}, 400)
        trajectory = Trajectory(id=new_trajectory_id(), meta=meta or {})
        self.trajectories[trajectory.id] = trajectory
        # The route is on the host the pool reached us at: harnesses reach it the same way.
        base = f"{request.scheme}://{request.host}"
        return _json({"id": trajectory.id, "base_url": f"{base}/t/{trajectory.id}/v1"})

    async def finish(self, request: web.Request) -> web.Response:
        trajectory = self.trajectories.get(request.match_info["id"])
        if trajectory is None:
            return _json({"error": "unknown trajectory"}, 404)
        body = await _read_json(request, default={})
        annotations = body.get("annotations") if isinstance(body, dict) else None
        if annotations is not None and not isinstance(annotations, dict):
            return _json({"error": "`annotations` must be an object"}, 400)
        if not trajectory.is_open and _changes(trajectory.annotations, annotations):
            # A repeat is answered as the first finish was. New annotations on it would be
            # silently lost, so they are refused instead.
            return _json({"error": "trajectory already finished; its annotations can't change"}, 409)
        if trajectory.is_open:
            trajectory.seal("finished", annotations)
            await self.backend.release(trajectory)
        return _json(
            {
                "id": trajectory.id,
                "status": trajectory.status,
                "samples": [s.to_json() for s in build_samples(trajectory.graph)],
            }
        )

    async def get(self, request: web.Request) -> web.Response:
        trajectory = self.trajectories.get(request.match_info["id"])
        if trajectory is None:
            return _json({"error": "unknown trajectory"}, 404)
        return _json(trajectory.document())

    # -- data plane -------------------------------------------------------------
    async def models(self, request: web.Request) -> web.Response:
        if request.match_info["id"] not in self.trajectories:
            return _openai_error("unknown trajectory", 404)
        return await self.backend.models(request)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        trajectory = self.trajectories.get(request.match_info["id"])
        if trajectory is None:
            return _openai_error("unknown trajectory", 404)
        if not trajectory.is_open:
            return _openai_error(f"trajectory is {trajectory.status}", 410, code="trajectory_closed")
        raw = await request.read()
        try:
            chat = parse_request(orjson.loads(raw))
        except (orjson.JSONDecodeError, RequestError) as error:
            return _openai_error(str(error), 400)
        task = asyncio.current_task()
        assert task is not None
        trajectory.inflight.add(task)
        trajectory.touch()
        try:
            return await self.backend.chat(trajectory, request, chat, raw)
        finally:
            trajectory.inflight.discard(task)
            trajectory.touch()


def _changes(current: dict[str, Any], update: dict[str, Any] | None) -> bool:
    """Whether ``update`` would change any annotation already recorded."""
    return any(key not in current or current[key] != value for key, value in (update or {}).items())


async def _read_json(request: web.Request, *, default: Any) -> Any:
    raw = await request.read()
    if not raw:
        return default
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError:
        return default
