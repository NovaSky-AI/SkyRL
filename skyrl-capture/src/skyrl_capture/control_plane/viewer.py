"""The viewer app: everything that reads what capture wrote.

Enabled by default and removable with ``--disable-viewer``, because it is not
part of capturing. It reads the record directory -- the same files a separate
viewer process would read -- so a deployment can run it on one replica, on a
standalone process, or not at all, and the answers are the same either way.

Reads never touch a proxy's memory. A trajectory mid-run is replayed from its
journal and a finished one is loaded from its committed record, which means
what the viewer shows is what survived, not what a process happens to be
holding. That is a product decision as much as an architectural one: a graph
on screen that a crash would erase is worse than one that is a second behind.

Listings are progressive. The index is built in the background, so the first
page arrives immediately and says so: ``indexing`` is true and ``total`` is
null until the scan finishes. A pager that prints a total it was told is
provisional prints a lie, so it is not told one.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import ORJSONResponse

from skyrl_capture.control_plane.models import ExportCreate
from skyrl_capture.domain.records import ExportError, TrajectoryError
from skyrl_capture.export.blocks import path_views
from skyrl_capture.export.service import ExportRequestError
from skyrl_capture.reader.records import RecordReader, TrajectoryQuery
from skyrl_capture.version import SCHEMA_VERSION, __version__

logger = logging.getLogger(__name__)

REFRESH_HELP = (
    "Rescan the record directory before answering. What a manual refresh in "
    "the UI sends; the index also refreshes on its own, and a refresh picks up "
    "new journals and newly committed records alike."
)


def build_viewer_app(
    *, reader: RecordReader, exports: Any, health: Any = None, lifespan: Any = None
) -> FastAPI:
    """Build the read API over one record directory.

    ``exports`` may be `None`, which serves reads without bulk exports -- what
    a process that only browses wants. ``health`` is supplied when this app is
    served on its own, so a standalone viewer answers `/healthz` too.
    """
    app = FastAPI(
        title="skyrl-capture viewer",
        version=__version__,
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
        lifespan=lifespan,
    )
    # So a caller that owns this app -- the capture runtime, a test, the
    # `view` command -- can reach the index without a second handle on it.
    app.state.reader = reader

    if health is not None:

        @app.get("/healthz")
        async def healthz() -> dict[str, Any]:
            return {
                "status": "ok",
                "version": __version__,
                "schema_version": SCHEMA_VERSION,
                **health.stats(),
            }

        @app.get("/readyz")
        async def readyz() -> dict[str, Any]:
            return {"ready": True, "viewer": "ok"}

    # -- runs -----------------------------------------------------------------
    @app.get("/v1/runs")
    async def list_runs(
        project: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        refresh: bool = Query(default=False, description=REFRESH_HELP),
    ) -> dict[str, Any]:
        """Runs, newest first, with the counts a listing shows.

        A run is derived, not stored: it is the grouping its trajectories
        imply. Nothing creates one and nothing writes to one, so there is no
        counter here that can disagree with its members.
        """
        await reader.apply_pending()
        if refresh:
            await reader.refresh()
        return {"data": reader.list_runs(project=project, limit=limit)}

    # -- trajectories ------------------------------------------------------------
    @app.get("/v1/trajectories")
    async def list_trajectories(
        project: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
        task_id: str | None = Query(default=None),
        step: int | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        cursor: str | None = Query(default=None),
        refresh: bool = Query(default=False, description=REFRESH_HELP),
    ) -> dict[str, Any]:
        await reader.apply_pending()
        if refresh:
            await reader.refresh()
        page = reader.list_trajectories(
            TrajectoryQuery(
                project=project,
                run_id=run_id,
                task_id=task_id,
                step=step,
                status=status,
                limit=limit,
                cursor=cursor,
            )
        )
        return {
            "data": page.items,
            "next_cursor": page.next_cursor,
            "has_more": page.next_cursor is not None,
            # Null while the index is still being built: a total that moves
            # under the caller is worse than no total.
            "total": page.total,
            "indexing": page.indexing,
            "indexed_trajectories": page.indexed_trajectories,
        }

    @app.get("/v1/trajectories/{identifier}")
    async def get_trajectory(identifier: str) -> dict[str, Any]:
        await reader.apply_pending()
        payload = await reader.get_trajectory(identifier)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"unknown trajectory {identifier!r}")
        return payload

    @app.get("/v1/trajectories/{identifier}/exchanges")
    async def list_exchanges(
        identifier: str,
        limit: int = Query(default=200, ge=1, le=1000),
        cursor: str | None = Query(default=None),
        provider: str | None = Query(default=None),
        model: str | None = Query(default=None),
        status: int | None = Query(default=None),
        retry_attempt: int | None = Query(default=None),
    ) -> dict[str, Any]:
        await reader.apply_pending()
        await _require(reader, identifier)
        rows, next_cursor = await reader.list_exchanges(
            identifier,
            limit=limit,
            cursor=cursor,
            provider=provider,
            model=model,
            status=status,
            retry_attempt=retry_attempt,
        )
        return {"data": rows, "next_cursor": next_cursor, "has_more": next_cursor is not None}

    @app.get("/v1/trajectories/{identifier}/graph")
    async def get_graph(identifier: str) -> dict[str, Any]:
        await reader.apply_pending()
        await _require(reader, identifier)
        return await reader.get_graph(identifier)

    @app.get("/v1/trajectories/{identifier}/paths")
    async def trajectory_paths(
        identifier: str,
        text: bool = Query(
            True,
            description=(
                "Include each block's decoded text and the path's logprobs. "
                "Pass false for the block shape alone -- what a mask strip "
                "needs, at a fraction of the bytes."
            ),
        ),
    ) -> dict[str, Any]:
        """The trajectory as training rows, decoded and blocked.

        One entry per root-to-leaf path, because a path is what an export row
        is -- so what is read on screen and what reaches the trainer are the
        same object. In tokens mode each path comes back as blocks of decoded
        text tagged `sampled`, `replayed`, `scaffold` or `given`, each carrying
        the token range it occupies in `input_ids`.

        Text and token offsets were captured by the renderer that created the
        IDs. This route only assembles them; neither it nor the browser owns a
        tokenizer. This is the same public API anything else scripts against;
        the UI has no private one.
        """
        await reader.apply_pending()
        view = await reader.view(identifier)
        if view is None:
            raise HTTPException(status_code=404, detail=f"unknown trajectory {identifier!r}")
        return {
            "trajectory": identifier,
            "mode": view.trajectory.mode,
            "tokenizer": view.trajectory.upstream_snapshot.get("tokenizer"),
            "paths": path_views(view, text=text),
        }

    # -- exports -------------------------------------------------------------
    if exports is not None:

        @app.post("/v1/exports", status_code=202)
        async def create_export(body: ExportCreate) -> dict[str, Any]:
            try:
                return await exports.create(
                    format=body.format,
                    project=body.project,
                    run=body.run,
                    trajectory=body.trajectory,
                    options=body.options,
                )
            except ExportRequestError as error:
                raise HTTPException(status_code=error.status, detail=error.detail) from error

        @app.get("/v1/exports/{identifier}")
        async def get_export(identifier: str) -> dict[str, Any]:
            payload = await exports.get(identifier)
            if payload is None:
                raise HTTPException(status_code=404, detail=f"unknown export {identifier!r}")
            return payload

        @app.get("/v1/exports/{identifier}/download")
        async def download_export(identifier: str) -> Response:
            """Managed download for deployments whose object store has no signed URLs."""
            try:
                found = await exports.artifact(identifier)
            except LookupError:
                raise HTTPException(
                    status_code=404, detail=f"unknown export {identifier!r}"
                ) from None
            if found is None:
                job = await exports.get(identifier)
                raise HTTPException(
                    status_code=409, detail=f"export is {job['status'] if job else 'unknown'}"
                )
            data, filename = found
            return Response(
                data,
                media_type="application/octet-stream",
                headers={"content-disposition": f'attachment; filename="{filename}"'},
            )

    @app.middleware("http")
    async def catch_domain_errors(request: Request, call_next: Any) -> Response:
        try:
            return await call_next(request)
        except (TrajectoryError, ExportError) as error:
            return ORJSONResponse({"detail": str(error)}, status_code=400)

    return app


async def _require(reader: RecordReader, identifier: str) -> None:
    if not await reader.trajectory_exists(identifier):
        raise HTTPException(status_code=404, detail=f"unknown trajectory {identifier!r}")


def record_viewer_app(root: str, *, public_url: str = "http://127.0.0.1:8750") -> FastAPI:
    """A standalone viewer over one record directory.

    What `skyrl-capture view --record` serves and what a separate viewer
    deployment runs. The same app the capture process mounts, plus the health
    document and the indexer's own lifecycle -- the index is a background task,
    and a background task belongs to the loop that will serve the requests.

    Raises `RecordNotFound` when there is no record at ``root``.
    """
    from skyrl_capture.control_plane.health import ViewerHealth
    from skyrl_capture.export.artifacts import ArtifactStore
    from skyrl_capture.export.jobs import ExportJobStore
    from skyrl_capture.export.service import ExportService
    from skyrl_capture.persistence.layout import export_artifacts_dir, export_jobs_dir

    reader = RecordReader(root)
    exports = ExportService(
        reader=reader,
        jobs=ExportJobStore(export_jobs_dir(reader.root)),
        artifacts=ArtifactStore(export_artifacts_dir(reader.root)),
        public_url=public_url,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The index is a background task, and a background task belongs to the
        # loop that will serve the requests.
        reader.start()
        exports.start()
        try:
            yield
        finally:
            await exports.stop()
            await reader.stop()

    return build_viewer_app(
        reader=reader, exports=exports, health=ViewerHealth(reader), lifespan=lifespan
    )
