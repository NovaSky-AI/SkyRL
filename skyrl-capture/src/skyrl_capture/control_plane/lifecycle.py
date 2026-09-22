"""The lifecycle app: create, finish, annotate, and say how capture is doing.

This is what a capture replica always mounts. It is small on purpose -- three
writes and three operational reads -- because it is the only part of the
control plane a capture process needs in order to be useful. Everything that
reads captured data lives in the viewer app, and a replica may be started
without it.

**There is no control credential.** Capture is an ephemeral in-cluster job,
brought up beside the inference server it captures and torn down with it; its
control plane is reachable only from inside that job's network, exactly like
the inference server it sits in front of. A deployment that needs more puts
this behind whatever fronts its inference server -- which is where a shared
secret belongs, not duplicated here.

What this app does *not* have is as deliberate as what it does. There is no
delete: a trajectory's record is the result of a run, and removing it is a
file operation on the record directory, not an API call. There is no run
metadata: a run is a grouping derived from its trajectories, not a row. And
there is no trajectory TTL, so nothing here expires anything.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse

from skyrl_capture.control_plane.commands import PersistenceUnavailable
from skyrl_capture.control_plane.models import (
    MetadataUpdate,
    TrajectoryCreate,
    TrajectoryFinish,
)
from skyrl_capture.domain.hashing import canonical_hash
from skyrl_capture.domain.records import (
    ExportError,
    MetadataError,
    TrajectoryConflict,
    TrajectoryError,
)
from skyrl_capture.version import SCHEMA_VERSION, __version__

logger = logging.getLogger(__name__)


def build_lifecycle_app(*, commands: Any, health: Any) -> FastAPI:
    app = FastAPI(
        title="skyrl-capture lifecycle",
        version=__version__,
        docs_url="/v1/docs",
        openapi_url="/v1/openapi.json",
    )

    # -- health ---------------------------------------------------------------
    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        """Liveness plus capture health.

        Surfaces commit pressure, capture gaps and store errors, because a
        capture outage is invisible from the provider response by design --
        this is where an operator sees it.

        **Always returns 200.** ``status`` is for a human and a dashboard; a
        liveness probe keyed on the HTTP status is therefore unaffected by a
        capture problem, which is the point. A proxy that is serving correctly
        must not be restarted because the disk is behind.
        """
        stats = health.stats()
        commits = stats["commits"]
        degraded = commits["failures"] or commits["refused"] or commits["unwritten_gaps"]
        return {
            "status": "degraded" if degraded else "ok",
            "persistence": "degraded" if commits["failures"] else "ok",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            **stats,
        }

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        """Can this replica take traffic?

        Deliberately narrow: it answers for the proxy, not for capture. Wiring
        capture health in here would let a capture failure pull a healthy
        replica out of the load balancer, which is the one thing the design
        forbids. Capture trouble is reported on ``/healthz`` and in metrics,
        where it pages a human instead of moving traffic.
        """
        return {"ready": True, "proxy": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        """Prometheus text exposition of the capture-path counters."""
        return Response(_metrics_text(health.stats()), media_type="text/plain; version=0.0.4")

    # -- trajectories -----------------------------------------------------------
    @app.post("/v1/trajectories", status_code=201)
    async def create_trajectory(body: TrajectoryCreate) -> dict[str, Any]:
        """Create one trajectory, durably, and return the route to send it to.

        The id is required and the caller's. Repeating this call with the same
        id and the same body returns the same answer -- including after a
        restart, because the hash of the creation body is on disk beside the
        trajectory. Reusing the id with a different body is a `409`.
        """
        try:
            created = await commands.create(
                trajectory_id=body.trajectory_id,
                project=body.project,
                run_id=body.run_id,
                task_id=body.task_id,
                step=body.step,
                labels=body.labels,
                annotations=body.annotations,
                bodies=body.bodies,
                source_metadata=body.source_metadata,
                request_hash=canonical_hash(body.model_dump()),
            )
        except TrajectoryConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except TrajectoryError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except PersistenceUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return created.public()

    @app.post("/v1/trajectories/{identifier}/finish")
    async def finish_trajectory(identifier: str, body: TrajectoryFinish) -> dict[str, Any]:
        """Close the trajectory, commit its record, and return it rendered.

        `200`, not `202`: by the time this answers the canonical record is on
        disk and the reply is rendered from it. An identical retry returns the
        same committed trajectory -- in whichever format it asks for, since the
        format decides the rendering rather than the record.
        """
        try:
            envelope = await commands.finish(
                identifier,
                labels=body.labels,
                annotations=body.annotations,
                command_result=body.command_result,
                export_format=body.format,
                options=body.options,
            )
        except TrajectoryConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ExportError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except TrajectoryError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except PersistenceUnavailable as error:
            # Retryable, and the SDK does retry it. Nothing was decided wrongly;
            # the disk did not take it.
            raise HTTPException(status_code=503, detail=str(error)) from error
        return envelope.public()

    # -- metadata ------------------------------------------------------------
    # There is no GET here. `labels` and `annotations` are fields on the
    # trajectory, so the viewer's `GET /v1/trajectories/{id}` already carries
    # them; a second route returning a subset of one document is a second thing
    # to keep agreeing with the first.
    @app.patch("/v1/trajectories/{identifier}/metadata")
    async def update_metadata(identifier: str, body: MetadataUpdate) -> dict[str, Any]:
        """Merge labels and annotations, before or after the trajectory finished.

        A reward usually arrives after the trial. An active trajectory takes
        the edit as one more journal record; a finished one has its committed
        record rewritten with a new revision.
        """
        try:
            return await commands.annotate(
                identifier,
                annotations=body.annotations,
                remove_annotations=body.remove_annotations,
                labels=body.labels,
                remove_labels=body.remove_labels,
            )
        except MetadataError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except TrajectoryError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.middleware("http")
    async def catch_domain_errors(request: Request, call_next: Any) -> Response:
        try:
            return await call_next(request)
        except TrajectoryConflict as error:
            return ORJSONResponse({"detail": str(error)}, status_code=409)
        except (TrajectoryError, ExportError) as error:
            return ORJSONResponse({"detail": str(error)}, status_code=400)
        except PersistenceUnavailable as error:
            return ORJSONResponse({"detail": str(error)}, status_code=503)

    return app


def _metrics_text(stats: dict[str, Any]) -> str:
    commits = stats["commits"]
    registry = stats["registry"]
    store = stats["store"]
    capture = stats["capture"]
    lines = [
        "# HELP capture_requests_total Requests served by the data plane.",
        "# TYPE capture_requests_total counter",
        f"capture_requests_total {stats['requests_served']}",
        "# HELP capture_commits_total Journal appends completed.",
        "# TYPE capture_commits_total counter",
        f"capture_commits_total {commits['commits']}",
        "# HELP capture_commits_pending Journal appends queued and not yet durable.",
        "# TYPE capture_commits_pending gauge",
        f"capture_commits_pending {commits['pending_commits']}",
        "# HELP capture_commits_pending_high_water The most appends ever queued at once.",
        "# TYPE capture_commits_pending_high_water gauge",
        f"capture_commits_pending_high_water {commits['pending_high_water']}",
        "# HELP capture_commit_oldest_pending_seconds Age of the oldest queued append.",
        "# TYPE capture_commit_oldest_pending_seconds gauge",
        f"capture_commit_oldest_pending_seconds {commits['oldest_pending_age_s']}",
        "# HELP capture_commits_refused_total Appends refused because the bound was reached.",
        "# TYPE capture_commits_refused_total counter",
        f"capture_commits_refused_total {commits['refused']}",
        "# HELP capture_commit_failures_total Appends that failed.",
        "# TYPE capture_commit_failures_total counter",
        f"capture_commit_failures_total {commits['failures']}",
        "# HELP capture_gaps_unwritten Capture gaps marked in memory and not yet on disk.",
        "# TYPE capture_gaps_unwritten gauge",
        f"capture_gaps_unwritten {commits['unwritten_gaps']}",
        "# HELP capture_undelivered_nodes Graph nodes of lost records waiting for one to carry them.",
        "# TYPE capture_undelivered_nodes gauge",
        f"capture_undelivered_nodes {commits['undelivered_nodes']}",
        "# HELP capture_hot_trajectories Trajectories this process holds in memory.",
        "# TYPE capture_hot_trajectories gauge",
        f"capture_hot_trajectories {registry['hot_trajectories']}",
        "# HELP capture_lazy_recoveries_total Trajectories recovered from a journal.",
        "# TYPE capture_lazy_recoveries_total counter",
        f"capture_lazy_recoveries_total {registry['recovered']}",
        "# HELP capture_journal_append_ms Mean journal append latency, including fsync.",
        "# TYPE capture_journal_append_ms gauge",
        f"capture_journal_append_ms {store['append_ms_mean']}",
        "# HELP capture_journal_fsync_ms Mean fsync latency.",
        "# TYPE capture_journal_fsync_ms gauge",
        f"capture_journal_fsync_ms {store['fsync_ms_mean']}",
        "# HELP capture_recovery_ms Mean time to recover one trajectory from its journal.",
        "# TYPE capture_recovery_ms gauge",
        f"capture_recovery_ms {store['recovery_ms_mean']}",
        "# HELP capture_torn_tails_total Journals found ending in a torn record.",
        "# TYPE capture_torn_tails_total counter",
        f"capture_torn_tails_total {store['torn_tails']}",
    ]
    if "close_wait_ms_per_turn" in capture:
        lines += [
            "# HELP capture_tito_close_wait_ms Mean time a TITO response waits for durability.",
            "# TYPE capture_tito_close_wait_ms gauge",
            f"capture_tito_close_wait_ms {capture['close_wait_ms_per_turn']}",
            "# HELP capture_tito_poisoned_total Trajectories token capture could not extend.",
            "# TYPE capture_tito_poisoned_total counter",
            f"capture_tito_poisoned_total {capture['trajectories_poisoned']}",
            "# HELP capture_tito_delivery_unconfirmed_total Responses whose delivery was never recorded.",
            "# TYPE capture_tito_delivery_unconfirmed_total counter",
            f"capture_tito_delivery_unconfirmed_total {capture['delivery_unconfirmed']}",
        ]
    else:
        lines += [
            "# HELP capture_text_errors_total Text capture failures, each one a gap.",
            "# TYPE capture_text_errors_total counter",
            f"capture_text_errors_total {capture['capture_errors']}",
        ]
    return "\n".join(lines) + "\n"
