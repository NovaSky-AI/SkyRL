"""Exports: creating a job, running it, and handing back the artifact.

An export job materializes a deterministic artifact into the exports directory
beside the record, and serves it from there. There is one copy and one way to
get it: `GET /v1/exports/{id}/download`.

The artifact is plain JSONL at every scope. What a consumer needs to judge the
dataset is on the job record already -- the exact selected trajectory IDs, the
options used, the record and byte counts, the checksum -- and per-trajectory
capture integrity travels inside the data, on each trajectory it describes.

This lives in the viewer service, not in capture. Bulk exports read finished
records, which is a read of the record directory rather than anything the
capture path owns; the only export capture itself renders is the one `finish`
returns, and that calls the same pure formatters below without a job, a queue
or a worker.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from typing import Any

import orjson

from skyrl_capture.compression import compress
from skyrl_capture.domain.models import ExportError, export_public, normalize_export_format
from skyrl_capture.export import formats
from skyrl_capture.export.artifacts import ArtifactStore, artifact_key
from skyrl_capture.export.jobs import ExportJob, ExportJobStore
from skyrl_capture.ids import export_id as new_export_id

logger = logging.getLogger(__name__)

_EXTENSION = {
    "graph": "jsonl",
    "replay": "jsonl",
    "text_samples": "jsonl",
    "token_samples": "jsonl",
}


class ExportRequestError(Exception):
    """A job that cannot be created, with the HTTP status that says why."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def artifact_filename(row: Any) -> str:
    extension = _EXTENSION.get(row["format"], "jsonl")
    scope = row["trajectory_id"] or row["run_id"] or row["project"] or "export"
    return f"{scope}-{row['format']}.{extension}.zst"


def render_records(
    view: Any, *, export_format: str, options: dict[str, Any], origin: Any = None
) -> list[dict[str, Any]]:
    """One trajectory's rows, in one format. Pure: a view in, records out."""
    if export_format == "graph":
        return list(formats.graph_records(view))
    if export_format == "replay":
        return formats.replay_records(view, origin=origin)
    if export_format == "text_samples":
        return formats.text_sample_records(
            view,
            allow_repeated_targets=bool(options.get("allow_repeated_targets", False)),
            mask_abandoned=bool(options.get("mask_abandoned", False)),
        )
    if export_format == "token_samples":
        return formats.token_sample_records(
            view,
            allow_repeated_targets=bool(options.get("allow_repeated_targets", False)),
            mask_abandoned=bool(options.get("mask_abandoned", False)),
            overlong_filtering=bool(options.get("overlong_filtering", False)),
        )
    raise ExportError(f"unsupported format {export_format!r}")


def render_artifact(views: Any, *, export_format: str, options: dict[str, Any], origin: Any) -> bytes:
    """The JSONL body for a set of trajectories, in observed order."""
    lines: list[bytes] = []
    for view in views:
        lines.extend(
            orjson.dumps(record)
            for record in render_records(
                view, export_format=export_format, options=options, origin=origin
            )
        )
    return b"\n".join(lines) + (b"\n" if lines else b"")


class ExportService:
    """Create, run and serve export jobs. Owned by the viewer service."""

    def __init__(
        self,
        *,
        reader: Any,
        jobs: ExportJobStore,
        artifacts: ArtifactStore,
        public_url: str,
        poll_interval: float = 0.2,
    ) -> None:
        self._reader = reader
        self._jobs = jobs
        self._artifacts = artifacts
        self._public_url = public_url.rstrip("/")
        self._poll_interval = poll_interval
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        # One job at a time per process, and drain() waits on the same lock, so
        # a caller that drains cannot observe a job that is still running.
        self._running = asyncio.Lock()

    # -- creating ------------------------------------------------------------
    async def create(
        self,
        *,
        format: str,
        project: str | None,
        run: str | None,
        trajectory: str | None,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Fix the snapshot, validate it, record the job, and schedule it.

        A project or run export selects every trajectory whose record is
        committed at the moment the request is accepted, and that selection is
        what the job carries -- re-running the same job produces byte-identical
        output. "At the moment the request is accepted" means after the record
        directory has been read through, not after as much of it as the
        background scan happened to have reached.
        """
        reader = self._reader
        # The whole directory, read through, before anything is selected. A
        # listing may be a scan behind; a dataset may not. A project or run
        # export requested while the index was still being built would
        # otherwise fix a snapshot that silently omits every trajectory the
        # scan had not reached -- and nothing downstream could tell.
        await reader.ensure_indexed()
        try:
            export_format = normalize_export_format(format)
        except ExportError as error:
            raise ExportRequestError(400, str(error)) from error
        scopes = [scope for scope in (project, run, trajectory) if scope]
        if len(scopes) != 1:
            raise ExportRequestError(400, "exactly one of project, run or trajectory is required")

        if project:
            selected = await reader.finished_trajectory_ids(project=project, run_id=None)
        elif run:
            if not reader.run_exists(run):
                raise ExportRequestError(404, f"unknown run {run!r}")
            selected = await reader.finished_trajectory_ids(project=None, run_id=run)
        else:
            identifier = trajectory or ""
            if not await reader.trajectory_exists(identifier):
                raise ExportRequestError(404, f"unknown trajectory {trajectory!r}")
            selected = [identifier]

        # Asking for token samples from text-mode capture cannot be satisfied.
        # Writing an empty artifact for it reads like "nothing matched" rather
        # than "this cannot be produced", so it is refused here, where the
        # snapshot is chosen -- no job is created to go and fail later.
        #
        # The test is the capture mode, not the record count: zero rows is
        # legitimate on its own, since a trajectory that made no calls exports
        # zero of anything.
        if export_format == "token_samples" and selected:
            modes = await reader.modes_for(selected)
            if not modes.get("tokens"):
                found_modes = ", ".join(f"{count} in {mode!r}" for mode, count in sorted(modes.items()))
                raise ExportRequestError(
                    400,
                    f"token_samples needs capture in 'tokens' mode, and none of the "
                    f"{len(selected)} selected trajectories is ({found_modes}). Exact token IDs "
                    f"are recorded only when this process runs a 'tokens' upstream; use "
                    f"text_samples for text capture.",
                )

        job = ExportJob(
            id=new_export_id(),
            format=export_format,
            project=project,
            run_id=run,
            trajectory_id=trajectory,
            selected_trajectory_ids=list(selected),
            options=dict(options or {}),
        )
        await self._jobs.put(job)
        self.schedule(job.id)
        return export_public(job.row())

    async def get(self, identifier: str) -> dict[str, Any] | None:
        """The public job record, with a download URL once it is ready."""
        job = await self._jobs.get(identifier)
        if job is None:
            return None
        payload = export_public(job.row())
        if job.status == "ready" and job.output_uri:
            payload["download_url"] = f"{self._public_url}/v1/exports/{job.id}/download"
        return payload

    async def artifact(self, identifier: str) -> tuple[bytes, str] | None:
        """The stored artifact and its filename, or `None` if not ready.

        Raises `LookupError` for a job that does not exist, so the caller can
        tell "not yet" from "never".
        """
        job = await self._jobs.get(identifier)
        if job is None:
            raise LookupError(identifier)
        if job.status != "ready":
            return None
        filename = artifact_filename(job.row())
        return await self._artifacts.read(artifact_key(job.id, filename)), filename

    # -- running -------------------------------------------------------------
    def schedule(self, identifier: str) -> None:
        self._queue.put_nowait(identifier)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="export-runner")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def drain(self) -> int:
        """Run every queued and unfinished job to completion.

        Used by the CLI and by tests. Waiting on ``_running`` is what makes this
        safe alongside the background loop: if the loop already claimed a job,
        this waits for it rather than reporting an empty queue while work is
        still in flight.
        """
        count = 0
        while True:
            drained_any = False
            while not self._queue.empty():
                identifier = self._queue.get_nowait()
                await self._execute(identifier)
                count += 1
                drained_any = True
            async with self._running:
                pass
            identifier = await self._claim_pending()
            if identifier is not None:
                await self._execute(identifier)
                count += 1
                drained_any = True
            if not drained_any and self._queue.empty():
                return count

    async def _execute(self, identifier: str) -> None:
        async with self._running:
            await self._run_export(identifier)

    async def _run(self) -> None:
        """Poll for work forever.

        Nothing short of cancellation may leave this loop. An export that
        fails is recorded and the runner carries on; a runner that dies stops
        every future export with no symptom except jobs sitting in ``pending``,
        which is far worse than one failed job.
        """
        while not self._stopping.is_set():
            try:
                identifier = await self._next_identifier()
                if identifier is None:
                    continue
                await self._execute(identifier)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("export runner iteration failed")
                await asyncio.sleep(self._poll_interval)

    async def _next_identifier(self) -> str | None:
        try:
            return await asyncio.wait_for(self._queue.get(), self._poll_interval)
        except TimeoutError:
            return await self._claim_pending()

    async def _claim_pending(self) -> str | None:
        unfinished = await self._jobs.unfinished()
        return unfinished[0].id if unfinished else None

    async def _run_export(self, identifier: str) -> None:
        job = await self._jobs.get(identifier)
        if job is None or job.status not in ("pending", "running"):
            return
        job.touch("running")
        await self._jobs.put(job)
        try:
            await self._materialize(job)
        except Exception as error:
            logger.exception("export %s failed", identifier)
            job.error = f"{type(error).__name__}: {error}"[:4000]
            job.touch("failed")
            await self._jobs.put(job)

    async def _materialize(self, job: ExportJob) -> None:
        selected = list(job.selected_trajectory_ids)

        # Scheduling is relative to the first call in the *export*, so several
        # trajectories land on one timeline instead of all restarting at zero.
        origin = await self._reader.earliest_request(selected)
        views = []
        for trajectory_id in selected:
            view = await self._reader.view(trajectory_id)
            if view is None:
                raise ExportError(f"trajectory {trajectory_id!r} is no longer readable")
            views.append(view)
        body = render_artifact(views, export_format=job.format, options=job.options, origin=origin)
        lines = body.splitlines()

        # One artifact shape at every scope: compressed JSONL. A project export
        # is a concatenation, not a bundle: what a consumer needs to judge it is
        # on the job record, and per-trajectory integrity travels inside the
        # data.
        artifact = compress(body)
        filename = artifact_filename(job.row())
        job.output_uri = await self._artifacts.write(artifact_key(job.id, filename), artifact)
        job.byte_count = len(artifact)
        job.record_count = len(lines)
        job.checksum = "sha256:" + hashlib.sha256(artifact).hexdigest()
        job.touch("ready")
        await self._jobs.put(job)
