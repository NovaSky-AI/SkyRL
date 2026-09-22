"""Process composition and lifecycle. Nothing else.

`build_runtime` constructs every component in dependency order and hands the
finished set to a `Runtime`, which starts and stops them. That is the whole
job. A `Runtime` is never given to a component, no component reaches through
it to find a sibling, and there is no `None` placeholder that something fills
in afterwards -- with one stated exception, the eviction callback, because
token capture's session manager reads through the registry that calls it.

**The mode is decided here, once.** A process captures text or it captures
tokens, for its whole life, and the proxy it is not was never built: no
tokenizer, no renderer, no token engine, no session manager and no trace
source exist in a text deployment.

**The viewer is decided here too.** A replica serves the read API and bulk
exports, or it does not. Only one process per record directory should, because
there is one indexer per record root; the rest run `--disable-viewer` and
capture alone.

The order is the dependency graph read top to bottom:

    record dir, stores, registry, coordinator, transport   (resources)
    the one capture proxy, data plane                      (the capture path)
    commands, health, lifecycle app                        (the lifecycle)
    reader, exports, viewer app                            (optional)
    runtime                                                (last)

Runtime does not own the ASGI server, listening socket, or serving thread.
Its start and stop methods are called by CaptureApplication's lifespan.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from skyrl_capture.config import Config, TitoUpstream, load_config
from skyrl_capture.control_plane.commands import CaptureCommands
from skyrl_capture.control_plane.health import HealthProvider
from skyrl_capture.control_plane.lifecycle import build_lifecycle_app
from skyrl_capture.control_plane.viewer import build_viewer_app
from skyrl_capture.data_plane.app import DataPlane
from skyrl_capture.export.artifacts import ArtifactStore
from skyrl_capture.export.jobs import ExportJobStore
from skyrl_capture.export.service import ExportService
from skyrl_capture.persistence import DiskActiveStore, DiskCommittedStore, ensure_record
from skyrl_capture.persistence.layout import export_artifacts_dir, export_jobs_dir
from skyrl_capture.reader.records import RecordReader
from skyrl_capture.transport.http import UpstreamTransport
from skyrl_capture.upstream.plugins import load_modules
from skyrl_capture.writer.commits import CommitCoordinator
from skyrl_capture.writer.registry import TrajectoryRegistry
from skyrl_capture.writer.sweep import FinishSweeper

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Own the resources and background tasks used by one capture application."""

    config: Config
    active: DiskActiveStore
    committed: DiskCommittedStore
    registry: TrajectoryRegistry
    commits: CommitCoordinator
    transport: UpstreamTransport
    # The one capture proxy this process was built with: `TextProxy` or
    # `TitoProxy`. There is no field for the other one.
    proxy: Any
    data_plane: DataPlane
    commands: CaptureCommands
    health: HealthProvider
    lifecycle: Any
    clock_epoch: str
    # Present only when the viewer is enabled.
    reader: RecordReader | None = None
    exports: ExportService | None = None
    viewer: Any = None
    # Background work the selected proxy wants running, if any. Token capture
    # samples event-loop lag; text capture asks for nothing.
    proxy_tasks: tuple[Callable[[], Any], ...] = ()
    sweeper: Any = None
    _running: list[asyncio.Task[None]] = field(default_factory=list, repr=False)

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self.reader is not None:
            self.reader.start()
        if self.exports is not None:
            self.exports.start()
        self._running = [
            asyncio.create_task(factory(), name=f"proxy-task-{index}")
            for index, factory in enumerate(self.proxy_tasks)
        ]
        if self.sweeper is not None:
            self._running.append(
                asyncio.create_task(self.sweeper.run_forever(), name="finish-sweep")
            )

    async def stop(self) -> None:
        """Stop taking work, let what is in flight land, then let go.

        Unfinished journals are left exactly as they are. A trajectory nobody
        finished is not an error and is not garbage: it is a trajectory whose
        caller has not come back, and a replacement process picks it up from
        the record on the next request for it.

        One that was *asked* to finish is different, and a last sweep completes
        those: the caller made its request, and shutting down is no reason to
        leave it unanswered.
        """
        if self.sweeper is not None:
            with contextlib.suppress(Exception):
                await self.sweeper.sweep()
        for task in self._running:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._running = []
        if self.exports is not None:
            await self.exports.stop()
        if self.reader is not None:
            await self.reader.stop()
        # Pending commits and delivery confirmations drain before the journals
        # are closed: everything accepted is written down.
        await self.commits.stop()
        with contextlib.suppress(Exception):
            await self.active.close()
        with contextlib.suppress(Exception):
            await self.transport.aclose()


async def build_runtime(config: Config | None = None) -> Runtime:
    config = config or load_config()
    # Before anything resolves an upstream type, so a contributed one is
    # registered by the time the next line asks the registry about it.
    load_modules(config.upstream_modules, source="upstream_modules")
    # Fail here rather than on the first request: an unknown upstream type is a
    # startup mistake, and the process that reports it at startup is the one
    # that can be fixed by relaunching.
    config.upstream.validate()

    # -- resources -----------------------------------------------------------
    root = config.require_record_dir()
    # Creating the layout is safe to race: the manifest is linked into place,
    # so one process writes it and the rest check it.
    record = await asyncio.to_thread(
        ensure_record, root, upstream=config.upstream.provenance()
    )
    active = DiskActiveStore(
        record,
        fsync=config.record.fsync,
        fsync_interval=config.record.fsync_interval,
        compress=config.record.compress,
    )
    committed = DiskCommittedStore(record)
    registry = TrajectoryRegistry(active=active, committed=committed)
    # Built before the writers, so they can tell it what they changed. A
    # replica that serves reads for the record it is writing should not show a
    # run a sweep behind itself, and it is the one process that can know.
    reader = RecordReader(record) if config.viewer else None
    note_change = reader.note_change if reader is not None else None
    commits = CommitCoordinator(
        active, capacity=config.record.commit_capacity, on_change=note_change
    )
    transport = UpstreamTransport(config.proxy)
    clock_epoch = uuid.uuid4().hex

    # -- the capture path: one mode, chosen once -------------------------------
    proxy: Any
    proxy_tasks: tuple[Callable[[], Any], ...] = ()
    if isinstance(config.upstream, TitoUpstream):
        # Imported here, not at module scope: these reach the renderer, and a
        # text deployment must not import one. `tests/test_packaging.py` holds
        # that line.
        from skyrl_capture.tito.engine import TokenEngine
        from skyrl_capture.tito.proxy import TitoProxy
        from skyrl_capture.tito.sessions import TokenSessionManager
        from skyrl_capture.tito.trace_store import StoredTraces

        sessions = TokenSessionManager(
            upstream=config.upstream,
            trace_budget=config.proxy.token_trace_budget,
            source=StoredTraces(registry),
        )
        proxy = TitoProxy(
            engine=TokenEngine(transport),
            sessions=sessions,
            upstream=config.upstream,
            commits=commits,
            header_allowlist=config.capture_header_allowlist,
            clock_epoch=clock_epoch,
            commit_timeout=config.finish_grace_seconds,
        )
        # The trace is the one thing a trajectory leaves behind in proxy
        # memory, and eviction is what releases it.
        registry.on_evict(sessions.forget)
        proxy_tasks = (proxy.sample_loop_lag,)
    else:
        from skyrl_capture.text.proxy import TextProxy

        proxy = TextProxy(
            transport=transport,
            upstream=config.upstream,
            proxy=config.proxy,
            commits=commits,
            header_allowlist=config.capture_header_allowlist,
            clock_epoch=clock_epoch,
        )

    data_plane = DataPlane(
        registry=registry,
        proxy=proxy,
        max_request_bytes=config.proxy.max_request_bytes,
    )

    # -- the lifecycle ---------------------------------------------------------
    commands = CaptureCommands(
        registry=registry,
        commits=commits,
        committed=committed,
        config=config,
        on_change=note_change,
    )
    sweeper = (
        FinishSweeper(
            commands=commands,
            active=active,
            registry=registry,
            interval=config.finish_sweep_seconds,
        )
        if config.finish_sweep_seconds > 0
        else None
    )
    health = HealthProvider(
        data_plane=data_plane,
        proxy=proxy,
        registry=registry,
        commits=commits,
        active=active,
        committed=committed,
        mode=config.upstream.mode,
        record=str(record),
        clock_epoch=clock_epoch,
    )
    lifecycle = build_lifecycle_app(commands=commands, health=health)

    # -- the viewer, if this replica serves one ---------------------------------
    exports: ExportService | None = None
    viewer: Any = None
    if reader is not None:
        exports = ExportService(
            reader=reader,
            jobs=ExportJobStore(export_jobs_dir(record)),
            artifacts=ArtifactStore(export_artifacts_dir(record)),
            public_url=config.proxy.public_url,
        )
        viewer = build_viewer_app(reader=reader, exports=exports)

    return Runtime(
        config=config,
        active=active,
        committed=committed,
        registry=registry,
        commits=commits,
        transport=transport,
        proxy=proxy,
        data_plane=data_plane,
        commands=commands,
        health=health,
        lifecycle=lifecycle,
        clock_epoch=clock_epoch,
        reader=reader,
        exports=exports,
        viewer=viewer,
        proxy_tasks=proxy_tasks,
        sweeper=sweeper,
    )
