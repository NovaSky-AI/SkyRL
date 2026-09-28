"""The write side of the lifecycle: create, finish, and metadata.

Three operations, and only the middle one is interesting. `finish` is the
barrier the whole design turns on:

1. close the route, so no new turn is accepted;
2. append `FinishRequested`, so a crash here leaves evidence of what was asked;
3. wait for the turns already in flight;
4. wait for their commits;
5. read the trajectory's own journal back, because that is where the captured
   bytes are;
6. compile the canonical record and rename it into place atomically;
7. render the requested export from what was committed, not from memory;
8. evict the hot state and delete the journal.

Every one of those steps is resumable. Before the rename, a retry reconstructs
and writes again; after it, a retry finds the record and returns it. The finish
request's hash is persisted, so "is this the same finish?" is a question a
replacement process can answer.

Steps 3 and 4 are bounded, and reaching the bound means **not finishing**:
`503`, retryable, trajectory untouched. The alternative -- compiling anyway and
declaring the difference missing -- cannot be made correct. A turn awaiting its
own commit is both an open turn and a pending commit, so it counts twice; the
append it is waiting on keeps running past the timeout, so it may well land
*after* the count is taken and before the journal is read; and the record would
then contain the exchange and say it was lost. There is no number to put there,
because the thing being counted has not stopped happening.

A trajectory that cannot finish stays open, which is a state the design already
supports -- nothing expires one -- and a retry finishes it. After a restart the
retry is trivial: the hung turn is gone with the process that held it, and the
journal has everything.

All three operations take a per-trajectory lock. They interleave otherwise --
each one has awaits in the middle -- and an annotation that landed after
`finish` read its snapshot would be written to a journal that is about to be
deleted.

There is no delete, no run mutation and no expiry sweep. A trajectory ends
because a caller finished it or because token capture poisoned it; nothing ends
one on a clock, and a journal nobody finished stays on disk until somebody
does.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from skyrl_capture.config import Config
from skyrl_capture.domain.hashing import canonical_hash
from skyrl_capture.domain.models import (
    CreatedTrajectory,
    normalize_export_format,
    route_base_url,
    validate_identifier,
)
from skyrl_capture.domain.records import (
    ActiveTrajectory,
    TrajectoryConflict,
    TrajectoryError,
    TrajectoryHeader,
    TrajectoryRecord,
    metadata_update,
    now,
)
from skyrl_capture.export.service import render_records
from skyrl_capture.export.view import view_of
from skyrl_capture.persistence.committed import CommittedStore
from skyrl_capture.persistence.journal import FinishRequested, MetadataUpdated
from skyrl_capture.writer.commits import CommitCoordinator
from skyrl_capture.writer.compile import compile_record, journal_aggregate
from skyrl_capture.writer.registry import TrajectoryRegistry

logger = logging.getLogger(__name__)


class _PerTrajectory:
    """One lock per trajectory, held only while somebody is using it.

    Reference-counted rather than left in a map, because a long run names
    hundreds of thousands of trajectories and a lock each would be a leak --
    and rather than dropped on release, because a caller already waiting on
    one would then be joined by the next caller on a *different* lock, which
    is the interleaving the lock exists to prevent.
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._users: dict[str, int] = {}

    @asynccontextmanager
    async def __call__(self, trajectory_id: str) -> AsyncIterator[None]:
        lock = self._locks.get(trajectory_id)
        if lock is None:
            lock = self._locks[trajectory_id] = asyncio.Lock()
        self._users[trajectory_id] = self._users.get(trajectory_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._users[trajectory_id] - 1
            if remaining:
                self._users[trajectory_id] = remaining
            else:
                del self._users[trajectory_id]
                del self._locks[trajectory_id]

    def held(self) -> int:
        return len(self._locks)


class PersistenceUnavailable(Exception):
    """The record could not be written. The caller should retry.

    Distinct from every other failure here because it is the one the SDK
    retries: nothing was decided wrongly, the disk did not take it.
    """


class FinishNotSettled(PersistenceUnavailable):
    """A barrier ran out of grace, so there is nothing correct to compile yet.

    Retryable for the same reason: the trajectory is intact, something is
    still writing to it, and a moment later it will not be. What this
    deliberately is *not* is a finished record with a guess about what it is
    missing -- see `_settled`.
    """


@dataclass(frozen=True, slots=True)
class FinishEnvelope:
    id: str
    status: str
    format: str
    records: list[dict[str, Any]]

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "format": self.format.replace("_", "-"),
            "records": self.records,
        }


def finish_request_hash(
    *, labels: list[str] | None, annotations: dict[str, Any] | None, command_result: str | None
) -> str:
    """What makes one finish the same finish as another.

    The export format and its options are deliberately excluded. They decide
    how the committed record is rendered for this reply, not what the record
    is, so asking for the same trajectory in a second format is a retry rather
    than a conflict.
    """
    return canonical_hash(
        {
            "labels": sorted(labels or []),
            "annotations": annotations or {},
            "command_result": command_result,
        }
    )


class CaptureCommands:
    def __init__(
        self,
        *,
        registry: TrajectoryRegistry,
        commits: CommitCoordinator,
        committed: CommittedStore,
        config: Config,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        self._registry = registry
        self._commits = commits
        self._committed = committed
        self._config = config
        # A reader in this process, if there is one. See `CommitCoordinator`.
        self._on_change = on_change or (lambda trajectory_id: None)
        self._active = registry.active_store
        self._cleanup: set[asyncio.Task[None]] = set()
        # One lifecycle operation per trajectory at a time. Not a data-plane
        # lock: turns are serialized by the session lock in token mode and by
        # nothing in text mode, and neither needs this.
        self._locks = _PerTrajectory()

    def _lock(self, trajectory_id: str) -> Any:
        return self._locks(trajectory_id)

    # -- create ----------------------------------------------------------------
    async def create(
        self,
        *,
        trajectory_id: str,
        project: str,
        run_id: str | None,
        task_id: str | None,
        step: int | None,
        labels: list[str] | None,
        annotations: dict[str, Any] | None,
        bodies: str,
        source_metadata: dict[str, Any] | None,
        request_hash: str,
    ) -> CreatedTrajectory:
        """Persist one trajectory and hand back the route to send it to.

        The id is the caller's. It is the first segment of every route this
        trajectory serves, the name of its files, and the session key sent to
        the inference engine -- one name for one trial, chosen by the side that
        knows what the trial is. Creation is durable before this returns: a
        route for a trajectory that is not on disk is a route no replacement
        process could serve.
        """
        upstream = self._config.upstream
        identifier = validate_identifier(trajectory_id)
        header = TrajectoryHeader(
            id=identifier,
            project=project,
            run_id=run_id,
            task_id=task_id,
            step=step,
            mode=upstream.mode,
            upstream=upstream.provenance(),
            labels=tuple(labels or ()),
            annotations=dict(annotations or {}),
            bodies=bodies,
            source_metadata=dict(source_metadata or {}),
            created_at=now(),
            create_request_hash=request_hash,
        )
        async with self._lock(identifier):
            try:
                active = await self._registry.create(header)
            except OSError as error:
                raise PersistenceUnavailable(
                    f"could not persist trajectory {identifier!r}: {error}"
                ) from error
        self._on_change(active.id)
        return CreatedTrajectory(
            id=active.id,
            base_url=route_base_url(
                self._config.proxy.public_url, active.id, upstream.client_suffix
            ),
            mode=active.header.mode,
            protocol=active.header.upstream["protocol"],
            status=active.status,
        )

    # -- finish ------------------------------------------------------------------
    async def finish(
        self,
        trajectory_id: str,
        *,
        labels: list[str] | None,
        annotations: dict[str, Any] | None,
        command_result: str | None,
        export_format: str = "graph",
        options: dict[str, Any] | None = None,
    ) -> FinishEnvelope:
        """Close, settle, compile, commit, render. Raises `TrajectoryError`
        for an unknown trajectory and `TrajectoryConflict` for a finish that
        contradicts one already recorded."""
        fmt = normalize_export_format(export_format)
        options = dict(options or {})
        request_hash = finish_request_hash(
            labels=labels, annotations=annotations, command_result=command_result
        )
        update = metadata_update(annotations=annotations, labels=labels)

        async with self._lock(trajectory_id):
            existing = await self._committed.get(trajectory_id)
            if existing is not None:
                return self._render(existing, fmt, options, request_hash)

            active = await self._registry.resolve(trajectory_id)
            if active is None:
                raise TrajectoryError(f"unknown trajectory {trajectory_id!r}")
            if active.finish_request_hash and active.finish_request_hash != request_hash:
                raise TrajectoryConflict(
                    f"trajectory {trajectory_id!r} is already finishing with a different "
                    "outcome; a finish names one result"
                )

            # 1-2. Close the route, and say so on disk before anything else.
            #      The next inference request reads `finalizing` and gets 410.
            #
            #      Skipped when this exact finish has already been recorded --
            #      a retry after the commit failed, or after a restart that
            #      replayed the request from the journal. Doing it twice would
            #      apply the same metadata twice and count a second revision
            #      for a correction nobody made.
            if not self._already_requested(active, request_hash):
                at = now()
                active.apply_metadata(update)
                active.request_finish(
                    command_result=command_result, request_hash=request_hash, at=at
                )
                try:
                    await self._active.append(
                        trajectory_id,
                        FinishRequested(
                            update=update,
                            command_result=command_result,
                            request_hash=request_hash,
                            at=at,
                        ),
                    )
                except OSError as error:
                    raise PersistenceUnavailable(
                        f"could not record the finish of {trajectory_id!r}: {error}"
                    ) from error

            # 3-4. The turns already accepted, and the commits they queued.
            #      Either one outliving the grace means there is nothing
            #      correct to compile yet, so this does not compile.
            await self._settled(active, trajectory_id)

            # 5-7. Compile from the journal, commit atomically, render from
            #      what was committed rather than from memory.
            record = await self._compile(active, request_hash)
            try:
                await self._committed.put(record)
            except OSError as error:
                raise PersistenceUnavailable(
                    f"could not commit the record for {trajectory_id!r}: {error}"
                ) from error

            # 8. Hot state goes, then the journal. In that order: a reader that
            #    catches the gap prefers the committed record anyway.
            self._registry.evict(trajectory_id)
            self._forget_journal(trajectory_id)
            self._on_change(trajectory_id)
            return self._render(record, fmt, options, request_hash)

    @staticmethod
    def _already_requested(active: ActiveTrajectory, request_hash: str) -> bool:
        """Whether this exact finish has already been recorded on this trajectory.

        The hash has to match, not merely be present: a trajectory marked
        `finalizing` without one -- by a poison, or by a turn that raced the
        gate -- has not had *this* finish's metadata applied, and skipping
        would drop it.
        """
        return bool(
            active.finish_requested_at is not None
            and active.finish_request_hash
            and active.finish_request_hash == request_hash
        )

    async def _settled(self, active: ActiveTrajectory, trajectory_id: str) -> None:
        """Wait for what this trajectory still owes. Raise if it still owes it.

        Both waits are bounded by `finish_grace_seconds`, and running out is
        not a state a record can be compiled from. Counting the difference as
        missing looks tempting and is wrong three ways over: a turn awaiting
        its own commit is both an open turn and a pending commit, so it counts
        twice; the append keeps running after the timeout, so it may land
        before the journal is read; and the record would then hold the exchange
        and declare it lost. The honest answer is that finishing has not
        happened yet.
        """
        grace = self._config.finish_grace_seconds
        if not await active.settle_turns(timeout=grace):
            raise FinishNotSettled(
                f"{active.in_flight} turn(s) are still in flight on {trajectory_id!r} after "
                f"{grace:.1f}s, so its record cannot be compiled yet; retry"
            )
        if not await self._commits.settle(active, timeout=grace):
            raise FinishNotSettled(
                f"a commit for {trajectory_id!r} was still queued after {grace:.1f}s, so its "
                "record cannot be compiled yet; retry"
            )
        # Gaps from earlier failures, now that nothing else is writing. Bounded
        # and best-effort: the in-memory count reaches the record either way.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._commits.flush_gaps(active), grace)

    async def _compile(self, active: ActiveTrajectory, request_hash: str) -> TrajectoryRecord:
        """Rebuild from the journal, merge what memory knows, compile."""
        records = await self._active.records(active.id)
        source = journal_aggregate(records, hot=active) or active
        status = "poisoned" if active.status == "poisoned" else "finished"
        return compile_record(
            source, status=status, finished_at=now(), finish_request_hash=request_hash
        )

    def _forget_journal(self, trajectory_id: str) -> None:
        """Delete the journal behind the reply. The committed record is visible,
        so the journal is redundant and its removal is not on the caller's path."""

        async def remove() -> None:
            with contextlib.suppress(Exception):
                await self._active.remove(trajectory_id)

        task = asyncio.create_task(remove(), name=f"drop-journal-{trajectory_id}")
        self._cleanup.add(task)
        task.add_done_callback(self._cleanup.discard)

    def _render(
        self,
        record: TrajectoryRecord,
        export_format: str,
        options: dict[str, Any],
        request_hash: str,
    ) -> FinishEnvelope:
        if record.finish_request_hash and record.finish_request_hash != request_hash:
            raise TrajectoryConflict(
                f"trajectory {record.id!r} has already finished with a different outcome; "
                "its record is committed and a finish names one result"
            )
        return FinishEnvelope(
            id=record.id,
            status=record.trajectory.status,
            format=export_format,
            records=render_records(
                view_of(record), export_format=export_format, options=options
            ),
        )

    # -- metadata -------------------------------------------------------------------
    async def annotate(
        self,
        trajectory_id: str,
        *,
        annotations: dict[str, Any] | None,
        remove_annotations: list[str] | None,
        labels: list[str] | None,
        remove_labels: list[str] | None,
    ) -> dict[str, Any]:
        """Merge labels and annotations, whether the trajectory is hot or done.

        Nothing is sealed by finishing: a reward can arrive long afterwards. An
        active trajectory takes it as one more journal record; a committed one
        is loaded, edited, given a new revision and replaced atomically. Exports
        already rendered are historical snapshots and are not rewritten.
        """
        update = metadata_update(
            annotations=annotations,
            remove_annotations=remove_annotations,
            labels=labels,
            remove_labels=remove_labels,
        )
        async with self._lock(trajectory_id):
            active = await self._registry.resolve(trajectory_id)
            if active is not None:
                active.apply_metadata(update)
                await self._active.append(trajectory_id, MetadataUpdated(update, now()))
                self._on_change(trajectory_id)
                return {
                    "labels": sorted(active.labels),
                    "annotations": dict(active.annotations),
                }
            try:
                record = await self._committed.update_metadata(trajectory_id, update)
            except KeyError:
                raise TrajectoryError(f"unknown trajectory {trajectory_id!r}") from None
            self._on_change(trajectory_id)
            return {
                "labels": list(record.trajectory.labels),
                "annotations": dict(record.trajectory.annotations),
                "revision": record.revision,
            }

    # -- what the test suite and the CLI settle on ------------------------------------
    async def drain(self) -> None:
        """Wait until everything submitted so far is durable."""
        for trajectory_id in self._registry.hot_ids():
            active = self._registry.hot(trajectory_id)
            if active is not None:
                await self._commits.settle(active)
