"""`RecordReader`: one reader over active journals and committed records.

There is no live reader and no offline reader. A trajectory is somewhere in
the record directory -- mid-run in `active/`, finished in `committed/` -- and
this answers for both:

    get(identifier) == committed.get(identifier) or active.replay(identifier)

Committed wins wherever both exist, which is what makes the moment of
finishing invisible to a reader: for the instant between the committed rename
and the journal's deletion the trajectory is in two places, and it is the same
trajectory in both.

Indexing is progressive. A record directory with a hundred thousand
trajectories in it must not delay the first page by the time it takes to read
all of them, so the API starts answering immediately, discovery runs behind
it, and a listing says how far it has got. Until it is finished, `total` is
null and pagination is provisional -- a page boundary can move as records are
discovered, and a caller that wants a stable count waits for `indexing` to go
false.

What the index keeps is summaries. Graphs and payloads are read on demand and
held in a small LRU, so the memory a viewer needs is a function of what is
being looked at rather than of how much was captured.

One indexer per record root. Several browsers may talk to it; two of them must
not each scan the same directory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from skyrl_capture.domain import timing
from skyrl_capture.domain.records import (
    CaptureIntegrity,
    TrajectoryDocument,
    TrajectoryHeader,
    TrajectoryRecord,
)
from skyrl_capture.export.view import TrajectoryView, view_of, view_of_active
from skyrl_capture.persistence.active import DiskActiveStore, read_journal, rebuild
from skyrl_capture.persistence.committed import DiskCommittedStore
from skyrl_capture.persistence.headers import DiskHeaderIndex, header_path, read_header
from skyrl_capture.persistence.journal import (
    ActiveRecord,
    CaptureGap,
    ExchangeCommitted,
    ExchangeDeliveryConfirmed,
    FinishRequested,
    MetadataUpdated,
    TrajectoryCreated,
    TrajectoryPoisoned,
)
from skyrl_capture.persistence.layout import (
    JOURNAL_SUFFIX,
    active_paths,
    open_record,
    trajectory_id_of,
)

logger = logging.getLogger(__name__)


# -- the summary an index keeps ---------------------------------------------------
@dataclass
class JournalSummary:
    """What an active journal says, without keeping its graph or its bytes.

    Node counts are summed from each delta rather than from a graph, which is
    exact: a delta carries only the nodes its exchange introduced, and an
    exchange appears once in a journal.
    """

    header: TrajectoryHeader
    offset: int = 0
    status: str = "created"
    labels: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)
    exchange_count: int = 0
    node_count: int = 0
    calls_after_close: int = 0
    integrity: CaptureIntegrity = field(default_factory=CaptureIntegrity)
    first_event_at: datetime | None = None
    finish_requested_at: datetime | None = None
    finished_at: datetime | None = None
    command_result: str | None = None
    revision: int = 0
    seen_exchange_ids: set[str] = field(default_factory=set)

    @classmethod
    def start(cls, header: TrajectoryHeader) -> JournalSummary:
        return cls(
            header=header,
            labels=list(header.labels),
            annotations=dict(header.annotations),
        )

    def apply(self, record: ActiveRecord) -> None:
        if isinstance(record, ExchangeCommitted):
            if record.exchange.id in self.seen_exchange_ids:
                return
            self.seen_exchange_ids.add(record.exchange.id)
            self.exchange_count += 1
            if record.graph is not None:
                self.node_count += len(record.graph.nodes)
            if record.exchange.row.get("late"):
                self.calls_after_close += 1
            if self.status == "created":
                self.status = "active"
            self.first_event_at = self.first_event_at or record.exchange.row.get(
                "request_start_at"
            ) or record.at
        elif isinstance(record, CaptureGap):
            # Zero means uncertainty rather than loss; see `record_gap`.
            if record.count <= 0:
                self.integrity.recovery_uncertain = True
            else:
                self.integrity.calls_missing += record.count
                self.integrity.complete = False
            if record.reason and record.reason not in self.integrity.errors:
                self.integrity.errors.append(record.reason)
        elif isinstance(record, MetadataUpdated):
            if not record.update.empty():
                self.labels, self.annotations = record.update.apply(self.labels, self.annotations)
                self.revision += 1
        elif isinstance(record, FinishRequested):
            if not record.update.empty():
                self.labels, self.annotations = record.update.apply(self.labels, self.annotations)
                self.revision += 1
            self.command_result = record.command_result or self.command_result
            self.finish_requested_at = self.finish_requested_at or record.at
            self.status = "finalizing"
        elif isinstance(record, TrajectoryPoisoned):
            # `complete` is untouched: see `ActiveTrajectory.poison`.
            self.status = "poisoned"
            self.finished_at = self.finished_at or record.at
            if record.reason not in self.integrity.errors:
                self.integrity.errors.append(record.reason)
        elif isinstance(record, (TrajectoryCreated, ExchangeDeliveryConfirmed)):
            # The header built this summary, and a delivery confirmation says
            # nothing a reader of an in-progress journal can act on: an
            # exchange awaiting confirmation is a response still being sent,
            # not an uncertain one. Uncertainty is decided by the process that
            # recovers the journal, not by one looking at it.
            pass

    def document(self) -> TrajectoryDocument:
        header = self.header
        return TrajectoryDocument(
            id=header.id,
            project=header.project,
            run_id=header.run_id,
            task_id=header.task_id,
            step=header.step,
            upstream_snapshot=dict(header.upstream),
            mode=header.mode,
            status=self.status,
            labels=sorted(self.labels),
            annotations=dict(self.annotations),
            bodies=header.bodies,
            created_at=header.created_at,
            first_event_at=self.first_event_at,
            finish_requested_at=self.finish_requested_at,
            finished_at=self.finished_at,
            exchange_count=self.exchange_count,
            node_count=self.node_count,
            calls_after_close=self.calls_after_close,
            integrity=self.integrity.copy(),
            command_result=self.command_result,
            source_metadata=dict(header.source_metadata),
            revision=self.revision,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryQuery:
    project: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    step: int | None = None
    status: str | None = None
    limit: int = 50
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryPage:
    items: list[dict[str, Any]]
    next_cursor: str | None
    #: Null while indexing: a total that changes under the caller is worse
    #: than no total, because a pager renders it as fact.
    total: int | None
    indexing: bool
    indexed_trajectories: int


class RecordReader:
    """The one reader. Committed first, active second, indexed in the background."""

    def __init__(
        self,
        root: str | Path,
        *,
        detail_cache: int = 64,
        batch: int = 256,
    ) -> None:
        self.root = open_record(root)
        self._active = DiskActiveStore(self.root)
        self._committed = DiskCommittedStore(self.root)
        self._headers_index = DiskHeaderIndex(self.root)
        self._batch = batch
        self._detail_cache = detail_cache
        # id -> summary document. Committed entries shadow active ones.
        self._committed_index: dict[str, TrajectoryDocument] = {}
        self._active_index: dict[str, JournalSummary] = {}
        # What each committed file looked like when it was read: its size and
        # modification time. A committed record is written once *and rewritten*
        # -- a reward that arrives after a trajectory finished replaces it with
        # a new revision -- so an index keyed on the id alone would skip that
        # file for ever. In one process the writer says what it changed; a
        # standalone viewer over a shared volume has only the filesystem, and
        # this is what it reads.
        self._committed_seen: dict[str, tuple[int, int]] = {}
        # path -> the header read from it.
        self._headers: dict[str, TrajectoryDocument] = {}
        self._details: OrderedDict[str, TrajectoryView] = OrderedDict()
        # Trajectories a writer in *this* process has just changed. Not a
        # change feed and not a subscription: a capture replica that also
        # serves reads knows which file it just wrote, and saying so is
        # cheaper and more accurate than waiting for the next sweep. A
        # standalone viewer never gets one of these and relies on the sweep,
        # which is the only difference between the two.
        self._dirty: set[str] = set()
        # The merged, sorted documents a listing pages through, rebuilt only
        # when the index changes. A record directory is meant to hold a
        # hundred thousand trajectories, and rebuilding every document to
        # answer a page of fifty is the kind of cost that does not show up
        # until somebody has one.
        self._snapshot: list[TrajectoryDocument] | None = None
        self._indexing = True
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.scans = 0

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        self._task = asyncio.create_task(self._index_forever(), name="record-indexer")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _index_forever(self) -> None:
        """One pass to build the index, then a slow pass to keep it current.

        The first pass yields between batches so the API answers while it runs.
        After it, refreshing is what a viewer asks for explicitly; this loop is
        only the backstop for a directory another process is writing.
        """
        try:
            await self.refresh()
        except Exception:
            logger.exception("initial record index failed")
            self._indexing = False
        while True:
            await asyncio.sleep(2.0)
            try:
                await self.refresh()
            except Exception:
                logger.exception("record index refresh failed")

    async def ensure_indexed(self) -> None:
        """Do not answer until the record directory has been read through once.

        What a bulk export needs and a listing does not. A page that is a
        little behind is a page; a training set that quietly omitted the
        trajectories the scan had not reached yet is a training set that is
        wrong in a way nothing downstream can detect.
        """
        await self.apply_pending()
        if self._indexing:
            await self.refresh()

    async def refresh(self) -> None:
        """Pick up new committed records and new or grown active journals.

        Both directions matter. A trajectory that finished between two reads
        moves from `active/` to `committed/`, and a viewer that refreshed only
        one of them would show it twice or lose it for a moment.

        Files are read in worker threads, a batch at a time, and the index is
        changed on the event loop. That is what makes the first scan
        progressive -- the API answers between batches -- and it is what keeps
        the index from being mutated by a thread while a listing walks it.
        """
        async with self._lock:
            await self._refresh_committed()
            await self._refresh_active()
            self.scans += 1
            # A completed pass is a completed pass, whoever asked for it. After
            # one, `total` is a real number and pagination is stable.
            self._indexing = False

    async def _refresh_committed(self) -> None:
        """Read the listing from the headers, not from the records.

        A header is the trajectory document, uncompressed, beside the record
        it describes. Reading one costs a small `read` where reading the
        record costs a decompression of every exchange, every node and, in
        tokens mode, every token array the trajectory holds -- to use a dozen
        fields off the front of it. That is the whole reason headers exist,
        and it is what lets a cold viewer over a large directory open at all.
        """
        present = await asyncio.to_thread(self._headers_index.paths)
        keys = {str(path) for path in present}

        vanished = [key for key in self._headers if key not in keys]
        for stale in vanished:
            # A header that is gone names a record that is gone.
            del self._headers[stale]
            self._committed_seen.pop(stale, None)
        if vanished:
            # The only case that can *remove* a row, so the only one that
            # cannot be folded in one document at a time.
            self._rebuild_committed_index()

        pending = await asyncio.to_thread(self._headers_todo, present)
        for index in range(0, len(pending), self._batch):
            loaded = await asyncio.to_thread(
                self._load_headers, pending[index : index + self._batch]
            )
            for key, stamp, document in loaded:
                self._committed_seen[key] = stamp
                if document is not None:
                    self._headers[key] = document
                    self._admit(document)
                    # Rewritten, so whatever was cached for it is stale.
                    self._details.pop(document.id, None)
            if loaded:
                self._touch_index()

    def _admit(self, document: TrajectoryDocument) -> None:
        """Fold one header into the listing.

        Per document rather than per pass, because the first scan of a large
        directory has to be answerable while it is still running: a listing
        that showed nothing until the last batch landed would make progressive
        indexing progressive in name only.

        A trajectory has one header path, addressed by its id, so this is not
        resolving a conflict between two of them. The revision guard is for
        ordering: a header is rewritten in place when a reward lands after the
        trajectory finished, and a read that returned the older bytes must not
        undo a newer revision already folded in. The next pass reconciles it.
        """
        seen = self._committed_index.get(document.id)
        if seen is None or document.revision >= seen.revision:
            self._committed_index[document.id] = document
        # Finished: whatever the journal said about it is now history.
        self._active_index.pop(document.id, None)

    def _rebuild_committed_index(self) -> None:
        """Collapse every header to one row per trajectory, from scratch.

        The listing is cached until the index is touched, so this says it has
        changed. Rebuilding is the only thing that can *remove* a row, and a
        removal that left the cached listing alone would keep serving a
        trajectory whose files are gone.
        """
        self._committed_index = {}
        for document in self._headers.values():
            self._admit(document)
        self._touch_index()

    def _headers_todo(self, paths: list[Path]) -> list[Path]:
        """Headers this index has not read, or has read an older version of.

        Compared by size and modification time, which is what a reader on the
        other side of a shared volume has to go on. A header is rewritten when
        a reward arrives after the trajectory finished, so an index that read
        each path once would never see that revision.
        """
        todo: list[Path] = []
        for path in paths:
            try:
                status = path.stat()
            except OSError:
                continue
            if self._committed_seen.get(str(path)) != (status.st_mtime_ns, status.st_size):
                todo.append(path)
        return todo

    def _load_headers(
        self, paths: list[Path]
    ) -> list[tuple[str, tuple[int, int], TrajectoryDocument | None]]:
        loaded: list[tuple[str, tuple[int, int], TrajectoryDocument | None]] = []
        for path in paths:
            try:
                # Stat before the read, so a rewrite that lands between the two
                # is picked up by the next pass rather than recorded as seen.
                status = path.stat()
                stamp = (status.st_mtime_ns, status.st_size)
            except OSError:
                continue
            document = read_header(path)
            if document is None:
                logger.warning("skipping unreadable run header %s", path)
            loaded.append((str(path), stamp, document))
        return loaded

    async def _refresh_active(self) -> None:
        paths = await asyncio.to_thread(active_paths, self.root)
        present: set[str] = set()
        for index in range(0, len(paths), self._batch):
            batch = paths[index : index + self._batch]
            offsets = {
                trajectory_id_of(path, JOURNAL_SUFFIX): summary.offset
                for path in batch
                if (summary := self._active_index.get(trajectory_id_of(path, JOURNAL_SUFFIX)))
            }
            scans = await asyncio.to_thread(self._read_active, batch, offsets)
            for identifier, scan in scans:
                present.add(identifier)
                self._apply_active(identifier, scan)
        for identifier in list(self._active_index):
            if identifier not in present:
                self._active_index.pop(identifier, None)
                self._touch_index()

    def _read_active(
        self, paths: list[Path], offsets: dict[str, int]
    ) -> list[tuple[str, Any]]:
        scans: list[tuple[str, Any]] = []
        for path in paths:
            identifier = trajectory_id_of(path, JOURNAL_SUFFIX)
            try:
                scans.append((identifier, read_journal(path, start=offsets.get(identifier, 0))))
            except Exception as error:
                logger.warning("skipping unreadable journal %s: %s", path, error)
                scans.append((identifier, None))
        return scans

    def _apply_active(self, identifier: str, scan: Any) -> None:
        if scan is None or identifier in self._committed_index:
            return
        summary = self._active_index.get(identifier)
        records = scan.records
        if summary is None:
            if not records or not isinstance(records[0], TrajectoryCreated):
                return
            summary = JournalSummary.start(records[0].header)
            self._active_index[identifier] = summary
            self._touch_index()
            records = records[1:]
        for record in records:
            summary.apply(record)
        summary.offset = scan.consumed
        if records:
            # Whatever was cached for this trajectory is now short of the file.
            self._details.pop(identifier, None)
            self._touch_index()

    def note_change(self, trajectory_id: str) -> None:
        """A local writer changed this trajectory's files. No I/O here."""
        self._dirty.add(trajectory_id)
        self._details.pop(trajectory_id, None)

    def _touch_index(self) -> None:
        """The index changed, so the listing snapshot is stale."""
        self._snapshot = None

    async def apply_pending(self) -> None:
        """Re-read what a local writer said it changed. Usually one file."""
        if not self._dirty:
            return
        async with self._lock:
            pending, self._dirty = self._dirty, set()
            for identifier in sorted(pending):
                await self._reindex(identifier)

    async def _reindex(self, trajectory_id: str) -> None:
        # Committed first: a trajectory that finished is no longer whatever its
        # journal last said, and for a moment both files exist.
        record = await self._committed.get(trajectory_id)
        if record is not None:
            # Recorded against the header's path, exactly as a scan would
            # record it. `_headers` is what the listing is rebuilt from, so a
            # row that reached the index by this path and not by that one
            # would be invisible to a rebuild -- dropped if it is still there,
            # and kept for ever once it is not.
            path = header_path(self.root, trajectory_id)
            key = str(path)
            try:
                status = path.stat()
                self._committed_seen[key] = (status.st_mtime_ns, status.st_size)
            except OSError:
                # The header is written just after the record, so this can be
                # a moment early. Leaving it unseen makes the next scan read
                # it, which is the right answer either way.
                self._committed_seen.pop(key, None)
            self._headers[key] = record.trajectory
            self._admit(record.trajectory)
            self._details.pop(trajectory_id, None)
            self._touch_index()
            return
        path = self._active.path_for(trajectory_id)
        summary = self._active_index.get(trajectory_id)
        scan = await asyncio.to_thread(
            read_journal, path, start=summary.offset if summary else 0
        )
        self._apply_active(trajectory_id, scan)
        if not path.exists():
            self._active_index.pop(trajectory_id, None)
            self._touch_index()

    # -- source ------------------------------------------------------------------------
    def source_info(self) -> dict[str, Any]:
        return {
            "source": "record",
            "record": str(self.root),
            "indexing": self._indexing,
            "indexed_trajectories": len(self._committed_index) + len(self._active_index),
            "committed": len(self._committed_index),
            "active": len(self._active_index),
        }

    # -- trajectories -------------------------------------------------------------------
    def _documents(self) -> list[TrajectoryDocument]:
        """Every indexed trajectory, committed shadowing active, newest first.

        Ids are sortable by generation time, so lexical descending is
        newest-first -- which is also the order a cursor pages through.

        Cached until the index changes: a page of fifty out of a hundred
        thousand should not cost a hundred thousand documents.
        """
        if self._snapshot is None:
            merged: dict[str, TrajectoryDocument] = {
                identifier: summary.document()
                for identifier, summary in self._active_index.items()
            }
            merged.update(self._committed_index)
            self._snapshot = sorted(
                merged.values(), key=lambda document: document.id, reverse=True
            )
        return self._snapshot

    def list_trajectories(self, query: TrajectoryQuery) -> TrajectoryPage:
        wanted = {
            "project": query.project,
            "run_id": query.run_id,
            "task_id": query.task_id,
            "step": query.step,
            "status": query.status,
        }
        rows = [
            document
            for document in self._documents()
            if all(
                value is None or value == "" or getattr(document, name) == value
                for name, value in wanted.items()
            )
        ]
        total = len(rows)
        if query.cursor:
            rows = [row for row in rows if row.id < query.cursor]
        page = rows[: query.limit]
        next_cursor = page[-1].id if len(rows) > query.limit and page else None
        return TrajectoryPage(
            items=[document.public() for document in page],
            next_cursor=next_cursor,
            # Provisional totals are worse than none: a pager prints them.
            total=None if self._indexing else total,
            indexing=self._indexing,
            indexed_trajectories=len(self._committed_index) + len(self._active_index),
        )

    async def get_trajectory(self, trajectory_id: str) -> dict[str, Any] | None:
        view = await self.view(trajectory_id)
        if view is None:
            return None
        payload = view.trajectory.public()
        payload["gap_distribution"] = timing.gap_distribution(view.exchanges)
        return payload

    async def trajectory_exists(self, trajectory_id: str) -> bool:
        if trajectory_id in self._committed_index or trajectory_id in self._active_index:
            return True
        return await self.view(trajectory_id) is not None

    async def record(self, trajectory_id: str) -> TrajectoryRecord | None:
        return await self._committed.get(trajectory_id)

    async def view(self, trajectory_id: str) -> TrajectoryView | None:
        """The whole trajectory: committed if it is finished, replayed if not."""
        cached = self._details.get(trajectory_id)
        if cached is not None:
            self._details.move_to_end(trajectory_id)
            return cached
        record = await self._committed.get(trajectory_id)
        if record is not None:
            view = view_of(record)
        else:
            active = await asyncio.to_thread(self._replay_active, trajectory_id)
            if active is None:
                return None
            view = view_of_active(active)
        self._details[trajectory_id] = view
        while len(self._details) > self._detail_cache:
            self._details.popitem(last=False)
        return view

    def _replay_active(self, trajectory_id: str) -> Any:
        scan = read_journal(self._active.path_for(trajectory_id))
        if not scan.records:
            return None
        return rebuild(scan.records, recovered=False)

    def forget(self, trajectory_id: str) -> None:
        """Drop a cached detail. What a writer calls when it changes one."""
        self._details.pop(trajectory_id, None)

    async def finished_trajectory_ids(
        self, *, project: str | None, run_id: str | None
    ) -> list[str]:
        """Committed trajectories in the scope, ordered so an export of the
        same snapshot is byte-identical.

        Committed only. A bulk export selects finished work; an active
        trajectory's rows would change under the job that was reading them.
        """
        await self.apply_pending()
        return sorted(
            document.id
            for document in self._committed_index.values()
            if (project is None or document.project == project)
            and (run_id is None or document.run_id == run_id)
            and document.status == "finished"
        )

    async def modes_for(self, trajectory_ids: list[str]) -> dict[str, int]:
        """How many of these were captured in each mode.

        Falls back to reading the trajectory when the index has not reached it
        yet, because the answer decides whether an export is refused -- and
        refusing one because the indexer was a moment behind would be a
        confusing way to be wrong.
        """
        counts: dict[str, int] = {}
        for identifier in trajectory_ids:
            document = self._committed_index.get(identifier)
            if document is None:
                summary = self._active_index.get(identifier)
                document = summary.document() if summary else None
            if document is None:
                view = await self.view(identifier)
                document = view.trajectory if view is not None else None
            if document is not None:
                counts[document.mode] = counts.get(document.mode, 0) + 1
        return counts

    async def earliest_request(self, trajectory_ids: list[str]) -> Any:
        starts = []
        for identifier in trajectory_ids:
            view = await self.view(identifier)
            if view is None:
                continue
            starts.extend(
                row["request_start_at"]
                for row in view.exchanges
                if row.get("request_start_at") is not None
            )
        return min(starts) if starts else None

    # -- exchanges and graph ------------------------------------------------------------
    async def list_exchanges(
        self,
        trajectory_id: str,
        *,
        limit: int,
        cursor: str | None,
        provider: str | None,
        model: str | None,
        status: int | None,
        retry_attempt: int | None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        view = await self.view(trajectory_id)
        if view is None:
            return [], None
        wanted = {
            "provider": provider,
            "model": model,
            "http_status": status,
            "retry_attempt": retry_attempt,
        }
        rows = [
            row
            for row in view.exchanges
            if all(value is None or row.get(name) == value for name, value in wanted.items())
            and (cursor is None or row["id"] > cursor)
        ]
        page = rows[:limit]
        next_cursor = page[-1]["id"] if len(rows) > limit and page else None
        return [exchange_public(row) for row in page], next_cursor

    async def get_graph(self, trajectory_id: str) -> dict[str, Any]:
        """The `/v1/trajectories/{id}/graph` document.

        Every question here is one the graph itself answers, asked of a graph
        rebuilt from whichever form the trajectory is in. Nothing is derived a
        second way, so a fork on screen is the fork an export sees.
        """
        view = await self.view(trajectory_id)
        if view is None:
            return {
                "trajectory": trajectory_id,
                "nodes": [],
                "leaf_node_ids": [],
                "leaf_assistant_node_ids": [],
                "branch_points": [],
            }
        graph = view.graph
        return {
            "trajectory": trajectory_id,
            "nodes": [node.public() for node in graph.ordered()],
            "leaf_node_ids": graph.leaves(),
            "leaf_assistant_node_ids": graph.leaves(author="model"),
            "branch_points": graph.branch_points(),
        }

    # -- runs ----------------------------------------------------------------------------
    def list_runs(self, *, project: str | None, limit: int) -> list[dict[str, Any]]:
        """Runs, derived from the trajectories that name them.

        A run is not a record. Nothing creates one, nothing writes metadata to
        one, and nothing has to keep its counters in step with its members:
        it is the grouping its members imply, computed on read.
        """
        runs: dict[str, dict[str, Any]] = {}
        for document in self._documents():
            if document.run_id is None:
                continue
            if project is not None and document.project != project:
                continue
            run = runs.setdefault(
                document.run_id,
                {
                    "id": document.run_id,
                    "project": document.project,
                    "created_at": document.created_at,
                    "trajectory_count": 0,
                    "task_ids": set(),
                    "step_counts": {},
                },
            )
            run["trajectory_count"] += 1
            run["created_at"] = min(run["created_at"], document.created_at)
            if document.task_id is not None:
                run["task_ids"].add(document.task_id)
            if document.step is not None:
                run["step_counts"][document.step] = run["step_counts"].get(document.step, 0) + 1
        ordered = sorted(runs.values(), key=lambda run: run["created_at"], reverse=True)
        return [_run_public(run) for run in ordered[:limit]]

    def run_exists(self, run_id: str) -> bool:
        return any(document.run_id == run_id for document in self._documents())


def _run_public(run: dict[str, Any]) -> dict[str, Any]:
    steps = sorted(run["step_counts"])
    return {
        "id": run["id"],
        "project": run["project"],
        "created_at": run["created_at"].isoformat(),
        "trajectory_count": run["trajectory_count"],
        "task_count": len(run["task_ids"]),
        "steps": steps,
        # Keyed by step as a string, because that is what JSON does to an
        # integer key anyway, and a reader that guesses wrong reads none.
        "step_counts": {str(step): run["step_counts"][step] for step in steps},
    }


def exchange_public(row: dict[str, Any]) -> dict[str, Any]:
    """The `/v1/trajectories/{id}/exchanges` row shape."""
    return {
        "id": row["id"],
        "sequence": row["sequence"],
        "provider": row["provider"],
        "model": row["model"],
        "endpoint_kind": row["endpoint_kind"],
        "method": row["method"],
        "path": row["path"],
        "http_status": row["http_status"],
        "streaming": row["streaming"],
        "transport_error": row["transport_error"],
        "provider_error": row["provider_error"],
        "retry_attempt": row["retry_attempt"],
        "completion_reason": row["completion_reason"],
        "request_start_at": row["request_start_at"].isoformat(),
        "response_end_at": row["response_end_at"].isoformat() if row["response_end_at"] else None,
        "ttft_ms": row["ttft_ms"],
        "duration_ms": row["duration_ms"],
        "gap_ms": row["gap_ms"],
        "overlapping": row["overlapping"],
        "chunk_count": row["chunk_count"],
        "stream_summary": row["stream_summary"],
        "usage": row["usage"],
        "request_byte_count": row["request_byte_count"],
        "response_byte_count": row["response_byte_count"],
        "provider_request_id": row["provider_request_id"],
        "provider_response_id": row["provider_response_id"],
        "previous_response_id": row["previous_response_id"],
        "input_prefix_node_id": row["input_prefix_node_id"],
        "input_node_ids": list(row["input_node_ids"] or []),
        "input_leaf_node_id": row["input_leaf_node_id"],
        "output_node_id": row["output_node_id"],
        "parent_output_node_id": row["parent_output_node_id"],
        "previous_exchange_id": row["previous_exchange_id"],
        "is_duplicate_retry": row["is_duplicate_retry"],
        "late": row["late"],
        "has_payload": row["has_payload"],
        "schema_version": row["schema_version"],
        "derivation_version": row["derivation_version"],
    }
