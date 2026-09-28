"""`ActiveStore`: the journal of every trajectory that has not finished.

Four operations, all of them about one trajectory:

* `create` writes the header record and makes it durable before creation is
  reported to the caller -- a route handed out for a trajectory that is not on
  disk would be a route no replacement process could serve.
* `append` adds one record and makes it durable. TITO awaits this before it
  closes a response, so "appended" has to mean "survives the process".
* `recover` rebuilds the hot aggregate from a journal this process did not
  write, truncating a torn tail so the replacement writer continues from the
  last whole record.
* `remove` deletes a journal, and is called only after its committed record is
  visible.

File handles are cached per trajectory and the append path is serialized per
trajectory, not globally: two trajectories on one process must not queue behind
each other's fsync. Everything that touches the filesystem runs in a worker
thread, so an fsync never blocks the event loop that is serving other turns.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from skyrl_capture.domain.records import (
    ActiveTrajectory,
    CapturedExchange,
    TrajectoryHeader,
)
from skyrl_capture.persistence import journal
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
    active_path,
    active_paths,
    fsync_dir,
    trajectory_id_of,
)

logger = logging.getLogger(__name__)

RECOVERY_UNCERTAIN = (
    "recovered by a replacement process: text capture commits behind the response, "
    "so an exchange may have been served inside the previous process's commit window"
)


class JournalClosed(OSError):
    """An append arrived for a trajectory whose record is already committed.

    An `OSError`, because every caller already treats one as a persistence
    failure -- which is exactly what this is. It must never be swallowed: a
    TITO response that closed cleanly on a dropped append would be the one
    thing the design promises cannot happen.
    """


class ActiveStore(Protocol):
    """What a trajectory in flight is written to, as product operations.

    Five of them, and no way to ask where any of it lives. `finish` needs a
    trajectory's journal read back, so that is `records` rather than a path a
    caller then opens -- a path is a fact about *this* implementation, and an
    interface that hands one out cannot be implemented by anything else.
    """

    async def create(self, header: TrajectoryHeader) -> None: ...

    async def append(self, trajectory_id: str, record: ActiveRecord) -> None: ...

    async def recover(self, trajectory_id: str) -> ActiveTrajectory | None: ...

    async def records(self, trajectory_id: str) -> list[ActiveRecord]:
        """Everything the journal holds, in order. Empty when there is none."""
        ...

    async def remove(self, trajectory_id: str) -> None: ...


# -- shard directories -----------------------------------------------------------------
def _prune_shard(directory: Path) -> None:
    """Drop a shard directory that holds nothing.

    A finished trajectory's journal is deleted, and the two-character directory
    it lived in is usually the last thing in it. Leaving those behind turns a
    record whose work is done into 256 empty directories, which reads like
    unfinished work and is not. `rmdir` removes a directory only while it is
    empty, so a shard another trajectory is still using is never touched.
    """
    with contextlib.suppress(OSError):
        directory.rmdir()


def _open_in_shard(path: Path) -> Any:
    """Open a journal for append, creating its shard directory.

    The retry is the other half of `_prune_shard`. Two processes share a record
    directory, and one may prune a shard in the window between this process
    creating that shard and opening a file inside it -- the shard is genuinely
    empty at that instant. Recreating it once is enough, because the loser now
    holds a file in it and no further prune can succeed.
    """
    for attempt in (0, 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return open(path, "ab")
        except FileNotFoundError:
            if attempt:
                raise
    raise AssertionError("unreachable")


# -- journal -> aggregate --------------------------------------------------------------
def rebuild(records: list[ActiveRecord], *, recovered: bool) -> ActiveTrajectory | None:
    """The aggregate a journal describes, or `None` if it has no header.

    A journal whose first whole record is not `TrajectoryCreated` is a file
    that was torn before creation became durable. There is nothing to attach
    the rest of it to, and inventing a header would invent a project.
    """
    if not records or not isinstance(records[0], TrajectoryCreated):
        return None
    active = ActiveTrajectory.create(records[0].header)
    active.recovered = recovered
    for record in records[1:]:
        apply_record(active, record)
    if recovered:
        _mark_uncertainty(active)
    return active


def apply_record(active: ActiveTrajectory, record: ActiveRecord) -> None:
    """Apply one journal record to an aggregate. Idempotent per exchange."""
    if isinstance(record, ExchangeCommitted):
        active.add_exchange(record.exchange, record.graph, record.at)
    elif isinstance(record, ExchangeDeliveryConfirmed):
        active.confirm_delivery(record.exchange_id)
    elif isinstance(record, CaptureGap):
        active.record_gap(record.count, record.reason, record.at)
    elif isinstance(record, MetadataUpdated):
        active.apply_metadata(record.update)
    elif isinstance(record, FinishRequested):
        active.apply_metadata(record.update)
        active.request_finish(
            command_result=record.command_result, request_hash=record.request_hash, at=record.at
        )
    elif isinstance(record, TrajectoryPoisoned):
        active.poison(record.reason, record.at)
    elif isinstance(record, TrajectoryCreated):
        pass  # The header is how the aggregate was built.


def _mark_uncertainty(active: ActiveTrajectory) -> None:
    """What a replacement process cannot vouch for, stated on the aggregate.

    The two doubts are different and are recorded differently. Text capture
    commits after the response has gone, so a previous process may have served
    an exchange inside its commit window and died before writing it: nothing
    identifies that exchange, so the doubt is trajectory-wide. TITO commits
    before the response closes, so every exchange it captured is exact and the
    doubt is per exchange -- whether the client got it.
    """
    if active.header.mode == "tokens":
        for exchange in active.exchanges:
            if not exchange.delivery_confirmed:
                exchange.delivery_uncertain = True
                if exchange.id not in active.integrity.delivery_uncertain_exchange_ids:
                    active.integrity.delivery_uncertain_exchange_ids.append(exchange.id)
    else:
        active.integrity.recovery_uncertain = True


def read_journal(path: Path, *, start: int = 0) -> journal.JournalScan:
    """Scan one journal file from ``start``. Missing file scans as empty."""
    try:
        buffer = path.read_bytes()
    except FileNotFoundError:
        return journal.JournalScan(records=[], consumed=0, truncated=False)
    if not buffer:
        return journal.JournalScan(records=[], consumed=0, truncated=False)
    return journal.scan(buffer, start=start)


# -- the disk implementation ------------------------------------------------------------
class DiskActiveStore:
    """Active journals under one record root.

    ``fsync`` is the durability policy: ``always`` is the only setting under
    which TITO's promise holds, because that promise is that a cleanly closed
    response has its exchange on disk. ``interval`` and ``never`` exist for
    benchmarking the cost of that promise, and they weaken it.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        fsync: str = "always",
        fsync_interval: float = 1.0,
        compress: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self._fsync = fsync
        self._fsync_interval = fsync_interval
        self._compress = compress
        self._handles: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_fsync: dict[str, float] = {}
        # Trajectories whose journal was deleted because their record is
        # committed. A turn that outlived `finish`'s grace can still arrive,
        # and re-creating the file for it would leave an orphan journal for a
        # trajectory that has finished -- invisible, because readers prefer
        # the committed record, and never collected.
        self._finished: set[str] = set()
        self._closed = False
        # What monitoring reports. Latencies are summed in nanoseconds and
        # divided on read: two clock reads per append, against an append that
        # costs an fsync.
        self.records_written = 0
        self.bytes_written = 0
        self.append_ns = 0
        self.fsync_ns = 0
        self.fsync_count = 0
        self.recoveries = 0
        self.recovery_ns = 0
        self.torn_tails = 0

    # -- locking -------------------------------------------------------------
    def _lock(self, trajectory_id: str) -> asyncio.Lock:
        lock = self._locks.get(trajectory_id)
        if lock is None:
            lock = self._locks[trajectory_id] = asyncio.Lock()
        return lock

    # -- the protocol ---------------------------------------------------------
    async def create(self, header: TrajectoryHeader) -> None:
        async with self._lock(header.id):
            # A trajectory id is not reused -- `TrajectoryRegistry.create`
            # refuses one whose record is committed -- but a store that was
            # told a name is finished should forget that when told to create it.
            self._finished.discard(header.id)
            await asyncio.to_thread(self._create_sync, header)

    async def append(self, trajectory_id: str, record: ActiveRecord) -> None:
        async with self._lock(trajectory_id):
            await asyncio.to_thread(self._append_sync, trajectory_id, record)

    async def records(self, trajectory_id: str) -> list[ActiveRecord]:
        """The journal's records, whole ones only. What `finish` compiles from."""
        return await asyncio.to_thread(self._records_sync, trajectory_id)

    def _records_sync(self, trajectory_id: str) -> list[ActiveRecord]:
        return read_journal(self.path_for(trajectory_id)).records

    async def recover(self, trajectory_id: str) -> ActiveTrajectory | None:
        async with self._lock(trajectory_id):
            recovered = await asyncio.to_thread(self._recover_sync, trajectory_id)
        if recovered is None:
            # There was no journal, so there is nothing to serialize against.
            # Dropped rather than kept, because an id nobody has heard of is
            # what a misrouted flood of requests is made of.
            self._locks.pop(trajectory_id, None)
        return recovered

    async def remove(self, trajectory_id: str) -> None:
        async with self._lock(trajectory_id):
            await asyncio.to_thread(self._remove_sync, trajectory_id)
        self._locks.pop(trajectory_id, None)

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._close_sync)

    # -- ids on disk ------------------------------------------------------------
    def journal_ids(self) -> list[str]:
        return [trajectory_id_of(path, JOURNAL_SUFFIX) for path in active_paths(self.root)]

    def path_for(self, trajectory_id: str) -> Path:
        return active_path(self.root, trajectory_id)

    # -- the synchronous half -----------------------------------------------------
    def _create_sync(self, header: TrajectoryHeader) -> None:
        path = self.path_for(header.id)
        handle = _open_in_shard(path)
        try:
            if handle.tell() == 0:
                handle.write(journal.encode_header())
            self._handles[header.id] = handle
            self._write(header.id, journal.TrajectoryCreated(header), force_fsync=True)
        except BaseException:
            self._handles.pop(header.id, None)
            with contextlib.suppress(Exception):
                handle.close()
            raise
        # The file itself is durable; its directory entry is what makes it
        # findable by a replacement process.
        fsync_dir(path.parent)

    def _handle_for(self, trajectory_id: str) -> Any:
        handle = self._handles.get(trajectory_id)
        if handle is not None:
            return handle
        path = self.path_for(trajectory_id)
        handle = _open_in_shard(path)
        if handle.tell() == 0:
            handle.write(journal.encode_header())
        self._handles[trajectory_id] = handle
        return handle

    def _append_sync(self, trajectory_id: str, record: ActiveRecord) -> None:
        if trajectory_id in self._finished:
            # Too late: the record is committed and cannot take another
            # exchange. Re-creating the journal would leave an orphan file for
            # a finished trajectory, and *dropping* this quietly would be
            # worse -- a TITO turn waiting on it would close cleanly with its
            # exchange nowhere. So it fails, and the caller decides: text
            # counts a gap, TITO fails the connection.
            raise JournalClosed(
                f"the record for {trajectory_id!r} is already committed, so its journal "
                f"cannot take a {type(record).__name__}"
            )
        self._handle_for(trajectory_id)
        self._write(trajectory_id, record)

    def _write(self, trajectory_id: str, record: ActiveRecord, *, force_fsync: bool = False) -> None:
        handle = self._handles[trajectory_id]
        started = time.perf_counter_ns()
        encoded = journal.encode_record(record, compress_payloads=self._compress)
        handle.write(encoded)
        handle.flush()
        self.records_written += 1
        self.bytes_written += len(encoded)
        if force_fsync or self._should_fsync(trajectory_id):
            fsync_started = time.perf_counter_ns()
            os.fsync(handle.fileno())
            self.fsync_ns += time.perf_counter_ns() - fsync_started
            self.fsync_count += 1
            self._last_fsync[trajectory_id] = time.monotonic()
        self.append_ns += time.perf_counter_ns() - started

    def _should_fsync(self, trajectory_id: str) -> bool:
        if self._fsync == "never":
            return False
        if self._fsync == "always":
            return True
        last = self._last_fsync.get(trajectory_id, 0.0)
        return time.monotonic() - last >= self._fsync_interval

    def _recover_sync(self, trajectory_id: str) -> ActiveTrajectory | None:
        path = self.path_for(trajectory_id)
        if not path.is_file():
            return None
        started = time.perf_counter_ns()
        scan = read_journal(path)
        if scan.truncated:
            # Truncate before anything appends. A record written after a torn
            # one would sit behind a tear no reader walks past, so it would be
            # durable and invisible -- worse than not written at all.
            self.torn_tails += 1
            logger.warning(
                "journal %s ends in a torn record; truncating to %d bytes and continuing",
                path,
                scan.consumed,
            )
            with open(path, "r+b") as handle:
                handle.truncate(scan.consumed)
                handle.flush()
                os.fsync(handle.fileno())
        active = rebuild(scan.records, recovered=True)
        self.recoveries += 1
        self.recovery_ns += time.perf_counter_ns() - started
        if active is None:
            logger.warning("journal %s has no usable header record; ignoring it", path)
            return None
        # Reopen for append, so the next record lands after the last whole one.
        self._handles.pop(trajectory_id, None)
        self._handle_for(trajectory_id)
        if active.integrity.recovery_uncertain:
            # Write the doubt down. A viewer reading this journal must see what
            # the process that adopted it knows, and memory is not where that
            # belongs.
            self._write(trajectory_id, CaptureGap(0, RECOVERY_UNCERTAIN, datetime.now(UTC)))
        return active

    def _remove_sync(self, trajectory_id: str) -> None:
        self._finished.add(trajectory_id)
        handle = self._handles.pop(trajectory_id, None)
        if handle is not None:
            with contextlib.suppress(Exception):
                handle.close()
        self._last_fsync.pop(trajectory_id, None)
        path = self.path_for(trajectory_id)
        path.unlink(missing_ok=True)
        _prune_shard(path.parent)

    def _close_sync(self) -> None:
        for handle in self._handles.values():
            with contextlib.suppress(Exception):
                handle.flush()
                if self._fsync != "never":
                    os.fsync(handle.fileno())
                handle.close()
        self._handles.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "records_written": self.records_written,
            "bytes_written": self.bytes_written,
            "open_journals": len(self._handles),
            "append_ms_mean": round(self.append_ns / 1e6 / max(1, self.records_written), 4),
            "fsync_ms_mean": round(self.fsync_ns / 1e6 / max(1, self.fsync_count), 4),
            "fsyncs": self.fsync_count,
            "recoveries": self.recoveries,
            "recovery_ms_mean": round(self.recovery_ns / 1e6 / max(1, self.recoveries), 3),
            "torn_tails": self.torn_tails,
        }


def payload_of(exchange: CapturedExchange) -> dict[str, Any]:
    """One exchange's captured bytes, in the shape the viewer asks for."""
    return {
        "request": exchange.request_body,
        "response": exchange.response_body,
        "chunks": exchange.chunks,
        "tokens": exchange.tokens,
    }
