"""`HeaderIndex`: a finished trajectory's document, beside its record.

    record/committed/<xx>/tr_....head.json     the document, uncompressed
    record/committed/<xx>/tr_....json.zst      the record it describes

Every listing needs the same dozen fields off the front of a trajectory --
project, run, status, times, counts -- and nothing else. Reading them out of
the record means decompressing every exchange, every node and, in tokens mode,
every token array the trajectory holds, to use none of it. At a hundred
thousand trajectories that is the difference between a viewer that opens and
one that does not.

So a commit writes the document twice: once inside the compressed record,
where it belongs to the record, and once beside it as plain JSON. Rebuilding
the whole index is then a pass over small files with no decompression, which
is what makes a cold viewer over a large directory fast.

The layout is deliberately flat and addressed by id, exactly like the record.
Grouping by project and run is not a directory structure: it is a query, and
`RecordReader.list_runs` computes it from these documents on read -- "a run is
the grouping its members imply". Putting that hierarchy on disk would make
arbitrary user strings into path segments and give a trajectory a location
that its own id could no longer find.

This index is authoritative: a listing is what the headers say, and nothing
reconciles it against the records on the read path. A header is written before
the journal that produced it is deleted, so a crash between them leaves the
journal, and replaying it writes both again -- there is no window in which a
trajectory is committed, invisible and unrecoverable. `rebuild` covers what
that rule cannot: a directory written by an older build, or one assembled by
copying records in. It is a repair, not a fallback.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import orjson

from skyrl_capture.domain.records import TrajectoryDocument
from skyrl_capture.persistence.layout import (
    COMMITTED_DIR,
    HEADER_SUFFIX,
    fsync_dir,
    shard,
)

__all__ = ["DiskHeaderIndex", "HeaderIndex", "header_path", "read_header"]


def header_path(root: Path, trajectory_id: str) -> Path:
    return root / COMMITTED_DIR / shard(trajectory_id) / f"{trajectory_id}{HEADER_SUFFIX}"


class HeaderIndex(Protocol):
    """Where a finished trajectory is listed, and how a listing is read."""

    async def put(self, document: TrajectoryDocument) -> None:
        """Write this trajectory's header, replacing any at its path."""
        ...

    def documents(self) -> list[TrajectoryDocument]: ...


class DiskHeaderIndex:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.headers_written = 0

    # -- writes ------------------------------------------------------------------
    async def put(self, document: TrajectoryDocument) -> None:
        await asyncio.to_thread(self.put_sync, document)

    def put_sync(self, document: TrajectoryDocument) -> None:
        path = header_path(self.root, document.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = orjson.dumps(document.document(), option=orjson.OPT_INDENT_2)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
        self.headers_written += 1

    def discard_sync(self, trajectory_id: str) -> None:
        header_path(self.root, trajectory_id).unlink(missing_ok=True)

    # -- reads -------------------------------------------------------------------
    def paths(self) -> list[Path]:
        """Every header, in a stable order."""
        from skyrl_capture.persistence.layout import scan_shards

        return scan_shards(self.root / COMMITTED_DIR, HEADER_SUFFIX)

    def documents(self) -> list[TrajectoryDocument]:
        found = []
        for path in self.paths():
            document = read_header(path)
            if document is not None:
                found.append(document)
        return found

    # -- repair ------------------------------------------------------------------
    def rebuild(self, committed: Any) -> dict[str, int]:
        """Write the index again from the records themselves.

        The expensive path, and the only one that decompresses a record to
        learn what is in it. It is a repair rather than a fallback: nothing
        calls it while serving, because a listing that silently fell back to
        reading every record would hide exactly the breakage an operator needs
        told about.

        Headers naming a record that is no longer there are dropped, which is
        what makes this a rebuild and not an append.
        """
        listed: set[str] = set()
        for trajectory_id in committed.committed_ids():
            record = committed.get_sync(trajectory_id)
            if record is not None:
                self.put_sync(record.trajectory)
                listed.add(record.trajectory.id)
        dropped = 0
        for path in self.paths():
            document = read_header(path)
            stale = (
                document is None
                # Naming a record that is not there: the index lists what this
                # directory holds, not what it once held.
                or document.id not in listed
                # Or sitting where its own id does not put it, which is a file
                # copied by hand. Left alone it would list a trajectory twice.
                or header_path(self.root, document.id) != path
            )
            if stale:
                path.unlink(missing_ok=True)
                dropped += 1
        return {"listed": len(listed), "dropped": dropped}

    def stats(self) -> dict[str, Any]:
        return {"trajectory_headers_written": self.headers_written}


def read_header(path: Path) -> TrajectoryDocument | None:
    try:
        return TrajectoryDocument.from_document(orjson.loads(path.read_bytes()))
    except Exception:
        # A header torn by a crash, or written by something that is not this.
        # The record it names is intact; `rebuild` is how it comes back.
        return None
