"""`CommittedStore`: one compiled record per finished trajectory.

A committed record is an artifact, not a history. It is written once, whole,
and read back without a reducer: the public document, the exchanges with their
captured bytes, and the graph nodes in order, including the exact token arrays
a token-mode trajectory carries. A viewer or an exporter opening one has
everything it needs and invokes no tokenizer.

Writing is the ordinary atomic-file dance, and every step of it matters::

    1. serialize and compress off the event loop
    2. write a temporary file *in the committed directory*
    3. flush and fsync it
    4. rename it onto the deterministic path
    5. fsync the directory

The temporary file shares a directory with its destination because rename is
atomic only within a filesystem, and a record directory may be a mount of its
own. A crash before step 4 leaves a temporary file and no record, so the retry
reconstructs and writes again; a crash after it leaves the record, so the retry
finds it and returns it. There is no state in between that a reader can see.

Bodies are stored as text when they decode as UTF-8 -- which they are, being
JSON request and response payloads -- and base64 otherwise. That keeps a
committed record legible to anything that can read JSON, and compresses far
better than base64 of the same bytes.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import orjson

from skyrl_capture.compression import compress, decompress
from skyrl_capture.domain.records import (
    CapturedExchange,
    MetadataUpdate,
    TrajectoryDocument,
    TrajectoryRecord,
)
from skyrl_capture.persistence.headers import DiskHeaderIndex
from skyrl_capture.persistence.journal import (
    exchange_row_in,
    exchange_row_out,
    node_document,
    node_from_document,
)
from skyrl_capture.persistence.layout import (
    COMMITTED_SUFFIX,
    committed_path,
    committed_paths,
    fsync_dir,
    trajectory_id_of,
)


class CommittedStore(Protocol):
    """What a finished trajectory is written to, as product operations.

    `exists` is separate from `get` because the write path asks it per cold
    trajectory -- "is this one already finished?" -- and decompressing a whole
    record to answer would be a strange way to spend that.
    """

    async def put(self, record: TrajectoryRecord) -> None: ...

    async def get(self, trajectory_id: str) -> TrajectoryRecord | None: ...

    async def exists(self, trajectory_id: str) -> bool: ...

    async def update_metadata(
        self, trajectory_id: str, update: MetadataUpdate
    ) -> TrajectoryRecord: ...

    async def relist(self, trajectory_id: str) -> None:
        """Make sure a committed record has its header.

        For the one window `put` cannot close by itself: a process that died
        between writing the record and writing its header. Whoever finds that
        journal calls this before deleting it, so the trajectory is listed
        before the evidence needed to relist it goes away.
        """
        ...


# -- bytes <-> JSON ----------------------------------------------------------------
def encode_body(data: bytes) -> dict[str, Any] | None:
    if not data:
        return None
    try:
        return {"encoding": "utf-8", "data": data.decode("utf-8")}
    except UnicodeDecodeError:
        return {"encoding": "base64", "data": base64.b64encode(data).decode("ascii")}


def decode_body(document: dict[str, Any] | None) -> bytes:
    if not document:
        return b""
    if document.get("encoding") == "base64":
        return base64.b64decode(document["data"])
    return str(document["data"]).encode("utf-8")


def record_document(record: TrajectoryRecord) -> dict[str, Any]:
    return {
        "record_version": record.record_version,
        "schema_version": record.schema_version,
        "derivation_version": record.derivation_version,
        "revision": record.revision,
        "finish_request_hash": record.finish_request_hash,
        "trajectory": record.trajectory.document(),
        "exchanges": [
            {
                "id": exchange.id,
                "sequence": exchange.sequence,
                "row": exchange_row_out(exchange.row),
                "delivery_confirmed": exchange.delivery_confirmed,
                "delivery_uncertain": exchange.delivery_uncertain,
                "request": encode_body(exchange.request_body),
                "response": encode_body(exchange.response_body),
                "chunks": exchange.chunks,
                "tokens": exchange.tokens,
            }
            for exchange in record.exchanges
        ],
        "nodes": [node_document(node) for node in record.nodes],
        "node_order": list(record.node_order),
    }


def record_from_document(document: dict[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord(
        record_version=int(document["record_version"]),
        schema_version=int(document["schema_version"]),
        derivation_version=int(document["derivation_version"]),
        revision=int(document.get("revision", 0)),
        finish_request_hash=document.get("finish_request_hash") or "",
        trajectory=TrajectoryDocument.from_document(document["trajectory"]),
        exchanges=tuple(
            CapturedExchange(
                id=row["id"],
                sequence=int(row["sequence"]),
                row=exchange_row_in(row["row"]),
                delivery_confirmed=bool(row.get("delivery_confirmed", True)),
                delivery_uncertain=bool(row.get("delivery_uncertain", False)),
                request_body=decode_body(row.get("request")),
                response_body=decode_body(row.get("response")),
                chunks=row.get("chunks"),
                tokens=row.get("tokens"),
            )
            for row in document.get("exchanges") or ()
        ),
        nodes=tuple(node_from_document(node) for node in document.get("nodes") or ()),
        node_order=tuple(document.get("node_order") or ()),
    )


def serialize(record: TrajectoryRecord) -> bytes:
    return compress(orjson.dumps(record_document(record)))


def deserialize(data: bytes) -> TrajectoryRecord:
    return record_from_document(orjson.loads(decompress(data)))


# -- the disk implementation --------------------------------------------------------
class DiskCommittedStore:
    def __init__(self, root: str | Path, headers: DiskHeaderIndex | None = None) -> None:
        self.root = Path(root).expanduser()
        # The header is written here, in the same call that writes the record,
        # so no caller has to remember the order. The index is authoritative,
        # and a record whose header is missing is a record no listing shows --
        # so writing it is never someone else's job.
        self.headers = headers if headers is not None else DiskHeaderIndex(self.root)
        self.records_written = 0
        self.bytes_written = 0
        self.compress_ns = 0
        self.write_ns = 0

    def path_for(self, trajectory_id: str) -> Path:
        return committed_path(self.root, trajectory_id)

    def committed_ids(self) -> list[str]:
        return [trajectory_id_of(path, COMMITTED_SUFFIX) for path in committed_paths(self.root)]

    async def put(self, record: TrajectoryRecord) -> None:
        await asyncio.to_thread(self._put_sync, record)

    async def get(self, trajectory_id: str) -> TrajectoryRecord | None:
        return await asyncio.to_thread(self._get_sync, trajectory_id)

    def get_sync(self, trajectory_id: str) -> TrajectoryRecord | None:
        """For a caller already running in a worker thread -- the indexer."""
        return self._get_sync(trajectory_id)

    async def exists(self, trajectory_id: str) -> bool:
        """Whether a record is committed, without reading it.

        The write path asks this per cold trajectory to find out whether it is
        already finished, and decompressing a whole record to answer would be
        a strange way to spend that.
        """
        return await asyncio.to_thread(self.path_for(trajectory_id).is_file)

    async def update_metadata(
        self, trajectory_id: str, update: MetadataUpdate
    ) -> TrajectoryRecord:
        """Load, edit, bump the revision, replace the file atomically.

        Raises `KeyError` when there is no committed record. Previously
        rendered exports are historical snapshots and are not rewritten; later
        reads and exports use what this wrote.
        """
        return await asyncio.to_thread(self._update_sync, trajectory_id, update)

    # -- the synchronous half ---------------------------------------------------
    async def relist(self, trajectory_id: str) -> None:
        await asyncio.to_thread(self._relist_sync, trajectory_id)

    def _relist_sync(self, trajectory_id: str) -> None:
        record = self._get_sync(trajectory_id)
        if record is not None:
            self.headers.put_sync(record.trajectory)

    def _put_sync(self, record: TrajectoryRecord) -> None:
        started = time.perf_counter_ns()
        payload = serialize(record)
        self.compress_ns += time.perf_counter_ns() - started
        self._write_bytes(record.id, payload)
        # After the record, so a torn commit leaves an unlisted record rather
        # than a header naming one that is not there. The journal outlives
        # both, and replaying it writes both again.
        self.headers.put_sync(record.trajectory)

    def _write_bytes(self, trajectory_id: str, payload: bytes) -> None:
        started = time.perf_counter_ns()
        path = self.path_for(trajectory_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique per write, not per process: a metadata rewrite and a commit
        # can be in flight together, and two writers sharing one temporary
        # name would interleave into a file that is neither.
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        with open(temporary, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
        self.records_written += 1
        self.bytes_written += len(payload)
        self.write_ns += time.perf_counter_ns() - started

    def _get_sync(self, trajectory_id: str) -> TrajectoryRecord | None:
        path = self.path_for(trajectory_id)
        try:
            return deserialize(path.read_bytes())
        except FileNotFoundError:
            return None

    def _update_sync(self, trajectory_id: str, update: MetadataUpdate) -> TrajectoryRecord:
        record = self._get_sync(trajectory_id)
        if record is None:
            raise KeyError(trajectory_id)
        updated = record.with_metadata(update)
        self._write_bytes(trajectory_id, serialize(updated))
        # The header is addressed by id, like the record, so a revision
        # replaces it where it already is. A listing cannot be made to lose a
        # row by an edit.
        self.headers.put_sync(updated.trajectory)
        return updated

    def stats(self) -> dict[str, Any]:
        return {
            "committed_records_written": self.records_written,
            "committed_bytes_written": self.bytes_written,
            "compress_ms_mean": round(self.compress_ns / 1e6 / max(1, self.records_written), 3),
            "commit_write_ms_mean": round(self.write_ns / 1e6 / max(1, self.records_written), 3),
        }
