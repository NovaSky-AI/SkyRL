"""The active journal: one append-only file per unfinished trajectory.

A journal is trajectory-local. There is no global sequence, no segment range
and no shared writer: the file is named by the trajectory it belongs to, and
the invariant that exactly one process appends to it comes from consistent-hash
routing rather than from anything in this format.

Framing is fixed-width, little-endian and dependency-free, so a reader in
another language can walk a journal with this file as its specification::

    journal := header record*
    header  := magic[8] version[u16] flags[u16] reserved[u32]
    record  := length[u32] crc32[u32] kind[u16] flags[u16] body[length]
    body    := doc_len[u32] doc[doc_len] blob_count[u16] (blob_len[u32] blob)*

``length`` counts only ``body``. ``kind`` names the record type; a kind this
reader does not know is skipped by its length, so an old reader can walk a
journal written by a newer writer. ``FLAG_ZSTD`` means ``body`` is
zstd-compressed as a whole. ``doc`` is JSON and carries every field except raw
byte strings, which travel as blobs so a request body is stored as the bytes
that were sent rather than base64 of them.

A record whose trailing bytes are missing, or whose CRC does not match, is a
torn tail: the scan stops there and says so. It does not guess past it, and a
replacement writer truncates to the last valid byte before appending -- writing
after a torn record would hide it behind a valid one.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any

import orjson

from skyrl_capture.compression import compress, decompress
from skyrl_capture.domain.graph import GraphDelta, GraphNode
from skyrl_capture.domain.records import (
    CapturedExchange,
    MetadataUpdate,
    TrajectoryHeader,
)
from skyrl_capture.version import JOURNAL_FORMAT_VERSION

MAGIC = b"ICAPTRJ1"
HEADER = struct.Struct("<8sHHI")
RECORD_HEADER = struct.Struct("<IIHH")
U32 = struct.Struct("<I")
U16 = struct.Struct("<H")

HEADER_SIZE = HEADER.size
RECORD_HEADER_SIZE = RECORD_HEADER.size

FLAG_ZSTD = 1 << 0

# Bodies above this are compressed. Below it the zstd frame costs more than it
# saves, and most lifecycle records are a few hundred bytes.
COMPRESS_ABOVE = 4096

_NODE_TIMES = ("output_started_at", "output_ended_at", "inserted_at")
_EXCHANGE_TIMES = ("request_start_at", "response_end_at")


class JournalFormatError(Exception):
    """A journal this build cannot read at all."""


# -- the records a journal holds --------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrajectoryCreated:
    """Creation, persisted before the route is handed back."""

    header: TrajectoryHeader


@dataclass(frozen=True, slots=True)
class ExchangeCommitted:
    """One captured exchange, complete: there are no partial-stream records.

    ``request_fingerprint`` is a hash of the standard request body and is
    diagnostic only. It must never deduplicate requests -- two identical
    prompts in one trajectory are intentional resampling, and collapsing them
    would delete a sample the trainer is entitled to.
    """

    exchange: CapturedExchange
    graph: GraphDelta | None
    at: datetime
    request_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class ExchangeDeliveryConfirmed:
    """The complete response reached the downstream transport.

    Appended after the ASGI send completes, which is necessarily after the
    exchange is durable. An exchange with no confirmation after it is not
    corrupt: it is a response whose delivery this process cannot vouch for.
    """

    exchange_id: str
    at: datetime


@dataclass(frozen=True, slots=True)
class CaptureGap:
    """This many exchanges were observed and not captured."""

    count: int
    reason: str | None
    at: datetime


@dataclass(frozen=True, slots=True)
class MetadataUpdated:
    update: MetadataUpdate
    at: datetime


@dataclass(frozen=True, slots=True)
class FinishRequested:
    """The route closes. Written before the compile, so a crash mid-finish
    leaves evidence that a finish was asked for and with what."""

    update: MetadataUpdate
    command_result: str | None
    request_hash: str
    at: datetime


@dataclass(frozen=True, slots=True)
class TrajectoryPoisoned:
    reason: str
    at: datetime


ActiveRecord = (
    TrajectoryCreated
    | ExchangeCommitted
    | ExchangeDeliveryConfirmed
    | CaptureGap
    | MetadataUpdated
    | FinishRequested
    | TrajectoryPoisoned
)

# Kinds are additive. Never renumber one.
KINDS: dict[type, int] = {
    TrajectoryCreated: 1,
    ExchangeCommitted: 2,
    ExchangeDeliveryConfirmed: 3,
    CaptureGap: 4,
    MetadataUpdated: 5,
    FinishRequested: 6,
    TrajectoryPoisoned: 7,
}
RECORD_BY_KIND: dict[int, type] = {kind: klass for klass, kind in KINDS.items()}


# -- header ---------------------------------------------------------------------------
def encode_header(flags: int = 0) -> bytes:
    return HEADER.pack(MAGIC, JOURNAL_FORMAT_VERSION, flags, 0)


def decode_header(buffer: bytes) -> tuple[int, int]:
    if len(buffer) < HEADER_SIZE:
        raise JournalFormatError("journal shorter than its header")
    magic, version, flags, _reserved = HEADER.unpack(buffer[:HEADER_SIZE])
    if magic != MAGIC:
        raise JournalFormatError(
            f"not a trajectory journal (magic {magic!r}). A global event log from an "
            "earlier build is not readable by this one; re-capture."
        )
    if version != JOURNAL_FORMAT_VERSION:
        raise JournalFormatError(
            f"journal format version {version}, but this build reads {JOURNAL_FORMAT_VERSION}"
        )
    return version, flags


# -- values <-> documents ---------------------------------------------------------------
def _time_out(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _time_in(value: Any) -> Any:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def node_document(node: GraphNode) -> dict[str, Any]:
    out = {item.name: getattr(node, item.name) for item in fields(node)}
    for name in _NODE_TIMES:
        out[name] = _time_out(out[name])
    return out


def node_from_document(row: dict[str, Any]) -> GraphNode:
    row = dict(row)
    for name in _NODE_TIMES:
        row[name] = _time_in(row.get(name))
    return GraphNode(**row)


def delta_document(delta: GraphDelta | None) -> dict[str, Any] | None:
    if delta is None:
        return None
    return {
        "exchange_id": delta.exchange_id,
        "nodes": [node_document(node) for node in delta.nodes],
        "input_prefix_node_id": delta.input_prefix_node_id,
        "input_node_ids": list(delta.input_node_ids),
        "input_leaf_node_id": delta.input_leaf_node_id,
        "output_node_id": delta.output_node_id,
        "parent_output_node_id": delta.parent_output_node_id,
        "matched_count": delta.matched_count,
        "is_duplicate_retry": delta.is_duplicate_retry,
    }


def delta_from_document(row: dict[str, Any] | None) -> GraphDelta | None:
    if row is None:
        return None
    return GraphDelta(
        exchange_id=row["exchange_id"],
        nodes=tuple(node_from_document(node) for node in row["nodes"]),
        input_prefix_node_id=row["input_prefix_node_id"],
        input_node_ids=tuple(row["input_node_ids"]),
        input_leaf_node_id=row["input_leaf_node_id"],
        output_node_id=row["output_node_id"],
        parent_output_node_id=row["parent_output_node_id"],
        matched_count=row["matched_count"],
        is_duplicate_retry=row["is_duplicate_retry"],
    )


def exchange_row_out(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for name in _EXCHANGE_TIMES:
        out[name] = _time_out(out.get(name))
    return out


def exchange_row_in(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for name in _EXCHANGE_TIMES:
        out[name] = _time_in(out.get(name))
    return out


def record_document(record: ActiveRecord) -> tuple[dict[str, Any], list[bytes]]:
    """The JSON document and the byte blobs one record serializes to."""
    if isinstance(record, TrajectoryCreated):
        return {"header": record.header.document()}, []
    if isinstance(record, ExchangeCommitted):
        exchange = record.exchange
        document = {
            "id": exchange.id,
            "sequence": exchange.sequence,
            "row": exchange_row_out(exchange.row),
            "delivery_confirmed": exchange.delivery_confirmed,
            "chunks": exchange.chunks,
            "tokens": exchange.tokens,
            "graph": delta_document(record.graph),
            "at": _time_out(record.at),
            "request_fingerprint": record.request_fingerprint,
        }
        return document, [exchange.request_body or b"", exchange.response_body or b""]
    if isinstance(record, ExchangeDeliveryConfirmed):
        return {"exchange_id": record.exchange_id, "at": _time_out(record.at)}, []
    if isinstance(record, CaptureGap):
        return {"count": record.count, "reason": record.reason, "at": _time_out(record.at)}, []
    if isinstance(record, MetadataUpdated):
        return {"update": record.update.document(), "at": _time_out(record.at)}, []
    if isinstance(record, FinishRequested):
        return {
            "update": record.update.document(),
            "command_result": record.command_result,
            "request_hash": record.request_hash,
            "at": _time_out(record.at),
        }, []
    if isinstance(record, TrajectoryPoisoned):
        return {"reason": record.reason, "at": _time_out(record.at)}, []
    raise TypeError(f"no journal encoding for {type(record).__name__}")


def record_from_document(klass: type, doc: dict[str, Any], blobs: list[bytes]) -> ActiveRecord:
    if klass is TrajectoryCreated:
        return TrajectoryCreated(TrajectoryHeader.from_document(doc["header"]))
    if klass is ExchangeCommitted:
        request_body = blobs[0] if len(blobs) > 0 else b""
        response_body = blobs[1] if len(blobs) > 1 else b""
        return ExchangeCommitted(
            exchange=CapturedExchange(
                id=doc["id"],
                sequence=doc["sequence"],
                row=exchange_row_in(doc["row"]),
                delivery_confirmed=bool(doc.get("delivery_confirmed", True)),
                request_body=request_body,
                response_body=response_body,
                chunks=doc.get("chunks"),
                tokens=doc.get("tokens"),
            ),
            graph=delta_from_document(doc.get("graph")),
            at=_time_in(doc["at"]),
            request_fingerprint=doc.get("request_fingerprint") or "",
        )
    if klass is ExchangeDeliveryConfirmed:
        return ExchangeDeliveryConfirmed(doc["exchange_id"], _time_in(doc["at"]))
    if klass is CaptureGap:
        return CaptureGap(int(doc["count"]), doc.get("reason"), _time_in(doc["at"]))
    if klass is MetadataUpdated:
        return MetadataUpdated(MetadataUpdate.from_document(doc["update"]), _time_in(doc["at"]))
    if klass is FinishRequested:
        return FinishRequested(
            update=MetadataUpdate.from_document(doc["update"]),
            command_result=doc.get("command_result"),
            request_hash=doc.get("request_hash") or "",
            at=_time_in(doc["at"]),
        )
    if klass is TrajectoryPoisoned:
        return TrajectoryPoisoned(doc["reason"], _time_in(doc["at"]))
    raise TypeError(f"no journal decoding for {klass.__name__}")


# -- framing ----------------------------------------------------------------------------
def encode_record(record: ActiveRecord, *, compress_payloads: bool = True) -> bytes:
    kind = KINDS[type(record)]
    doc, blobs = record_document(record)
    encoded = orjson.dumps(doc)
    parts = [U32.pack(len(encoded)), encoded, U16.pack(len(blobs))]
    for blob in blobs:
        parts.append(U32.pack(len(blob)))
        parts.append(blob)
    body = b"".join(parts)
    flags = 0
    if compress_payloads and len(body) > COMPRESS_ABOVE:
        body = compress(body)
        flags |= FLAG_ZSTD
    return RECORD_HEADER.pack(len(body), zlib.crc32(body) & 0xFFFFFFFF, kind, flags) + body


@dataclass(frozen=True, slots=True)
class FramedRecord:
    kind: int
    flags: int
    body: bytes
    offset: int

    def decode(self) -> ActiveRecord | None:
        """The record, or `None` for a kind this build does not know."""
        klass = RECORD_BY_KIND.get(self.kind)
        if klass is None:
            return None
        body = decompress(self.body) if self.flags & FLAG_ZSTD else self.body
        view = memoryview(body)
        (doc_len,) = U32.unpack_from(view, 0)
        offset = U32.size
        doc = orjson.loads(view[offset : offset + doc_len])
        offset += doc_len
        (count,) = U16.unpack_from(view, offset)
        offset += U16.size
        blobs: list[bytes] = []
        for _ in range(count):
            (length,) = U32.unpack_from(view, offset)
            offset += U32.size
            blobs.append(bytes(view[offset : offset + length]))
            offset += length
        return record_from_document(klass, doc, blobs)


@dataclass(slots=True)
class JournalScan:
    records: list[ActiveRecord] = field(default_factory=list)
    #: Bytes consumed by whole, valid records, including the file header. A
    #: replacement writer truncates to this and appends from there.
    consumed: int = HEADER_SIZE
    truncated: bool = False
    unknown_kinds: int = 0


def scan(buffer: bytes, *, start: int = 0) -> JournalScan:
    """Every whole record in ``buffer``, and whether the tail was torn.

    ``start`` resumes an incremental read: it is a previous scan's ``consumed``,
    and the header is checked only when reading from the beginning.
    """
    if start <= 0:
        decode_header(buffer)
        offset = HEADER_SIZE
    else:
        offset = start
    total = len(buffer)
    result = JournalScan(consumed=offset)
    while offset < total:
        if total - offset < RECORD_HEADER_SIZE:
            result.truncated = True
            break
        length, crc, kind, flags = RECORD_HEADER.unpack_from(buffer, offset)
        body_start = offset + RECORD_HEADER_SIZE
        body_end = body_start + length
        if body_end > total:
            result.truncated = True
            break
        body = buffer[body_start:body_end]
        if (zlib.crc32(body) & 0xFFFFFFFF) != crc:
            result.truncated = True
            break
        framed = FramedRecord(kind=kind, flags=flags, body=body, offset=offset)
        decoded = framed.decode()
        if decoded is None:
            result.unknown_kinds += 1
        else:
            result.records.append(decoded)
        offset = body_end
        result.consumed = offset
    return result
