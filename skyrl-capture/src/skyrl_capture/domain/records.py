"""The two shapes a trajectory has, and the one document both answer with.

A trajectory is hot or it is finished, and the difference is not a status
column -- it is which object holds it:

* `ActiveTrajectory` is the hot aggregate. One per locally used trajectory,
  owned by one process, mutated between awaits and never shared. It is what a
  turn commits into and what `finish` compiles from.
* `TrajectoryRecord` is the finished artifact. Frozen, versioned, written
  once, and viewer-ready: a reader opens it and has the graph, the exchanges
  and the exact token associations without replaying a lifecycle or invoking a
  tokenizer.

Both project the same `TrajectoryDocument`, which is the JSON `/v1` serves.
That is deliberate: a viewer looking at a trajectory mid-run and the same
viewer looking at it an hour later are reading one document shape, so nothing
downstream has to know which half of the system answered.

There is no global reducer here. Each of these objects is changed by the
handful of methods below, by the one process that owns it.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from skyrl_capture.domain.graph import ConversationGraph
from skyrl_capture.version import DERIVATION_VERSION, RECORD_FORMAT_VERSION, SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover - typing only
    from concurrent.futures import Future

    from skyrl_capture.domain.graph import GraphNode

# A trajectory takes turns while it is `created` or `active`; `finalizing`
# means finish is running and new turns are refused; the rest are terminal.
LIVE_STATUSES = ("created", "active")
DONE_STATUSES = ("finished", "poisoned")


def now() -> datetime:
    """The one clock a record is stamped with."""
    return datetime.now(UTC)


# -- errors ---------------------------------------------------------------------
class TrajectoryError(Exception):
    pass


class TrajectoryConflict(TrajectoryError):
    """The caller reused a trajectory id, or a finish hash, with a new body."""


class MetadataError(Exception):
    pass


class ExportError(Exception):
    pass


# -- integrity -------------------------------------------------------------------
@dataclass
class CaptureIntegrity:
    """What capture knows it does *not* know about this trajectory.

    ``recovery_uncertain`` and ``delivery_uncertain_exchange_ids`` describe two
    different doubts and must not be collapsed. Recovery uncertainty is about
    capture: a replacement process picked this trajectory up, and text capture
    commits asynchronously, so an exchange may have been served by the previous
    process inside its commit window and never written. Delivery uncertainty is
    about the *client*: the exchange is durably captured and exact, and what is
    unknown is whether the complete response reached the caller before the
    process died.
    """

    complete: bool = True
    calls_missing: int = 0
    recovery_uncertain: bool = False
    delivery_uncertain_exchange_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "complete": self.complete and not self.calls_missing,
            "calls_missing": self.calls_missing,
            "recovery_uncertain": self.recovery_uncertain,
            "delivery_uncertain": list(self.delivery_uncertain_exchange_ids),
            "errors": list(self.errors),
        }

    def copy(self) -> CaptureIntegrity:
        return CaptureIntegrity(
            complete=self.complete,
            calls_missing=self.calls_missing,
            recovery_uncertain=self.recovery_uncertain,
            delivery_uncertain_exchange_ids=list(self.delivery_uncertain_exchange_ids),
            errors=list(self.errors),
        )

    @classmethod
    def from_document(cls, document: dict[str, Any] | None) -> CaptureIntegrity:
        document = document or {}
        return cls(
            complete=bool(document.get("complete", True)),
            calls_missing=int(document.get("calls_missing", 0)),
            recovery_uncertain=bool(document.get("recovery_uncertain", False)),
            delivery_uncertain_exchange_ids=list(document.get("delivery_uncertain") or []),
            errors=list(document.get("errors") or []),
        )


# -- creation ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrajectoryHeader:
    """What creation decided, and nothing that changes afterwards.

    It is the first record in an active journal and the thing a recovery
    rebuilds an aggregate around. ``create_request_hash`` is what makes a
    repeated create idempotent across a restart: the answer to "is this the
    same creation?" has to survive the process that answered it the first time,
    so it is persisted rather than kept in a process-local table.
    """

    id: str
    project: str
    run_id: str | None
    task_id: str | None
    step: int | None
    mode: str
    upstream: dict[str, Any]
    labels: tuple[str, ...]
    annotations: dict[str, Any]
    bodies: str
    source_metadata: dict[str, Any]
    created_at: datetime
    create_request_hash: str

    def document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": self.step,
            "mode": self.mode,
            "upstream": self.upstream,
            "labels": list(self.labels),
            "annotations": self.annotations,
            "bodies": self.bodies,
            "source_metadata": self.source_metadata,
            "created_at": self.created_at.isoformat(),
            "create_request_hash": self.create_request_hash,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> TrajectoryHeader:
        created_at = document["created_at"]
        return cls(
            id=document["id"],
            project=document["project"],
            run_id=document.get("run_id"),
            task_id=document.get("task_id"),
            step=document.get("step"),
            mode=document["mode"],
            upstream=dict(document.get("upstream") or {}),
            labels=tuple(document.get("labels") or ()),
            annotations=dict(document.get("annotations") or {}),
            bodies=document.get("bodies") or "full",
            source_metadata=dict(document.get("source_metadata") or {}),
            created_at=(
                datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
            ),
            create_request_hash=document.get("create_request_hash") or "",
        )


# -- exchanges ---------------------------------------------------------------------
@dataclass(slots=True)
class ExchangeSummary:
    """One committed exchange, as the hot aggregate keeps it.

    ``row`` is the derived exchange -- ids, timings, status, filtered headers,
    parsed parameters, provider ids and the graph association -- and it is the
    same dictionary `/v1/.../exchanges` and the exporters read. The two
    delivery flags are the whole of the TITO ambiguity: ``delivery_confirmed``
    is set once the ASGI send completed, and ``delivery_uncertain`` is set by a
    recovery that found a durable exchange with no confirmation after it.
    """

    id: str
    sequence: int
    row: dict[str, Any]
    delivery_confirmed: bool = True
    delivery_uncertain: bool = False

    def summary(self) -> ExchangeSummary:
        """This exchange without its payload: what stays in hot memory."""
        return ExchangeSummary(
            id=self.id,
            sequence=self.sequence,
            row=self.row,
            delivery_confirmed=self.delivery_confirmed,
            delivery_uncertain=self.delivery_uncertain,
        )


@dataclass(slots=True)
class CapturedExchange(ExchangeSummary):
    """A summary plus the bytes: what a journal record and a committed record
    carry, and what a hot aggregate sheds once the journal record is durable."""

    request_body: bytes = b""
    response_body: bytes = b""
    chunks: dict[str, Any] | None = None
    tokens: dict[str, Any] | None = None


# -- the public document -------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrajectoryDocument:
    """What `/v1/trajectories/{id}` answers, for a hot or a finished trajectory."""

    id: str
    project: str
    run_id: str | None
    task_id: str | None
    step: int | None
    upstream_snapshot: dict[str, Any]
    mode: str
    status: str
    labels: list[str]
    annotations: dict[str, Any]
    bodies: str
    created_at: datetime
    first_event_at: datetime | None
    finish_requested_at: datetime | None
    finished_at: datetime | None
    exchange_count: int
    node_count: int
    calls_after_close: int
    integrity: CaptureIntegrity
    command_result: str | None
    source_metadata: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    revision: int = 0

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "step": self.step,
            "upstream": self.upstream_snapshot,
            "mode": self.mode,
            "status": self.status,
            "labels": self.labels,
            "annotations": self.annotations,
            "bodies": self.bodies,
            "created_at": self.created_at.isoformat(),
            "first_event_at": self.first_event_at.isoformat() if self.first_event_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "capture": {
                "exchange_count": self.exchange_count,
                "node_count": self.node_count,
                "calls_after_close": self.calls_after_close,
                **self.integrity.public(),
            },
            "command_result": self.command_result,
            "source_metadata": self.source_metadata,
            "schema_version": self.schema_version,
            "revision": self.revision,
        }

    def document(self) -> dict[str, Any]:
        """The committed record's copy: the public shape, losslessly."""
        return {
            **self.public(),
            "finish_requested_at": (
                self.finish_requested_at.isoformat() if self.finish_requested_at else None
            ),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> TrajectoryDocument:
        capture = document.get("capture") or {}

        def when(name: str) -> datetime | None:
            value = document.get(name)
            return datetime.fromisoformat(value) if value else None

        created = document["created_at"]
        return cls(
            id=document["id"],
            project=document["project"],
            run_id=document.get("run_id"),
            task_id=document.get("task_id"),
            step=document.get("step"),
            upstream_snapshot=dict(document.get("upstream") or {}),
            mode=document["mode"],
            status=document["status"],
            labels=list(document.get("labels") or []),
            annotations=dict(document.get("annotations") or {}),
            bodies=document.get("bodies") or "full",
            created_at=datetime.fromisoformat(created) if isinstance(created, str) else created,
            first_event_at=when("first_event_at"),
            finish_requested_at=when("finish_requested_at"),
            finished_at=when("finished_at"),
            exchange_count=int(capture.get("exchange_count", 0)),
            node_count=int(capture.get("node_count", 0)),
            calls_after_close=int(capture.get("calls_after_close", 0)),
            integrity=CaptureIntegrity.from_document(capture),
            command_result=document.get("command_result"),
            source_metadata=dict(document.get("source_metadata") or {}),
            schema_version=int(document.get("schema_version", SCHEMA_VERSION)),
            revision=int(document.get("revision", 0)),
        )


# -- metadata -----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MetadataUpdate:
    """A validated metadata edit: merge annotations, add and remove labels.

    Validated once, at the edge, and then applied the same way to a hot
    aggregate and to a committed record -- so a reward that arrives before
    finish and one that arrives after it land identically.
    """

    annotations: dict[str, Any] = field(default_factory=dict)
    remove_annotations: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    remove_labels: tuple[str, ...] = ()

    def empty(self) -> bool:
        return not (self.annotations or self.remove_annotations or self.labels or self.remove_labels)

    def apply(
        self, labels: list[str], annotations: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        """The edited pair. Annotations merge rather than replace, so two
        writers touching different keys do not clobber each other."""
        merged = {**annotations, **self.annotations}
        for key in self.remove_annotations:
            merged.pop(key, None)
        tags = (set(labels) | set(self.labels)) - set(self.remove_labels)
        return sorted(tags), merged

    def document(self) -> dict[str, Any]:
        return {
            "annotations": self.annotations,
            "remove_annotations": list(self.remove_annotations),
            "labels": list(self.labels),
            "remove_labels": list(self.remove_labels),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> MetadataUpdate:
        return cls(
            annotations=dict(document.get("annotations") or {}),
            remove_annotations=tuple(document.get("remove_annotations") or ()),
            labels=tuple(document.get("labels") or ()),
            remove_labels=tuple(document.get("remove_labels") or ()),
        )


def validate_annotations(values: dict[str, Any]) -> dict[str, Any]:
    for name in values:
        if not name or not name.strip():
            raise MetadataError("annotation names must be non-empty")
    return values


def validate_labels(labels: list[str]) -> list[str]:
    cleaned: list[str] = []
    for label in labels:
        if not label or not label.strip():
            raise MetadataError("labels must be non-empty strings")
        if "=" in label:
            raise MetadataError(
                f"label {label!r} looks like a key=value pair; labels are bare tags, "
                "use an annotation for a value"
            )
        if label not in cleaned:
            cleaned.append(label)
    return cleaned


def metadata_update(
    *,
    annotations: dict[str, Any] | None = None,
    remove_annotations: list[str] | None = None,
    labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
) -> MetadataUpdate:
    """Validate an edit at the edge. Raises `MetadataError`."""
    return MetadataUpdate(
        annotations=validate_annotations(dict(annotations or {})),
        remove_annotations=tuple(remove_annotations or ()),
        labels=tuple(validate_labels(list(labels or []))),
        remove_labels=tuple(remove_labels or ()),
    )


# -- the hot aggregate -----------------------------------------------------------------
@dataclass
class ActiveTrajectory:
    """One locally hot trajectory: everything this process knows about it.

    Owned by one process and mutated between awaits, so it needs no lock of its
    own -- what serializes turns on a TITO trajectory is the session lock, and
    what serializes its journal appends is the commit coordinator's per
    trajectory chain.

    Beyond the aggregate the design names, this carries the mutable metadata
    (labels and annotations are edited after creation, and the header is not),
    the lifecycle timestamps a document reports, and ``sequence``, which is the
    per-trajectory exchange counter that orders its journal.
    """

    header: TrajectoryHeader
    graph: ConversationGraph = field(default_factory=ConversationGraph)
    exchanges: list[ExchangeSummary] = field(default_factory=list)
    sequence: int = 0
    status: str = "created"
    integrity: CaptureIntegrity = field(default_factory=CaptureIntegrity)
    in_flight: int = 0
    pending_commit: Future | None = None

    labels: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)
    command_result: str | None = None
    first_event_at: datetime | None = None
    finish_requested_at: datetime | None = None
    finished_at: datetime | None = None
    finish_request_hash: str = ""
    revision: int = 0
    calls_after_close: int = 0
    # Set when the journal this aggregate came from was written by an earlier
    # process. Only text capture turns it into recovery uncertainty; a TITO
    # trajectory's doubt is per exchange and is carried by the delivery flags.
    recovered: bool = False
    # Created only while `finish` is waiting for turns already in flight.
    _settled: asyncio.Event | None = field(default=None, repr=False, compare=False)

    @classmethod
    def create(cls, header: TrajectoryHeader) -> ActiveTrajectory:
        return cls(
            header=header,
            labels=list(header.labels),
            annotations=dict(header.annotations),
        )

    @property
    def id(self) -> str:
        return self.header.id

    @property
    def accepting(self) -> bool:
        """`created` and `active` take traffic; nothing else does."""
        return self.status in LIVE_STATUSES

    # -- turns in flight --------------------------------------------------------
    def open_turn(self) -> None:
        self.in_flight += 1

    def close_turn(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if self.in_flight == 0 and self._settled is not None:
            self._settled.set()

    async def settle_turns(self, *, timeout: float) -> bool:
        """Wait for the turns already accepted on this trajectory.

        A caller that finishes the instant its last response arrives must still
        find that response in the graph. Bounded: a turn that outlives the
        grace lands late, and says so on the exchange.
        """
        if not self.in_flight:
            return True
        waiter = self._settled = self._settled or asyncio.Event()
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
        except TimeoutError:
            return False
        finally:
            self._settled = None
        return True

    # -- mutation -------------------------------------------------------------
    def activate(self, at: datetime) -> None:
        if self.status == "created":
            self.status = "active"
        self.first_event_at = self.first_event_at or at

    def next_sequence(self) -> int:
        """Claim the next per-trajectory exchange sequence."""
        value = self.sequence
        self.sequence += 1
        return value

    def add_exchange(self, exchange: ExchangeSummary, delta: Any | None, at: datetime) -> None:
        """Attach a committed exchange and the graph change it made.

        Idempotent by exchange id, because a recovery replays the same journal
        records the live path already applied when it is resuming a trajectory
        this process itself wrote.
        """
        if any(existing.id == exchange.id for existing in self.exchanges):
            return
        self.exchanges.append(exchange)
        self.sequence = max(self.sequence, exchange.sequence + 1)
        if exchange.row.get("late"):
            self.calls_after_close += 1
        if delta is not None:
            self.graph.apply(delta, at=at)
        self.activate(exchange.row.get("request_start_at") or at)

    def confirm_delivery(self, exchange_id: str) -> None:
        for exchange in self.exchanges:
            if exchange.id == exchange_id:
                exchange.delivery_confirmed = True
                exchange.delivery_uncertain = False
                if exchange_id in self.integrity.delivery_uncertain_exchange_ids:
                    self.integrity.delivery_uncertain_exchange_ids.remove(exchange_id)
                return

    def record_gap(self, count: int, reason: str | None, at: datetime) -> None:
        """Mark capture incomplete, or -- with ``count`` of zero -- uncertain.

        A zero-count gap says capture cannot vouch for this trajectory without
        claiming anything is missing, which is exactly what a text trajectory
        picked up by a replacement process is: the previous process may have
        served an exchange inside its commit window, and nothing identifies
        whether it did. Written down rather than kept in memory, so a viewer
        reading the journal sees the same doubt the writer has.
        """
        if count <= 0:
            self.integrity.recovery_uncertain = True
        else:
            self.integrity.calls_missing += count
            self.integrity.complete = False
        if reason and reason not in self.integrity.errors:
            self.integrity.errors.append(reason)
        self.activate(at)

    def apply_metadata(self, update: MetadataUpdate) -> None:
        # An empty edit changes nothing, and a revision that moved without a
        # change would make every finish look like a correction.
        if update.empty():
            return
        self.labels, self.annotations = update.apply(self.labels, self.annotations)
        self.revision += 1

    def request_finish(
        self, *, command_result: str | None, request_hash: str, at: datetime
    ) -> None:
        if self.status not in DONE_STATUSES:
            self.status = "finalizing"
        self.command_result = command_result or self.command_result
        self.finish_requested_at = self.finish_requested_at or at
        self.finish_request_hash = request_hash or self.finish_request_hash

    def poison(self, reason: str, at: datetime) -> None:
        """Stop a trajectory whose graph can no longer be extended correctly.

        ``complete`` is deliberately left alone. A poison is not a capture
        failure -- the turn that caused it was captured exactly, and so was
        everything before it. What ended is the trajectory, which `status`
        says, and why, which `errors` says. Conflating the two would make
        every poisoned trajectory look like one capture had lost data from.
        """
        if self.status in DONE_STATUSES:
            return
        self.status = "poisoned"
        self.command_result = self.command_result or f"token capture: {reason}"
        self.finished_at = self.finished_at or at
        if reason not in self.integrity.errors:
            self.integrity.errors.append(reason)

    def shed_payloads(self) -> None:
        """Drop raw bodies and token arrays from hot memory.

        Called once an exchange's journal record is durable: from then on the
        journal is where the bytes live, and holding them here would bound a
        run by its request bytes rather than by its graph.
        """
        self.exchanges = [
            exchange.summary() if isinstance(exchange, CapturedExchange) else exchange
            for exchange in self.exchanges
        ]

    # -- reads ------------------------------------------------------------------
    def document(self, *, status: str | None = None) -> TrajectoryDocument:
        header = self.header
        return TrajectoryDocument(
            id=header.id,
            project=header.project,
            run_id=header.run_id,
            task_id=header.task_id,
            step=header.step,
            upstream_snapshot=dict(header.upstream),
            mode=header.mode,
            status=status or self.status,
            labels=sorted(self.labels),
            annotations=copy.deepcopy(self.annotations),
            bodies=header.bodies,
            created_at=header.created_at,
            first_event_at=self.first_event_at,
            finish_requested_at=self.finish_requested_at,
            finished_at=self.finished_at,
            exchange_count=len(self.exchanges),
            node_count=len(self.graph),
            calls_after_close=self.calls_after_close,
            integrity=self.integrity.copy(),
            command_result=self.command_result,
            source_metadata=dict(header.source_metadata),
            revision=self.revision,
        )

    def exchange_rows(self) -> list[dict[str, Any]]:
        """The exchange rows in per-trajectory sequence order."""
        return [
            exchange.row
            for exchange in sorted(self.exchanges, key=lambda item: (item.sequence, item.id))
        ]

    def exchange_by_response_id(self, provider_response_id: str) -> str | None:
        for exchange in self.exchanges:
            if exchange.row.get("provider_response_id") == provider_response_id:
                return exchange.id
        return None


# -- the finished artifact ---------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryRecord:
    """One finished trajectory, compiled once and never replayed.

    Everything a viewer path or an exporter needs is materialized here: the
    public document, the exchanges with their payloads, the graph nodes and
    their order, and -- in token mode -- the exact token arrays on the nodes.
    Nothing reading this record runs a reducer, and nothing invokes a tokenizer.

    ``revision`` and ``finish_request_hash`` are what make rewriting it safe. A
    metadata edit after finish bumps the revision and replaces the file; a
    retried finish compares its hash and either returns this record or is a
    `409`.
    """

    record_version: int
    schema_version: int
    derivation_version: int
    revision: int
    finish_request_hash: str
    trajectory: TrajectoryDocument
    exchanges: tuple[CapturedExchange, ...]
    nodes: tuple[GraphNode, ...]
    node_order: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.trajectory.id

    def with_metadata(self, update: MetadataUpdate) -> TrajectoryRecord:
        """This record, edited. A new value -- the file is replaced atomically."""
        labels, annotations = update.apply(
            list(self.trajectory.labels), dict(self.trajectory.annotations)
        )
        from dataclasses import replace

        return replace(
            self,
            revision=self.revision + 1,
            trajectory=replace(
                self.trajectory,
                labels=labels,
                annotations=annotations,
                revision=self.revision + 1,
            ),
        )


def empty_record(document: TrajectoryDocument) -> TrajectoryRecord:
    """A record with no exchanges: what a trajectory that made no calls is."""
    return TrajectoryRecord(
        record_version=RECORD_FORMAT_VERSION,
        schema_version=SCHEMA_VERSION,
        derivation_version=DERIVATION_VERSION,
        revision=document.revision,
        finish_request_hash="",
        trajectory=document,
        exchanges=(),
        nodes=(),
        node_order=(),
    )
