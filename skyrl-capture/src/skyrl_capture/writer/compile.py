"""Compiling a finished trajectory into the artifact a reader opens.

Finish reads the trajectory's own journal back, because that -- not memory --
is what holds the captured bytes: an exchange's payload leaves the hot
aggregate as soon as its record is durable. What memory still holds that the
journal may not is the integrity the disk refused to take, so the two are
merged here rather than one of them being trusted alone.

The result is deterministic. Given the same journal and the same finish
request, two processes compile the same bytes, which is what makes a retry
after a crash safe: whichever attempt gets to the rename first, the record is
the same record.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from skyrl_capture.domain.records import (
    ActiveTrajectory,
    CapturedExchange,
    TrajectoryRecord,
)
from skyrl_capture.persistence.journal import ActiveRecord
from skyrl_capture.version import DERIVATION_VERSION, RECORD_FORMAT_VERSION, SCHEMA_VERSION


def merge_integrity(committed: ActiveTrajectory, hot: ActiveTrajectory) -> None:
    """Carry what memory knows and the journal does not.

    A gap the disk refused is the case this exists for: capture observed a call
    it could not write, and the write it could not do includes the record that
    would have said so. The final artifact must still report it, so the larger
    of the two counts wins and the errors are unioned.
    """
    integrity = committed.integrity
    integrity.calls_missing = max(integrity.calls_missing, hot.integrity.calls_missing)
    integrity.recovery_uncertain = integrity.recovery_uncertain or hot.integrity.recovery_uncertain
    integrity.complete = integrity.complete and hot.integrity.complete and not integrity.calls_missing
    for error in hot.integrity.errors:
        if error not in integrity.errors:
            integrity.errors.append(error)
    for exchange_id in hot.integrity.delivery_uncertain_exchange_ids:
        if exchange_id not in integrity.delivery_uncertain_exchange_ids:
            integrity.delivery_uncertain_exchange_ids.append(exchange_id)
    uncertain = set(integrity.delivery_uncertain_exchange_ids)
    for exchange in committed.exchanges:
        if exchange.id in uncertain:
            exchange.delivery_uncertain = True


def merge_metadata(committed: ActiveTrajectory, hot: ActiveTrajectory) -> None:
    """Take the live aggregate's metadata and lifecycle stamps.

    The journal carries every metadata edit, so these usually agree. They
    disagree when the edit that closed the trajectory is the one being
    processed, and the live answer is the right one.
    """
    committed.labels = sorted(set(committed.labels) | set(hot.labels))
    committed.annotations = {**committed.annotations, **hot.annotations}
    committed.command_result = hot.command_result or committed.command_result
    committed.finish_requested_at = hot.finish_requested_at or committed.finish_requested_at
    committed.first_event_at = committed.first_event_at or hot.first_event_at
    committed.revision = max(committed.revision, hot.revision)
    committed.calls_after_close = max(committed.calls_after_close, hot.calls_after_close)


def compile_record(
    active: ActiveTrajectory,
    *,
    status: str,
    finished_at: datetime,
    finish_request_hash: str,
) -> TrajectoryRecord:
    """The committed artifact for one trajectory.

    Exchanges come out in per-trajectory sequence order and nodes in graph
    order, so a reader gets the same traversal every exporter already assumes
    without sorting anything itself.
    """
    active.finished_at = active.finished_at or finished_at
    document = active.document(status=status)
    ordered = active.graph.ordered()
    exchanges = sorted(active.exchanges, key=lambda item: (item.sequence, item.id))
    return TrajectoryRecord(
        record_version=RECORD_FORMAT_VERSION,
        schema_version=SCHEMA_VERSION,
        derivation_version=DERIVATION_VERSION,
        revision=document.revision,
        finish_request_hash=finish_request_hash,
        trajectory=document,
        exchanges=tuple(_captured(exchange) for exchange in exchanges),
        nodes=tuple(ordered),
        node_order=tuple(node.id for node in ordered),
    )


def _captured(exchange: Any) -> CapturedExchange:
    """A `CapturedExchange`, even for a summary whose payload was shed.

    A shed payload is not an error: the journal held the bytes and the journal
    is what was read back. This only covers the case where an exchange was
    marked in memory and never written -- there are no bytes to compile, and
    the trajectory already reports the gap.
    """
    if isinstance(exchange, CapturedExchange):
        return exchange
    return CapturedExchange(
        id=exchange.id,
        sequence=exchange.sequence,
        row=exchange.row,
        delivery_confirmed=exchange.delivery_confirmed,
        delivery_uncertain=exchange.delivery_uncertain,
    )


def journal_aggregate(
    records: list[ActiveRecord], *, hot: ActiveTrajectory
) -> ActiveTrajectory | None:
    """The aggregate a journal describes, merged with what memory knows."""
    from skyrl_capture.persistence.active import rebuild

    committed = rebuild(records, recovered=hot.recovered)
    if committed is None:
        return None
    merge_metadata(committed, hot)
    merge_integrity(committed, hot)
    committed.status = hot.status
    return committed
