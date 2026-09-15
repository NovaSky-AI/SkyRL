"""Structured server timing for boundaries owned by the Tinker backend."""

import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SERVER_STAGE_SCHEMA_VERSION = "1.0.0"
SERVER_STAGE_PREFIX = "SKYRL_SERVER_STAGE "
ServerStage = Literal["inference_engine_construction", "sampler_weight_sync"]


@dataclass(frozen=True)
class ServerRequestContext:
    request_id: str
    publication_id: str
    model_id: str
    adapter_generation: int | None


_REQUEST_CONTEXT: ContextVar[ServerRequestContext | None] = ContextVar("skyrl_server_request_context", default=None)


class ServerStageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = SERVER_STAGE_SCHEMA_VERSION
    span_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    adapter_generation: int | None = Field(ge=0)
    stage: ServerStage
    cold: bool
    owning_component: Literal["tinker_engine_controller"]
    rank: int = Field(ge=0)
    node: str = Field(min_length=1)
    start_monotonic_ns: int = Field(ge=0)
    end_monotonic_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)
    success: bool
    error_type: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_interval_and_outcome(self):
        if self.end_monotonic_ns < self.start_monotonic_ns:
            raise ValueError("server stage end precedes start")
        if self.duration_ns != self.end_monotonic_ns - self.start_monotonic_ns:
            raise ValueError("server stage duration does not match monotonic endpoints")
        if self.success and (self.error_type is not None or self.error is not None):
            raise ValueError("successful server stage cannot carry an error")
        if not self.success and (not self.error_type or not self.error):
            raise ValueError("failed server stage must carry error type and message")
        return self


def server_stage_json_schema() -> dict:
    return ServerStageRecord.model_json_schema()


@contextmanager
def server_request_context(request_id: str, model_id: str, request_data: dict):
    """Correlate backend-owned timing with one durable Tinker request."""
    session_seq_id = request_data.get("sampling_session_seq_id")
    seq_id = request_data.get("seq_id")
    if session_seq_id is not None and seq_id is not None:
        publication_id = f"{model_id}/sampling-session-{session_seq_id}/sequence-{seq_id}"
    else:
        publication_id = f"{model_id}/request-{request_id}"
    token = _REQUEST_CONTEXT.set(
        ServerRequestContext(
            request_id=request_id,
            publication_id=publication_id,
            model_id=model_id,
            adapter_generation=session_seq_id,
        )
    )
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


def emit_server_stage(record: ServerStageRecord) -> None:
    """Write one unwrapped JSON line for durable log collection."""
    print(SERVER_STAGE_PREFIX + record.model_dump_json(), flush=True)


@contextmanager
def record_server_stage(stage: ServerStage, model_id: str, cold: bool, clock=None, node: str | None = None):
    """Emit one validated rank-local stage without changing operation failures."""
    context = _REQUEST_CONTEXT.get()
    if context is None:
        yield
        return
    if context.model_id != model_id:
        raise ValueError("server timing model differs from the active request")
    clock = clock or time.monotonic_ns
    started = clock()
    operation_error = None
    try:
        yield
    except BaseException as error:
        operation_error = error
        raise
    finally:
        ended = clock()
        try:
            record = ServerStageRecord(
                span_id=f"{context.request_id}/{stage}/{started}",
                request_id=context.request_id,
                publication_id=context.publication_id,
                model_id=model_id,
                adapter_generation=context.adapter_generation,
                stage=stage,
                cold=cold,
                owning_component="tinker_engine_controller",
                rank=0,
                node=node or socket.gethostname(),
                start_monotonic_ns=started,
                end_monotonic_ns=ended,
                duration_ns=ended - started,
                success=operation_error is None,
                error_type=type(operation_error).__name__ if operation_error else None,
                error=str(operation_error) if operation_error else None,
            )
            emit_server_stage(record)
        except Exception as timing_error:
            if operation_error is None:
                raise
            operation_error.add_note(f"Server timing emission failed: {timing_error}")


def collect_server_stage_records(lines) -> tuple[ServerStageRecord, ...]:
    """Validate structured stage records embedded in ordinary server logs."""
    if isinstance(lines, str):
        lines = lines.splitlines()
    records = []
    for line in lines:
        _, marker, payload = line.partition(SERVER_STAGE_PREFIX)
        if marker:
            records.append(ServerStageRecord.model_validate_json(payload))
    span_ids = [record.span_id for record in records]
    if len(span_ids) != len(set(span_ids)):
        raise ValueError("server stage span_id values must be unique")
    return tuple(records)


def derive_server_stage_table(records) -> dict:
    """Retain raw stages and derive rank-max summaries without summing ranks."""
    validated = tuple(ServerStageRecord.model_validate(record) for record in records)
    grouped = {}
    for record in validated:
        key = (record.publication_id, record.stage, record.cold)
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (publication_id, stage, cold), group in sorted(grouped.items()):
        ranks = {}
        for record in group:
            ranks.setdefault(record.rank, []).append(record.duration_ns)
        summaries.append(
            {
                "publication_id": publication_id,
                "stage": stage,
                "cold": cold,
                "aggregation": "rank_max_not_sum",
                "rank_max_duration_ns": max(max(durations) for durations in ranks.values()),
                "rank_durations_ns": {str(rank): durations for rank, durations in sorted(ranks.items())},
            }
        )
    return {
        "schema_version": SERVER_STAGE_SCHEMA_VERSION,
        "records": [record.model_dump(mode="json") for record in validated],
        "summaries": summaries,
        "unavailable_stages": [
            "base_weight_loading",
            "compilation",
            "cuda_graph_capture",
            "kv_cache_initialization",
            "transport_layout_initialization",
            "adapter_activation",
        ],
    }
