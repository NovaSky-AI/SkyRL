"""Versioned profiling receipts shared by timing and transport benchmarks."""

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"
DERIVATION_VERSION = "1.0.0"

Arm = Literal["baseline_a", "native_candidate"]
ProfilerMode = Literal["none", "trainer", "receiver"]
Lifecycle = Literal["cold", "warm"]
PhaseKind = Literal[
    "cold_start", "generation_zero", "excluded_warmup", "recurring", "housekeeping"
]
Operation = Literal[
    "scheduler_allocation",
    "service_connection",
    "model_creation",
    "inference_engine_construction",
    "base_weight_loading",
    "compilation",
    "cuda_graph_capture",
    "kv_cache_initialization",
    "transport_layout_initialization",
    "adapter_publication",
    "adapter_activation",
    "sample",
    "reference_forward",
    "training_forward_backward",
    "optimizer",
    "profiler_start",
    "profiler_stop",
    "trace_export_flush",
    "trace_summary_processing",
    "checkpoint",
    "unload",
]


class ReceiptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelIdentity(ReceiptModel):
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CodeProvenance(ReceiptModel):
    repository_url: str = Field(min_length=1)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty: bool
    image_uri: str = Field(min_length=1)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class TransportIdentity(ReceiptModel):
    implementation: str = Field(min_length=1)
    revision: str = Field(min_length=1)


class PhaseClassification(ReceiptModel):
    lifecycle: Lifecycle
    kind: PhaseKind
    optimizer_step: int | None = Field(default=None, ge=0)


class ClientTiming(ReceiptModel):
    start_monotonic_ns: int = Field(ge=0)
    end_monotonic_ns: int = Field(ge=0)
    duration_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_duration(self):
        if self.end_monotonic_ns < self.start_monotonic_ns:
            raise ValueError("client monotonic end precedes start")
        if self.duration_ns != self.end_monotonic_ns - self.start_monotonic_ns:
            raise ValueError("client duration does not match monotonic endpoints")
        return self


class ServerSpan(ReceiptModel):
    span_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    adapter_generation: int = Field(ge=0)
    start_monotonic_ns: int = Field(ge=0)
    end_monotonic_ns: int = Field(ge=0)
    rank: int = Field(ge=0)
    node: str = Field(min_length=1)
    owning_component: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_end(self):
        if self.end_monotonic_ns < self.start_monotonic_ns:
            raise ValueError("server span end precedes start")
        return self

    @property
    def duration_ns(self) -> int:
        return self.end_monotonic_ns - self.start_monotonic_ns


class AdapterTransfer(ReceiptModel):
    logical_bytes: int | None = Field(default=None, ge=0)
    transferred_bytes: int | None = Field(default=None, ge=0)
    source_dtype: str | None = None
    destination_dtype: str | None = None
    layout_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")


class MemoryObservation(ReceiptModel):
    gpu_allocated_bytes: int | None = Field(default=None, ge=0)
    gpu_reserved_bytes: int | None = Field(default=None, ge=0)
    cgroup_current_bytes: int | None = Field(default=None, ge=0)
    cgroup_peak_bytes: int | None = Field(default=None, ge=0)


class TraceArtifact(ReceiptModel):
    path: str
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_rank: int | None = Field(default=None, ge=0)
    window: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_complete_or_disabled(self):
        values = (
            self.path,
            self.size_bytes,
            self.sha256,
            self.selected_rank,
            self.window,
        )
        if any(value not in (None, "") for value in values) and any(
            value in (None, "") for value in values
        ):
            raise ValueError("trace artifact metadata must be complete when supplied")
        return self


class OperationOutcome(ReceiptModel):
    success: bool
    error_type: str | None = None
    error: str | None = None
    active_generation_after: int | None = Field(default=None, ge=0)
    first_sample_generation: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_error(self):
        if self.success and (self.error_type is not None or self.error is not None):
            raise ValueError("successful operation cannot carry an error")
        if not self.success and (not self.error_type or not self.error):
            raise ValueError("failed operation must carry error type and message")
        return self


class PhaseReceipt(ReceiptModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    operation: Operation
    model: ModelIdentity
    provenance: CodeProvenance
    arm: Arm
    transport: TransportIdentity
    profiler_mode: ProfilerMode
    classification: PhaseClassification
    adapter_generation: int = Field(ge=0)
    client_timing: ClientTiming
    server_spans: tuple[ServerSpan, ...]
    unattributed_duration_ns: int = Field(ge=0)
    adapter_transfer: AdapterTransfer
    outcome: OperationOutcome
    memory: MemoryObservation
    trace: TraceArtifact

    @model_validator(mode="after")
    def validate_receipt(self):
        validate_distributed_spans(self.server_spans)
        if self.unattributed_duration_ns > self.client_timing.duration_ns:
            raise ValueError(
                "unattributed duration exceeds the client operation duration"
            )
        if any(
            span.adapter_generation != self.adapter_generation
            for span in self.server_spans
        ):
            raise ValueError(
                "server span adapter generation does not match its receipt"
            )
        if self.profiler_mode == "none" and self.trace.path:
            raise ValueError(
                "profiler-disabled operation cannot carry a trace artifact"
            )
        if (
            self.operation in {"adapter_publication", "adapter_activation"}
            and self.outcome.success
        ):
            if self.outcome.active_generation_after != self.adapter_generation:
                raise ValueError(
                    "successful publication or activation must report its active generation"
                )
        if self.operation == "sample" and self.outcome.success:
            if self.outcome.first_sample_generation != self.adapter_generation:
                raise ValueError(
                    "successful sample must report the generation it observed"
                )
        return self


def classify_phase(
    operation: Operation, adapter_generation: int, optimizer_step: int | None
) -> PhaseClassification:
    if operation in {
        "scheduler_allocation",
        "service_connection",
        "model_creation",
        "inference_engine_construction",
        "base_weight_loading",
        "compilation",
        "cuda_graph_capture",
        "kv_cache_initialization",
        "transport_layout_initialization",
    }:
        return PhaseClassification(
            lifecycle="cold", kind="cold_start", optimizer_step=optimizer_step
        )
    if adapter_generation == 0 and operation in {
        "adapter_publication",
        "adapter_activation",
        "sample",
    }:
        return PhaseClassification(
            lifecycle="cold", kind="generation_zero", optimizer_step=optimizer_step
        )
    if operation in {
        "profiler_start",
        "profiler_stop",
        "trace_export_flush",
        "trace_summary_processing",
        "checkpoint",
        "unload",
    }:
        return PhaseClassification(
            lifecycle="warm", kind="housekeeping", optimizer_step=optimizer_step
        )
    if optimizer_step == 0:
        return PhaseClassification(
            lifecycle="warm", kind="excluded_warmup", optimizer_step=optimizer_step
        )
    if optimizer_step is None:
        raise ValueError(f"optimizer_step is required for recurring {operation}")
    return PhaseClassification(
        lifecycle="warm", kind="recurring", optimizer_step=optimizer_step
    )


@contextmanager
def measure_monotonic_client(clock=None):
    """Yield a holder populated with a validated monotonic interval on exit."""
    if clock is None:
        clock = time.monotonic_ns
    holder = {}
    start = clock()
    try:
        yield holder
    finally:
        end = clock()
        holder["timing"] = ClientTiming(
            start_monotonic_ns=start,
            end_monotonic_ns=end,
            duration_ns=end - start,
        )


def build_phase_receipt(**values) -> PhaseReceipt:
    return PhaseReceipt.model_validate(values)


def receipt_json_schema() -> dict:
    return PhaseReceipt.model_json_schema()


def validate_distributed_spans(spans) -> tuple[ServerSpan, ...]:
    validated = tuple(ServerSpan.model_validate(span) for span in spans)
    span_ids = [span.span_id for span in validated]
    if len(span_ids) != len(set(span_ids)):
        raise ValueError("server span_id values must be unique within a receipt")
    return validated


def summarize_distributed_spans(spans) -> dict:
    """Keep rank-local durations separate; monotonic clocks are not cross-node clocks."""
    validated = validate_distributed_spans(spans)
    return {
        "aggregation": "not_summed",
        "combined_duration_ns": None,
        "spans": [
            {
                "span_id": span.span_id,
                "rank": span.rank,
                "node": span.node,
                "duration_ns": span.duration_ns,
            }
            for span in validated
        ],
    }


def derive_receipt_table(receipts, input_paths) -> dict:
    validated = [PhaseReceipt.model_validate(receipt) for receipt in receipts]
    return {
        "derivation_version": DERIVATION_VERSION,
        "inputs": [str(path) for path in input_paths],
        "rows": [
            {
                "run_id": receipt.run_id,
                "operation_id": receipt.operation_id,
                "operation": receipt.operation,
                "classification": receipt.classification.model_dump(mode="json"),
                "client_duration_ns": receipt.client_timing.duration_ns,
                "server_attribution": summarize_distributed_spans(receipt.server_spans),
                "success": receipt.outcome.success,
            }
            for receipt in validated
        ],
    }


def write_receipts(path: Path, receipts) -> None:
    """Create an immutable canonical JSONL receipt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        for receipt in receipts:
            validated = PhaseReceipt.model_validate(receipt)
            output.write(validated.model_dump_json() + "\n")
        output.flush()
        os.fsync(output.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksum_manifest(path: Path, files, root: Path) -> None:
    entries = []
    for file_path in sorted(
        (Path(item) for item in files),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = file_path.relative_to(root)
        entries.append(f"{sha256_file(file_path)}  {relative.as_posix()}\n")
    with path.open("x") as output:
        output.writelines(entries)
        output.flush()
        os.fsync(output.fileno())


def verify_checksum_manifest(path: Path, root: Path) -> None:
    for line in path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        candidate = (root / relative).resolve()
        if root.resolve() not in candidate.parents:
            raise ValueError(f"manifest path escapes release directory: {relative}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
