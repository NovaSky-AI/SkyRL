"""In-memory training-operation queue and local API-to-engine transport.

The API process is the single owner of operation identity, ordering, payload
retention, and future completion.  The background engine claims ready work over
one Unix-domain socket.  Large model-pass payloads therefore never enter the
database while the existing process boundary remains explicit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import struct
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.extra.in_memory_future_store import InMemoryFutureStore

_FRAME_LENGTH = struct.Struct("!Q")
_STREAM_CHUNK_BYTES = 1024 * 1024
_DEFAULT_GAP_RESERVE_BYTES = 64 * 1024**2

TRANSPORTED_REQUEST_TYPES = frozenset(
    {
        types.RequestType.FORWARD,
        types.RequestType.FORWARD_BACKWARD,
        types.RequestType.OPTIM_STEP,
        types.RequestType.SAVE_WEIGHTS,
        types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
        types.RequestType.LOAD_WEIGHTS,
    }
)


class OperationBackpressure(RuntimeError):
    """The bounded queue cannot accept another non-gap-filling operation."""


class OperationExpired(RuntimeError):
    """A retry arrived after its compact operation record was released."""


class OperationState(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class OperationPayloadFormat(str, Enum):
    JSON = "json"
    PROTO = "proto"
    PROTO_ZSTD = "proto_zstd"


@dataclass
class Operation:
    request_id: int
    request_type: types.RequestType
    model_id: str
    seq_id: int | None
    payload: bytes
    payload_format: OperationPayloadFormat
    fingerprint: str
    arrival: int
    state: OperationState = OperationState.QUEUED

    @property
    def terminal(self) -> bool:
        return self.state in (OperationState.COMPLETED, OperationState.FAILED)


@dataclass
class _ModelQueue:
    next_seq_id: int | None
    by_seq_id: dict[int, Operation] = field(default_factory=dict)
    unsequenced: deque[Operation] = field(default_factory=deque)
    retired_through_seq_id: int = 0
    pending_count: int = 0

    def fills_blocking_gap(self, seq_id: int | None) -> bool:
        if seq_id is None or not self.by_seq_id:
            return False
        if self.next_seq_id is not None:
            return seq_id == self.next_seq_id and any(value > seq_id for value in self.by_seq_id)
        lowest = min(self.by_seq_id)
        return seq_id < lowest


class TrainingOperationQueue:
    """Single-owner, gap-buffered queue for model-pass and optimizer work.

    Queue methods do not await. FastAPI therefore updates the queue from one
    event-loop thread at a time.
    """

    def __init__(
        self,
        future_store: InMemoryFutureStore,
        *,
        max_pending_per_model: int = 512,
        max_payload_bytes: int = 2 * 1024**3,
        gap_reserve_bytes: int = _DEFAULT_GAP_RESERVE_BYTES,
        max_terminal_records: int = 10_000,
    ) -> None:
        self.future_store = future_store
        self.max_pending_per_model = max_pending_per_model
        self.max_payload_bytes = max_payload_bytes
        self.gap_reserve_bytes = min(gap_reserve_bytes, max_payload_bytes)
        self.max_terminal_records = max_terminal_records
        self._models: dict[str, _ModelQueue] = {}
        self._by_request_id: dict[int, Operation] = {}
        self._terminal_order: deque[int] = deque()
        self._payload_bytes = 0
        self._pending_count = 0
        self._arrival = 0

    @property
    def payload_bytes(self) -> int:
        return self._payload_bytes

    @property
    def pending_count(self) -> int:
        return self._pending_count

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def has_operations(self, model_id: str) -> bool:
        queue = self._models.get(model_id)
        if queue is None:
            return False
        if queue.retired_through_seq_id > 0:
            return True
        return any(operation.model_id == model_id for operation in self._by_request_id.values())

    def register_model(self, model_id: str, next_seq_id: int | None) -> None:
        retired_through_seq_id = 0 if next_seq_id is None else max(0, next_seq_id - 1)
        self._models.setdefault(
            model_id,
            _ModelQueue(
                next_seq_id=next_seq_id,
                retired_through_seq_id=retired_through_seq_id,
            ),
        )

    @staticmethod
    def _fingerprint(
        request_type: types.RequestType,
        payload: bytes,
        payload_format: OperationPayloadFormat,
    ) -> str:
        return hashlib.sha256(
            request_type.value.encode() + b"\0" + payload_format.value.encode() + b"\0" + payload
        ).hexdigest()

    def existing_request_id(
        self,
        request_type: types.RequestType,
        model_id: str,
        seq_id: int | None,
        payload: bytes,
        payload_format: OperationPayloadFormat = OperationPayloadFormat.JSON,
    ) -> int | None:
        """Return the original request ID for an exact sequenced retry."""
        if seq_id is None:
            return None
        queue = self._models.get(model_id)
        if queue is None:
            return None
        if seq_id <= queue.retired_through_seq_id:
            raise OperationExpired(f"Training request sequence {seq_id} is older than the retained retry history")
        if (existing := queue.by_seq_id.get(seq_id)) is None:
            return None
        if existing.terminal and not self.future_store.has_future(existing.request_id):
            raise OperationExpired(f"Training request sequence {seq_id} is older than the retained retry history")
        if existing.request_type != request_type or existing.fingerprint != self._fingerprint(
            request_type, payload, payload_format
        ):
            raise ValueError("Training request sequence number was reused")
        return existing.request_id

    def enqueue(
        self,
        request_type: types.RequestType,
        model_id: str,
        seq_id: int | None,
        payload: bytes,
        payload_format: OperationPayloadFormat = OperationPayloadFormat.JSON,
    ) -> int:
        """Admit an operation or return the original future for an exact retry."""
        queue = self._models.get(model_id)
        if queue is None:
            raise KeyError(f"model '{model_id}' is not registered in the operation queue")

        fingerprint = self._fingerprint(request_type, payload, payload_format)
        if (
            request_id := self.existing_request_id(
                request_type,
                model_id,
                seq_id,
                payload,
                payload_format,
            )
        ) is not None:
            return request_id

        gap_filler = queue.fills_blocking_gap(seq_id)
        ready_sequence = seq_id is None or queue.next_seq_id is None or seq_id == queue.next_seq_id
        if queue.pending_count >= self.max_pending_per_model:
            raise OperationBackpressure(
                f"model '{model_id}' has {self.max_pending_per_model} training operations pending"
            )
        if not ready_sequence and not gap_filler and queue.pending_count >= self.max_pending_per_model - 1:
            raise OperationBackpressure(f"model '{model_id}' is reserving one pending slot for its next sequence")

        new_payload_bytes = self._payload_bytes + len(payload)
        if new_payload_bytes > self.max_payload_bytes:
            raise OperationBackpressure(
                f"training operation queue holds {self._payload_bytes} payload bytes "
                f"(limit {self.max_payload_bytes})"
            )
        if (
            not ready_sequence
            and not gap_filler
            and new_payload_bytes > self.max_payload_bytes - self.gap_reserve_bytes
        ):
            raise OperationBackpressure("training operation queue is reserving payload space for its next sequence")

        request_id = self.future_store.create_future(request_type=request_type)
        self._arrival += 1
        operation = Operation(
            request_id=request_id,
            request_type=request_type,
            model_id=model_id,
            seq_id=seq_id,
            payload=payload,
            payload_format=payload_format,
            fingerprint=fingerprint,
            arrival=self._arrival,
        )
        self._by_request_id[request_id] = operation
        if seq_id is None:
            queue.unsequenced.append(operation)
        else:
            queue.by_seq_id[seq_id] = operation
        queue.pending_count += 1
        self._pending_count += 1
        self._payload_bytes += len(payload)
        return request_id

    def claim_ready(self) -> list[Operation]:
        """Claim one same-type batch for one model in sequence order."""
        candidates: list[Operation] = []
        for queue in self._models.values():
            candidate = self._next_open(queue)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return []

        first = min(candidates, key=lambda operation: operation.arrival)
        queue = self._models[first.model_id]
        claimed = [first]
        if first.seq_id is not None and first.request_type in (
            types.RequestType.FORWARD,
            types.RequestType.FORWARD_BACKWARD,
        ):
            seq_id = first.seq_id + 1
            while (operation := queue.by_seq_id.get(seq_id)) is not None:
                if operation.state is not OperationState.QUEUED or operation.request_type != first.request_type:
                    break
                claimed.append(operation)
                seq_id += 1
        elif first.seq_id is None and first.request_type in (
            types.RequestType.FORWARD,
            types.RequestType.FORWARD_BACKWARD,
        ):
            for operation in list(queue.unsequenced)[1:]:
                if operation.state is not OperationState.QUEUED or operation.request_type != first.request_type:
                    break
                claimed.append(operation)

        for operation in claimed:
            operation.state = OperationState.CLAIMED
        return claimed

    def acknowledge_claim(self, request_ids: list[int]) -> None:
        """Release API-side request bodies after the engine owns a complete copy."""
        for request_id in request_ids:
            operation = self._by_request_id[request_id]
            if operation.state is not OperationState.CLAIMED:
                raise ValueError(f"operation {request_id} is not claimed")
            self._payload_bytes -= len(operation.payload)
            operation.payload = b""

    def complete(self, request_id: int, status: RequestStatus, result_data: str) -> None:
        operation = self._by_request_id[request_id]
        if operation.terminal:
            return
        if operation.state is not OperationState.CLAIMED:
            raise ValueError(f"operation {request_id} completed without a claim")
        operation.state = OperationState.COMPLETED if status == RequestStatus.COMPLETED else OperationState.FAILED
        self._models[operation.model_id].pending_count -= 1
        self._pending_count -= 1
        self.future_store.complete_future(request_id, status, result_data)
        self._advance(self._models[operation.model_id])
        self._terminal_order.append(request_id)
        self._trim_terminal_records()

    def _next_open(self, queue: _ModelQueue) -> Operation | None:
        if queue.next_seq_id is not None:
            operation = queue.by_seq_id.get(queue.next_seq_id)
            if operation is not None:
                return operation if operation.state is OperationState.QUEUED else None
            if any(not operation.terminal for operation in queue.by_seq_id.values()):
                return None
            if queue.unsequenced and queue.unsequenced[0].state is OperationState.QUEUED:
                return queue.unsequenced[0]
            return None

        sequenced = [operation for operation in queue.by_seq_id.values() if not operation.terminal]
        if sequenced:
            operation = min(sequenced, key=lambda item: item.seq_id)
            if operation.state is not OperationState.QUEUED:
                return None
            queue.next_seq_id = operation.seq_id
            return operation
        if queue.unsequenced and queue.unsequenced[0].state is OperationState.QUEUED:
            return queue.unsequenced[0]
        return None

    def _advance(self, queue: _ModelQueue) -> None:
        if queue.next_seq_id is not None:
            while (operation := queue.by_seq_id.get(queue.next_seq_id)) is not None and operation.terminal:
                queue.next_seq_id += 1
        while queue.unsequenced and queue.unsequenced[0].terminal:
            queue.unsequenced.popleft()

    def _trim_terminal_records(self) -> None:
        while len(self._terminal_order) > self.max_terminal_records:
            request_id = self._terminal_order.popleft()
            operation = self._by_request_id.pop(request_id, None)
            if operation is None:
                continue
            if operation.seq_id is not None:
                queue = self._models[operation.model_id]
                if queue.by_seq_id.get(operation.seq_id) is operation:
                    queue.by_seq_id.pop(operation.seq_id)
                    queue.retired_through_seq_id = max(queue.retired_through_seq_id, operation.seq_id)


async def _send_async(writer: asyncio.StreamWriter, header: dict[str, Any], payload: bytes = b"") -> None:
    encoded_header = json.dumps({**header, "payload_size": len(payload)}, separators=(",", ":")).encode()
    writer.write(_FRAME_LENGTH.pack(len(encoded_header)))
    writer.write(encoded_header)
    await writer.drain()
    view = memoryview(payload)
    for start in range(0, len(view), _STREAM_CHUNK_BYTES):
        writer.write(view[start : start + _STREAM_CHUNK_BYTES])
        await writer.drain()


async def _receive_async(reader: asyncio.StreamReader) -> tuple[dict[str, Any], bytes]:
    (header_size,) = _FRAME_LENGTH.unpack(await reader.readexactly(_FRAME_LENGTH.size))
    header = json.loads(await reader.readexactly(header_size))
    payload = await reader.readexactly(header.pop("payload_size"))
    return header, payload


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray(size)
    view = memoryview(chunks)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            raise EOFError("operation transport closed")
        received += count
    return bytes(chunks)


def _send_sync(sock: socket.socket, header: dict[str, Any], payload: bytes = b"") -> None:
    encoded_header = json.dumps({**header, "payload_size": len(payload)}, separators=(",", ":")).encode()
    sock.sendall(_FRAME_LENGTH.pack(len(encoded_header)))
    sock.sendall(encoded_header)
    if payload:
        sock.sendall(payload)


def _receive_sync(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    (header_size,) = _FRAME_LENGTH.unpack(_recv_exact(sock, _FRAME_LENGTH.size))
    header = json.loads(_recv_exact(sock, header_size))
    payload = _recv_exact(sock, header.pop("payload_size"))
    return header, payload


class OperationTransportServer:
    """Expose the API-owned queue to exactly one local engine process."""

    def __init__(self, queue: TrainingOperationQueue, socket_path: str) -> None:
        self.queue = queue
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None
        self._connected = False

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=self.socket_path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._connected:
            writer.close()
            await writer.wait_closed()
            return
        self._connected = True
        try:
            while True:
                header, payload = await _receive_async(reader)
                kind = header["kind"]
                if kind == "claim":
                    operations = self.queue.claim_ready()
                    await _send_async(writer, {"kind": "batch", "count": len(operations)})
                    for operation in operations:
                        await _send_async(
                            writer,
                            {
                                "kind": "operation",
                                "request_id": operation.request_id,
                                "request_type": operation.request_type.value,
                                "model_id": operation.model_id,
                                "seq_id": operation.seq_id,
                                "payload_format": operation.payload_format.value,
                            },
                            operation.payload,
                        )
                        ack, _ = await _receive_async(reader)
                        if ack.get("kind") != "claim_ack" or ack.get("request_id") != operation.request_id:
                            raise ValueError(f"invalid claim acknowledgement: {ack}")
                        self.queue.acknowledge_claim([operation.request_id])
                        await _send_async(writer, {"kind": "ack"})
                elif kind == "complete":
                    self.queue.complete(
                        int(header["request_id"]),
                        RequestStatus(header["status"]),
                        payload.decode(),
                    )
                    await _send_async(writer, {"kind": "ack"})
                else:
                    raise ValueError(f"unknown operation transport message: {kind}")
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self._connected = False
            writer.close()
            await writer.wait_closed()


@dataclass
class ClaimedOperation:
    request_id: int
    request_type: types.RequestType
    model_id: str
    seq_id: int | None
    payload: bytes
    payload_format: OperationPayloadFormat = OperationPayloadFormat.JSON


class OperationTransportClient:
    """Synchronous engine-side client for the API queue."""

    def __init__(self, socket_path: str) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(socket_path)

    def claim(self) -> list[ClaimedOperation]:
        _send_sync(self.socket, {"kind": "claim"})
        header, _ = _receive_sync(self.socket)
        if header["kind"] != "batch":
            raise ValueError(f"expected batch, received {header['kind']}")
        operations = []
        for _ in range(header["count"]):
            item, payload = _receive_sync(self.socket)
            operation = ClaimedOperation(
                request_id=int(item["request_id"]),
                request_type=types.RequestType(item["request_type"]),
                model_id=item["model_id"],
                seq_id=item["seq_id"],
                payload=payload,
                payload_format=OperationPayloadFormat(item["payload_format"]),
            )
            operations.append(operation)
            _send_sync(
                self.socket,
                {"kind": "claim_ack", "request_id": operation.request_id},
            )
            ack, _ = _receive_sync(self.socket)
            if ack["kind"] != "ack":
                raise ValueError(f"expected claim ack, received {ack['kind']}")
        return operations

    def complete(self, request_id: int, result: Any) -> None:
        status = RequestStatus.FAILED if isinstance(result, types.ErrorResponse) else RequestStatus.COMPLETED
        _send_sync(
            self.socket,
            {"kind": "complete", "request_id": request_id, "status": status.value},
            result.model_dump_json().encode(),
        )
        ack, _ = _receive_sync(self.socket)
        if ack["kind"] != "ack":
            raise ValueError(f"expected completion ack, received {ack['kind']}")

    def close(self) -> None:
        self.socket.close()
