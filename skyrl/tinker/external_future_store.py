import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus


class SequenceConflictError(ValueError):
    """Raised when a sequence number is reused for a different request."""


@dataclass
class _ExternalFuture:
    request_type: types.RequestType
    request_data: Any
    status: RequestStatus
    event: asyncio.Event
    result_data: str | None = None
    completed_at: float | None = None
    sequence_key: tuple[str, int] | None = None


class ExternalFutureStore:
    """In-process futures for samples forwarded directly to inference."""

    def __init__(self, *, terminal_retention_seconds: float = 600, max_entries: int = 10_000):
        self._next_id = itertools.count(start=2**31)
        self._futures: dict[int, _ExternalFuture] = {}
        self._sequences: dict[tuple[str, int], int] = {}
        self._terminal: deque[int] = deque()
        self._terminal_retention_seconds = terminal_retention_seconds
        self._max_entries = max_entries

    def create(
        self,
        request_type: types.RequestType,
        model_id: str,
        request_data: Any,
        *,
        seq_id: int | None = None,
    ) -> int:
        """Create a future, returning the original ID for an identical retry."""
        self._cleanup()
        sequence_key = (model_id, seq_id) if seq_id is not None else None
        if sequence_key is not None and (request_id := self._sequences.get(sequence_key)) is not None:
            future = self._futures[request_id]
            if future.request_type != request_type or future.request_data != request_data:
                raise SequenceConflictError("Training request sequence number was reused")
            return request_id

        request_id = next(self._next_id)
        self._futures[request_id] = _ExternalFuture(
            request_type=request_type,
            request_data=request_data,
            status=RequestStatus.PENDING,
            event=asyncio.Event(),
            sequence_key=sequence_key,
        )
        if sequence_key is not None:
            self._sequences[sequence_key] = request_id
        self._cleanup()
        return request_id

    def complete(self, request_id: int, status: RequestStatus, result_data: str) -> bool:
        """Complete a future, returning false if it has already expired."""
        future = self._futures.get(request_id)
        if future is None:
            return False
        future.status = status
        future.result_data = result_data
        if future.completed_at is None:
            future.completed_at = time.monotonic()
            self._terminal.append(request_id)
        future.event.set()
        return True

    async def wait(self, request_id: int, timeout: float) -> tuple[RequestStatus, types.RequestType, str | None]:
        """Wait for a future, raising KeyError or TimeoutError when unavailable."""
        future = self._futures[request_id]
        if future.status == RequestStatus.PENDING:
            await asyncio.wait_for(future.event.wait(), timeout)
        return future.status, future.request_type, future.result_data

    def __contains__(self, request_id: int) -> bool:
        return request_id in self._futures

    def _cleanup(self) -> None:
        now = time.monotonic()
        while self._terminal:
            request_id = self._terminal[0]
            future = self._futures[request_id]
            assert future.completed_at is not None
            expired = now - future.completed_at > self._terminal_retention_seconds
            if not expired and len(self._futures) <= self._max_entries:
                break
            self._terminal.popleft()
            self._drop(request_id)

    def _drop(self, request_id: int) -> None:
        future = self._futures.pop(request_id)
        if future.sequence_key is not None:
            del self._sequences[future.sequence_key]
