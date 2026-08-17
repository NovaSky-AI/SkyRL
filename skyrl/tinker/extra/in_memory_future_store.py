import asyncio
import itertools
import time
from dataclasses import dataclass, field

from skyrl.tinker.db_models import RequestStatus


@dataclass
class _StoredFuture:
    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: RequestStatus = RequestStatus.PENDING
    result_data: str | None = None
    completed_at: float | None = None


class InMemoryFutureStore:
    """Store API-process-owned futures without routing them through the engine database."""

    def __init__(self, *, terminal_retention_sec: float = 600, max_terminal_futures: int = 10_000):
        self._terminal_retention_sec = terminal_retention_sec
        self._max_terminal_futures = max_terminal_futures
        self._next_id = itertools.count(start=-1, step=-1)
        self._futures: dict[int, _StoredFuture] = {}

    def create_future(self) -> int:
        self._cleanup_terminal_futures()
        request_id = next(self._next_id)
        self._futures[request_id] = _StoredFuture()
        return request_id

    def complete_future(self, request_id: int, status: RequestStatus, result_data: str) -> None:
        future = self._futures[request_id]
        future.status = status
        future.result_data = result_data
        future.completed_at = time.monotonic()
        future.event.set()

    async def wait_for_future(self, request_id: int, timeout: float) -> tuple[RequestStatus, str | None] | None:
        future = self._futures.get(request_id)
        if future is None:
            raise KeyError(request_id)
        if future.status == RequestStatus.PENDING:
            try:
                await asyncio.wait_for(future.event.wait(), timeout)
            except asyncio.TimeoutError:
                return None
        return future.status, future.result_data

    def _cleanup_terminal_futures(self) -> None:
        now = time.monotonic()
        expired = [
            request_id
            for request_id, future in self._futures.items()
            if future.completed_at is not None and now - future.completed_at > self._terminal_retention_sec
        ]
        for request_id in expired:
            del self._futures[request_id]

        terminal = sorted(
            (future.completed_at, request_id)
            for request_id, future in self._futures.items()
            if future.completed_at is not None
        )
        overflow = len(terminal) - self._max_terminal_futures
        for _, request_id in terminal[: max(overflow, 0)]:
            del self._futures[request_id]
