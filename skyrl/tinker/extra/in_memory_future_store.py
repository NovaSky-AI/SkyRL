import asyncio
import itertools
import time
from collections import deque
from dataclasses import dataclass, field

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus


@dataclass
class _StoredFuture:
    request_type: types.RequestType | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    status: RequestStatus = RequestStatus.PENDING
    result_data: str | None = None
    completed_at: float | None = None


class InMemoryFutureStore:
    """Store API-process-owned futures without routing them through the engine database."""

    def __init__(
        self,
        *,
        terminal_retention_sec: float = 600,
        max_terminal_futures: int = 10_000,
        max_terminal_bytes: int = 2 * 1024**3,
    ):
        self._terminal_retention_sec = terminal_retention_sec
        self._max_terminal_futures = max_terminal_futures
        self._max_terminal_bytes = max_terminal_bytes
        self._next_id = itertools.count(start=-1, step=-1)
        self._futures: dict[int, _StoredFuture] = {}
        self._terminal_futures: deque[tuple[float, int, int]] = deque()
        self._terminal_bytes = 0

    def create_future(self, request_type: types.RequestType | None = None) -> int:
        self._cleanup_terminal_futures()
        request_id = next(self._next_id)
        self._futures[request_id] = _StoredFuture(request_type=request_type)
        return request_id

    def request_type(self, request_id: int) -> types.RequestType | None:
        future = self._futures.get(request_id)
        if future is None:
            raise KeyError(request_id)
        return future.request_type

    def has_future(self, request_id: int) -> bool:
        self._cleanup_terminal_futures()
        return request_id in self._futures

    def complete_future(self, request_id: int, status: RequestStatus, result_data: str) -> None:
        future = self._futures[request_id]
        future.status = status
        future.result_data = result_data
        future.completed_at = time.monotonic()
        result_bytes = len(result_data.encode())
        self._terminal_bytes += result_bytes
        self._terminal_futures.append((future.completed_at, request_id, result_bytes))
        future.event.set()
        self._cleanup_terminal_futures()

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
        while self._terminal_futures and (
            now - self._terminal_futures[0][0] > self._terminal_retention_sec
            or len(self._terminal_futures) > self._max_terminal_futures
            or self._terminal_bytes > self._max_terminal_bytes
        ):
            _, request_id, result_bytes = self._terminal_futures.popleft()
            self._terminal_bytes -= result_bytes
            del self._futures[request_id]
