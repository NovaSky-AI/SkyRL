import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pydantic import BaseModel
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger


@dataclass
class ExternalFuture:
    request_id: int
    model_id: str | None
    request_data: dict
    status: RequestStatus = RequestStatus.PENDING
    result_data: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)
    persistence_error: Exception | None = None


class ExternalFutureStore:
    """Keeps forwarded sample futures off the database hot path."""

    _PERSIST_BATCH_SIZE = 64
    _PERSIST_QUEUE_SIZE = 2048

    def __init__(self, db_engine, db_write_lock: AbstractAsyncContextManager):
        self.db_engine = db_engine
        self.db_write_lock = db_write_lock
        self._entries: dict[int, ExternalFuture] = {}
        self._persist_queue: asyncio.Queue[ExternalFuture] = asyncio.Queue(maxsize=self._PERSIST_QUEUE_SIZE)
        self._persist_worker: asyncio.Task | None = None
        self._persist_error: Exception | None = None
        self._next_request_id = -1

    async def start(self) -> None:
        async with AsyncSession(self.db_engine) as session:
            statement = select(func.min(FutureDB.request_id)).where(FutureDB.request_id < 0)
            minimum_request_id = (await session.exec(statement)).one()
        if minimum_request_id is not None:
            self._next_request_id = minimum_request_id - 1
        self._persist_worker = asyncio.create_task(self._persist_loop())

    def create(self, model_id: str | None, request_data: BaseModel) -> int:
        request_id = self._next_request_id
        self._next_request_id -= 1
        self._entries[request_id] = ExternalFuture(
            request_id=request_id,
            model_id=model_id,
            request_data=request_data.model_dump(mode="json"),
        )
        return request_id

    async def wait(self, request_id: int, timeout: float) -> tuple[RequestStatus, types.RequestType, str | None] | None:
        entry = self._entries.get(request_id)
        if entry is None:
            raise KeyError(request_id)
        try:
            await asyncio.wait_for(entry.event.wait(), timeout)
        except asyncio.TimeoutError:
            return None
        if entry.persistence_error is not None:
            self._entries.pop(request_id, None)
            raise RuntimeError(f"Failed to persist external future {request_id}") from entry.persistence_error
        return entry.status, types.RequestType.EXTERNAL, entry.result_data

    async def complete(self, request_id: int, result_data: BaseModel, status: RequestStatus) -> None:
        entry = self._entries[request_id]
        entry.result_data = result_data.model_dump_json()
        entry.status = status
        entry.completed_at = datetime.now(timezone.utc)
        await self._persist_queue.put(entry)

    async def flush(self) -> None:
        await self._persist_queue.join()
        if self._persist_error is not None:
            error, self._persist_error = self._persist_error, None
            raise RuntimeError("External future persistence failed") from error

    async def close(self) -> None:
        try:
            await self.flush()
        finally:
            if self._persist_worker is not None:
                self._persist_worker.cancel()
                await asyncio.gather(self._persist_worker, return_exceptions=True)

    async def _persist_loop(self) -> None:
        while True:
            entries = [await self._persist_queue.get()]
            while len(entries) < self._PERSIST_BATCH_SIZE:
                try:
                    entries.append(self._persist_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await self._persist(entries)
            except Exception:
                # One bad row or transient error must not poison the whole
                # batch: retry each entry on its own and fail only the ones
                # that cannot be persisted individually.
                logger.exception(
                    "Batched external future persistence failed request_ids=%s..%s, retrying individually",
                    entries[0].request_id,
                    entries[-1].request_id,
                )
                for entry in entries:
                    try:
                        await self._persist([entry])
                    except Exception as error:
                        self._persist_error = error
                        entry.persistence_error = error
                        logger.exception("External future persistence failed request_id=%s", entry.request_id)
                    entry.event.set()
                    if entry.persistence_error is None:
                        self._entries.pop(entry.request_id, None)
            else:
                for entry in entries:
                    entry.event.set()
                    self._entries.pop(entry.request_id, None)
            finally:
                for _ in entries:
                    self._persist_queue.task_done()

    async def _persist(self, entries: list[ExternalFuture]) -> None:
        async with self.db_write_lock:
            async with AsyncSession(self.db_engine) as session:
                session.add_all(
                    [
                        FutureDB(
                            request_id=entry.request_id,
                            request_type=types.RequestType.EXTERNAL,
                            model_id=entry.model_id,
                            request_data=entry.request_data,
                            result_data=entry.result_data,
                            status=entry.status,
                            created_at=entry.created_at,
                            completed_at=entry.completed_at,
                        )
                        for entry in entries
                    ]
                )
                await session.commit()
