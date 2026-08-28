import asyncio
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic

from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlmodel import func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.db_models import ExternalFutureIdSequenceDB, FutureDB, RequestStatus
from skyrl.tinker.db_observability import SLOW_DATABASE_SECONDS, database_pool_status
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
    _PERSIST_MAX_ATTEMPTS = 3
    _PERSIST_RETRY_INITIAL_DELAY_SEC = 0.05
    # Reserving a large range keeps ID allocation off the database hot path.
    # Long-lived processes reserve another durable range before they exhaust it.
    _REQUEST_ID_RESERVATION_SIZE = 1_000_000

    def __init__(self, db_engine, db_write_lock: AbstractAsyncContextManager):
        self.db_engine = db_engine
        self.db_write_lock = db_write_lock
        self._entries: dict[int, ExternalFuture] = {}
        self._persist_queue: asyncio.Queue[ExternalFuture] = asyncio.Queue(maxsize=self._PERSIST_QUEUE_SIZE)
        self._persist_worker: asyncio.Task | None = None
        self._persist_error: Exception | None = None
        self._next_request_id = -1
        self._request_id_reservation_floor = -1
        self._request_id_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._reserve_request_ids()
        self._persist_worker = asyncio.create_task(self._persist_loop())

    async def create(self, model_id: str | None, request_data: BaseModel) -> int:
        if self._persist_error is not None:
            raise RuntimeError("External future persistence is unavailable") from self._persist_error
        if self._next_request_id <= self._request_id_reservation_floor:
            async with self._request_id_lock:
                if self._next_request_id <= self._request_id_reservation_floor:
                    await self._reserve_request_ids()
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

    async def complete(
        self,
        request_id: int,
        result_data: BaseModel,
        status: RequestStatus,
        *,
        cancellation_safe: bool = True,
    ) -> None:
        entry = self._entries[request_id]
        entry.result_data = result_data.model_dump_json()
        entry.status = status
        entry.completed_at = datetime.now(timezone.utc)
        if not cancellation_safe:
            await self._persist_queue.put(entry)
            return
        queue_put = asyncio.create_task(self._persist_queue.put(entry))
        try:
            await asyncio.shield(queue_put)
        except asyncio.CancelledError:
            await queue_put
            raise

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
            persistence_started = monotonic()
            try:
                await self._persist_with_retry(entries)
            except Exception as error:
                persistence_elapsed = monotonic() - persistence_started
                self._persist_error = error
                logger.error(
                    "External future persistence failed failure_stage=persistence batch_size=%s "
                    "queue_depth=%s persistence_seconds=%.3f pool=%s error_type=%s",
                    len(entries),
                    self._persist_queue.qsize(),
                    persistence_elapsed,
                    database_pool_status(self.db_engine),
                    type(error).__name__,
                )
                for entry in entries:
                    entry.persistence_error = error
                    entry.event.set()
            else:
                persistence_elapsed = monotonic() - persistence_started
                if persistence_elapsed >= SLOW_DATABASE_SECONDS:
                    logger.warning(
                        "Slow external future persistence failure_stage=persistence batch_size=%s "
                        "queue_depth=%s persistence_seconds=%.3f pool=%s",
                        len(entries),
                        self._persist_queue.qsize(),
                        persistence_elapsed,
                        database_pool_status(self.db_engine),
                    )
                for entry in entries:
                    entry.event.set()
                    self._entries.pop(entry.request_id, None)
            finally:
                for _ in entries:
                    self._persist_queue.task_done()

    async def _reserve_request_ids(self) -> None:
        async with self.db_write_lock:
            async with AsyncSession(self.db_engine) as session:
                statement = select(func.min(FutureDB.request_id)).where(FutureDB.request_id < 0)
                minimum_request_id = (await session.exec(statement)).one()
                initial_request_id = minimum_request_id - 1 if minimum_request_id is not None else -1

                values = {"singleton_id": 1, "next_request_id": initial_request_id}
                if self.db_engine.dialect.name == "sqlite":
                    insert_statement = sqlite_insert(ExternalFutureIdSequenceDB).values(**values)
                else:
                    insert_statement = postgresql_insert(ExternalFutureIdSequenceDB).values(**values)
                await session.exec(insert_statement.on_conflict_do_nothing(index_elements=["singleton_id"]))

                reservation = await session.exec(
                    update(ExternalFutureIdSequenceDB)
                    .where(ExternalFutureIdSequenceDB.singleton_id == 1)
                    .values(
                        next_request_id=(ExternalFutureIdSequenceDB.next_request_id - self._REQUEST_ID_RESERVATION_SIZE)
                    )
                    .returning(ExternalFutureIdSequenceDB.next_request_id)
                )
                reservation_floor = reservation.scalar_one()
                await session.commit()
                self._request_id_reservation_floor = reservation_floor
                self._next_request_id = reservation_floor + self._REQUEST_ID_RESERVATION_SIZE

    async def _persist_with_retry(self, entries: list[ExternalFuture]) -> None:
        for attempt in range(1, self._PERSIST_MAX_ATTEMPTS + 1):
            try:
                await self._persist(entries)
                return
            except Exception as error:
                if attempt == self._PERSIST_MAX_ATTEMPTS or not _is_transient_database_error(error):
                    raise
                delay = self._PERSIST_RETRY_INITIAL_DELAY_SEC * 2 ** (attempt - 1)
                logger.warning(
                    "Transient external future persistence failure; retrying "
                    "failure_stage=persistence batch_size=%s attempt=%s max_attempts=%s "
                    "retry_delay_seconds=%.3f pool=%s error_type=%s",
                    len(entries),
                    attempt,
                    self._PERSIST_MAX_ATTEMPTS,
                    delay,
                    database_pool_status(self.db_engine),
                    type(error).__name__,
                )
                await asyncio.sleep(delay)

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


def _is_transient_database_error(error: Exception) -> bool:
    return (
        isinstance(error, (OperationalError, SQLAlchemyTimeoutError))
        or isinstance(error, DBAPIError)
        and error.connection_invalidated
    )
