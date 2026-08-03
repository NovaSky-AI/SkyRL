"""Shared waiting for asynchronous request results."""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.utils.log import logger

# How often the shared poller looks for newly finished requests. A single query
# covers every waiter, so this can stay tight without the load scaling up with
# the number of in-flight requests.
POLL_INTERVAL_SEC = 0.05

# Chunk size for `IN (...)` lookups, kept under SQLite's bound-parameter cap.
_MAX_IDS_PER_QUERY = 500


@dataclass(frozen=True)
class FutureResult:
    """Terminal state of a request."""

    status: RequestStatus
    result_data: dict | None


async def complete_future(
    db_engine,
    request_id: int,
    status: RequestStatus,
    result_data: dict | None,
    waiter: "FutureWaiter | None" = None,
) -> bool:
    """Write a terminal result for one request, returning False if it no longer exists.

    Updates by primary key rather than loading the row through the ORM first:
    the row carries the request's full payload, which a completion write has no
    use for. When a ``waiter`` is given its waiters are resolved directly, so a
    result produced in this process does not wait to be rediscovered by polling.
    """
    async with AsyncSession(db_engine) as session:
        result = await session.exec(
            update(FutureDB)
            .where(FutureDB.request_id == request_id)
            .values(result_data=result_data, status=status, completed_at=datetime.now(timezone.utc))
        )
        await session.commit()

    if not result.rowcount:
        return False

    if waiter is not None:
        waiter.notify(request_id, status, result_data)
    return True


class FutureWaiter:
    """Resolves many in-flight requests using one periodic batched query.

    Callers await :meth:`wait`, which registers an asyncio future and returns
    once a background task sees the request reach a terminal status.

    Polling per caller instead would cost one query per in-flight request per
    tick. Under a multi-LoRA RL workload -- hundreds of concurrent rollouts, each
    long-polling its own future -- that is the dominant read load on the
    database, so every waiter shares a single query here.
    """

    def __init__(self, db_engine, poll_interval_sec: float = POLL_INTERVAL_SEC):
        self._db_engine = db_engine
        self._poll_interval_sec = poll_interval_sec
        self._waiters: dict[int, set[asyncio.Future]] = {}
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the background poller."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background poller."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def wait(self, request_id: int, timeout: float) -> FutureResult | None:
        """Wait for ``request_id`` to reach a terminal status.

        Returns None if ``timeout`` elapses first. Raises KeyError if no such
        request exists.
        """
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(request_id, set()).add(waiter)
        self._wakeup.set()
        try:
            return await asyncio.wait_for(waiter, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            remaining = self._waiters.get(request_id)
            if remaining is not None:
                remaining.discard(waiter)
                if not remaining:
                    del self._waiters[request_id]

    def notify(self, request_id: int, status: RequestStatus, result_data: dict | None) -> None:
        """Resolve waiters for a result this process just wrote.

        Lets the sample-forwarding path hand off directly instead of waiting for
        the poller to rediscover its own write.
        """
        self._resolve(request_id, FutureResult(status=status, result_data=result_data))

    def _resolve(self, request_id: int, outcome: "FutureResult | BaseException") -> None:
        for waiter in list(self._waiters.get(request_id, ())):
            if waiter.done():
                continue
            if isinstance(outcome, BaseException):
                waiter.set_exception(outcome)
            else:
                waiter.set_result(outcome)

    async def _run(self) -> None:
        while True:
            if not self._waiters:
                # Safe against lost wakeups: registration adds to _waiters and
                # sets the event with no await in between, so an empty dict here
                # means nothing has been registered yet.
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep the poller alive; waiters fall back on their own timeouts.
                logger.exception("Future poller iteration failed")
            await asyncio.sleep(self._poll_interval_sec)

    async def _poll_once(self) -> None:
        request_ids = list(self._waiters)
        async with AsyncSession(self._db_engine) as session:
            for start in range(0, len(request_ids), _MAX_IDS_PER_QUERY):
                chunk = request_ids[start : start + _MAX_IDS_PER_QUERY]
                statement = select(FutureDB.request_id, FutureDB.status, FutureDB.result_data).where(
                    FutureDB.request_id.in_(chunk)
                )
                rows = (await session.exec(statement)).all()

                found = set()
                for request_id, status, result_data in rows:
                    found.add(request_id)
                    if status in (RequestStatus.COMPLETED, RequestStatus.FAILED):
                        self._resolve(request_id, FutureResult(status=status, result_data=result_data))

                # Rows are never deleted, so an id with no row never existed.
                for request_id in set(chunk) - found:
                    self._resolve(request_id, KeyError(request_id))
