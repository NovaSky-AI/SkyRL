"""asyncio helpers for surfacing background-task failures as real exceptions."""

import asyncio
from typing import Any, Awaitable, Iterable, Optional, TypeVar

T = TypeVar("T")
TASK_SHUTDOWN_GRACE_S = 10.0


class BackgroundFailure:
    """Holds the first exception raised by a set of background tasks and wakes consumers waiting on them.

    Not an exception itself, so it never crosses a Ray boundary: the recorded exception is re-raised as-is.
    """

    def __init__(self) -> None:
        self._exc: Optional[BaseException] = None
        self._event = asyncio.Event()

    @property
    def failed(self) -> bool:
        return self._exc is not None

    def record(self, exc: BaseException, source: str) -> None:
        """Record ``exc`` if it is the first failure, and wake every ``guard`` waiter."""
        if self._exc is not None:
            return
        exc.add_note(f"raised in background {source}")
        self._exc = exc
        self._event.set()

    def raise_if_failed(self) -> None:
        if self._exc is not None:
            raise self._exc

    async def guard(self, aw: Awaitable[T]) -> T:
        """Await ``aw``, raising the recorded failure instead if one is already recorded or arrives first.

        A result that completes together with the failure is returned, so e.g. a popped queue item is not lost.
        """
        if self._exc is not None:
            if asyncio.iscoroutine(aw):
                aw.close()
            raise self._exc
        task = asyncio.ensure_future(aw)
        failure_wait = asyncio.ensure_future(self._event.wait())
        try:
            await asyncio.wait({task, failure_wait}, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                return task.result()
            self.raise_if_failed()
            raise AssertionError("failure event set without a recorded exception")
        finally:
            task.cancel()
            failure_wait.cancel()


async def cancel_background_tasks(
    tasks: Iterable[asyncio.Task[Any]], grace_s: float = TASK_SHUTDOWN_GRACE_S
) -> None:
    """Cancel background tasks without waiting indefinitely for cleanup."""
    tasks = set(tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        done, _ = await asyncio.wait(tasks, timeout=grace_s)
        for task in done:
            if not task.cancelled():
                task.exception()
