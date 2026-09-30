"""Driver-side deadlines that turn a hung training step into a typed, picklable error.

``step_deadline`` scopes a time budget to the current context and cancels the awaited code when it runs out.
``ray_get`` bounds its wait by the time left, and ``stage`` / ``operation`` record where the step was when the budget
ran out. With no active deadline every helper
is a no-op, so code outside a deadline behaves as before. Only the driver's wait is interrupted: a hung worker
keeps running until teardown.
"""

import asyncio
import contextlib
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Optional, Type

import ray


class StepTimeoutError(RuntimeError):
    """A training step exceeded its ``trainer.step_timeout_s`` budget.

    Deliberately not a ``TimeoutError``, so ``except TimeoutError`` retry handlers don't swallow it.
    """

    def __init__(self, global_step: int, stage: str, operation: str, budget_s: float, elapsed_s: float) -> None:
        # All fields go to args so the default BaseException pickling round-trips them.
        super().__init__(global_step, stage, operation, budget_s, elapsed_s)
        self.global_step = global_step
        self.stage = stage
        self.operation = operation
        self.budget_s = budget_s
        self.elapsed_s = elapsed_s

    def __str__(self) -> str:
        return (
            f"step {self.global_step} exceeded its {self.budget_s:g}s budget after {self.elapsed_s:.1f}s "
            f"(stage={self.stage!r}, operation={self.operation!r})"
        )


class WeightSyncTimeoutError(StepTimeoutError):
    """Weight sync exceeded its ``trainer.weight_sync_timeout_s`` budget."""


@dataclass
class _Deadline:
    global_step: int
    budget_s: float
    started_at: float
    expires_at: float
    error_cls: Type[StepTimeoutError]
    # Innermost stage / operation unwound by the expiry cancellation.
    failed_stage: Optional[str] = None
    failed_operation: Optional[str] = None

    def error(self, stage: str, operation: str) -> StepTimeoutError:
        return self.error_cls(self.global_step, stage, operation, self.budget_s, time.monotonic() - self.started_at)

    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at


_ACTIVE: ContextVar[Optional[_Deadline]] = ContextVar("skyrl_step_deadline", default=None)
_STAGE: ContextVar[str] = ContextVar("skyrl_step_stage", default="unknown")
_OPERATION: ContextVar[str] = ContextVar("skyrl_step_operation", default="unknown")


@contextlib.asynccontextmanager
async def step_deadline(
    global_step: int, budget_s: Optional[float], error_cls: Type[StepTimeoutError] = StepTimeoutError
) -> AsyncIterator[None]:
    """Bound the enclosed block by ``budget_s`` seconds. No-op when ``budget_s`` is None.

    On expiry the awaited code is cancelled and ``error_cls`` is raised. A ``TimeoutError`` raised by the block
    itself passes through unchanged. A nested deadline can only shorten its parent: if the parent expires first,
    the parent stays in effect.
    """
    if budget_s is None:
        yield
        return
    now = time.monotonic()
    parent = _ACTIVE.get()
    if parent is not None and parent.expires_at <= now + budget_s:
        yield
        return
    d = _Deadline(global_step, budget_s, now, now + budget_s, error_cls)
    token = _ACTIVE.set(d)
    try:
        async with asyncio.timeout(budget_s) as cm:
            yield
    except TimeoutError:
        if not cm.expired():
            raise
        raise d.error(d.failed_stage or _STAGE.get(), d.failed_operation or "await") from None
    finally:
        _ACTIVE.reset(token)


def remaining() -> Optional[float]:
    """Seconds left on the active deadline, or None when there is none."""
    d = _ACTIVE.get()
    if d is None:
        return None
    return max(0.0, d.expires_at - time.monotonic())


def _expired_deadline() -> Optional[_Deadline]:
    d = _ACTIVE.get()
    return d if d is not None and d.expired() else None


@contextlib.contextmanager
def stage(name: str) -> Iterator[None]:
    """Name the current stage of the step (entered by every ``Timer``)."""
    token = _STAGE.set(name)
    try:
        yield
    except BaseException:
        d = _expired_deadline()
        if d is not None and d.failed_stage is None:
            d.failed_stage = name
        raise
    finally:
        _STAGE.reset(token)


@contextlib.contextmanager
def operation(name: str) -> Iterator[None]:
    """Name the operation currently awaited within a stage."""
    token = _OPERATION.set(name)
    try:
        yield
    except BaseException:
        d = _expired_deadline()
        if d is not None and d.failed_operation is None:
            d.failed_operation = name
        raise
    finally:
        _OPERATION.reset(token)


def ray_get(refs: Any, operation: str) -> Any:
    """``ray.get`` bounded by the active deadline; identical to ``ray.get(refs)`` when there is none."""
    d = _ACTIVE.get()
    if d is None:
        return ray.get(refs)
    try:
        return ray.get(refs, timeout=remaining())
    except ray.exceptions.GetTimeoutError:
        raise d.error(_STAGE.get(), operation) from None
