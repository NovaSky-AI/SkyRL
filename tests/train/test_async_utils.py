"""CPU unit tests for skyrl.train.utils.async_utils."""

import asyncio

import pytest

from skyrl.train.utils.async_utils import (
    BackgroundFailure,
    cancel_background_tasks,
    cleanup_preserving_primary,
)

# --------------------------------------------------------------------------------------
# BackgroundFailure
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_guard_raises_failure_recorded_while_blocked():
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()
    err = ValueError("worker died")

    async def fail_later():
        await asyncio.sleep(0.01)
        failure.record(err, "worker")

    task = asyncio.create_task(fail_later())
    with pytest.raises(ValueError) as exc_info:
        await asyncio.wait_for(failure.guard(queue.get()), timeout=5)
    await task
    assert exc_info.value is err
    queue.put_nowait("still available")
    assert await asyncio.wait_for(queue.get(), timeout=1) == "still available"


@pytest.mark.asyncio
async def test_record_keeps_first_exception_and_buffered_item():
    failure = BackgroundFailure()
    first, second = RuntimeError("first"), RuntimeError("second")
    failure.record(first, "generation worker")
    failure.record(second, "generation worker")
    assert failure.failed
    with pytest.raises(RuntimeError) as exc_info:
        failure.raise_if_failed()
    assert exc_info.value is first
    assert first.__notes__ == ["raised in background generation worker"]
    assert not hasattr(second, "__notes__")
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("item")
    with pytest.raises(RuntimeError) as exc_info:
        await failure.guard(queue.get())
    assert exc_info.value is first
    assert queue.get_nowait() == "item"


@pytest.mark.asyncio
async def test_guard_prefers_result_over_simultaneous_failure():
    """An item popped in the same iteration the failure lands must be returned, not dropped."""
    failure = BackgroundFailure()
    queue: asyncio.Queue = asyncio.Queue()

    def put_and_fail():
        queue.put_nowait("item")
        failure.record(RuntimeError("boom"), "worker")

    asyncio.get_running_loop().call_soon(put_and_fail)
    assert await failure.guard(queue.get()) == "item"
    assert queue.empty()


@pytest.mark.asyncio
async def test_cancel_background_tasks_bounds_an_unresponsive_task():
    started = asyncio.Event()
    release = asyncio.Event()

    async def ignore_cancel_once():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(ignore_cancel_once())
    await started.wait()
    await cancel_background_tasks([task], grace_s=0.01)
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, timeout=1)


# --------------------------------------------------------------------------------------
# cleanup_preserving_primary
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_error_propagates_on_success():
    async def failing_cleanup():
        raise ConnectionError("engine unreachable")

    with pytest.raises(ConnectionError):
        async with cleanup_preserving_primary(failing_cleanup, "resume_generation"):
            pass


@pytest.mark.asyncio
async def test_cleanup_runs_on_cancel():
    calls = []

    async def cleanup():
        calls.append("cleanup")

    started = asyncio.Event()

    async def body():
        async with cleanup_preserving_primary(cleanup, "resume_generation"):
            started.set()
            await asyncio.sleep(3600)

    task = asyncio.create_task(body())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == ["cleanup"]
