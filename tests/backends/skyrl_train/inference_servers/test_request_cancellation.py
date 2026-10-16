import asyncio

import pytest
from starlette.requests import Request

from skyrl.backends.skyrl_train.inference_servers.request_cancellation import (
    RequestDisconnectedError,
    run_until_disconnected,
)


def _request() -> tuple[Request, asyncio.Queue[dict]]:
    messages = asyncio.Queue()
    messages.put_nowait({"type": "http.request", "body": b"{}", "more_body": False})
    return Request({"type": "http", "headers": []}, messages.get), messages


@pytest.mark.asyncio
async def test_operation_result_wins_while_client_remains_connected():
    request, _ = _request()

    assert await run_until_disconnected(request, lambda: asyncio.sleep(0, result=42)) == 42


@pytest.mark.asyncio
async def test_disconnect_cancels_and_drains_operation():
    request, messages = _request()
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaned_up.set()

    call = asyncio.create_task(run_until_disconnected(request, operation))
    await asyncio.wait_for(started.wait(), timeout=1)
    messages.put_nowait({"type": "http.disconnect"})

    with pytest.raises(RequestDisconnectedError):
        await asyncio.wait_for(call, timeout=1)
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_handler_cancellation_waits_for_operation_cleanup():
    request, _ = _request()
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            await cleanup_release.wait()
            raise

    call = asyncio.create_task(run_until_disconnected(request, operation))
    await asyncio.wait_for(started.wait(), timeout=1)
    call.cancel()
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    call.cancel()
    await asyncio.sleep(0)
    assert not call.done()

    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=1)
