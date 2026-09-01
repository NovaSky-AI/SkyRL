import asyncio

import pytest

from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.extra.in_memory_future_store import InMemoryFutureStore


@pytest.mark.asyncio
async def test_concurrent_waiters_receive_completed_result():
    store = InMemoryFutureStore()
    request_id = store.create_future()
    waiters = [asyncio.create_task(store.wait_for_future(request_id, 1)) for _ in range(3)]

    store.complete_future(request_id, RequestStatus.COMPLETED, '{"value":1}')

    assert await asyncio.gather(*waiters) == [
        (RequestStatus.COMPLETED, '{"value":1}'),
        (RequestStatus.COMPLETED, '{"value":1}'),
        (RequestStatus.COMPLETED, '{"value":1}'),
    ]


@pytest.mark.asyncio
async def test_timed_out_waiter_does_not_cancel_future():
    store = InMemoryFutureStore()
    request_id = store.create_future()

    assert await store.wait_for_future(request_id, 0) is None
    store.complete_future(request_id, RequestStatus.FAILED, '{"error":"boom"}')

    assert await store.wait_for_future(request_id, 1) == (
        RequestStatus.FAILED,
        '{"error":"boom"}',
    )


@pytest.mark.asyncio
async def test_unknown_future_raises_key_error():
    store = InMemoryFutureStore()

    with pytest.raises(KeyError):
        await store.wait_for_future(-1, 1)


@pytest.mark.asyncio
async def test_terminal_future_retention_is_bounded():
    store = InMemoryFutureStore(max_terminal_futures=1)
    first_id = store.create_future()
    store.complete_future(first_id, RequestStatus.COMPLETED, "{}")
    second_id = store.create_future()
    store.complete_future(second_id, RequestStatus.COMPLETED, "{}")

    store.create_future()

    with pytest.raises(KeyError):
        await store.wait_for_future(first_id, 1)
    assert await store.wait_for_future(second_id, 1) == (
        RequestStatus.COMPLETED,
        "{}",
    )


@pytest.mark.asyncio
async def test_terminal_result_bytes_are_bounded():
    store = InMemoryFutureStore(max_terminal_bytes=4)
    first_id = store.create_future()
    store.complete_future(first_id, RequestStatus.COMPLETED, "1234")
    second_id = store.create_future()
    store.complete_future(second_id, RequestStatus.COMPLETED, "56")

    with pytest.raises(KeyError):
        await store.wait_for_future(first_id, 1)
    assert await store.wait_for_future(second_id, 1) == (RequestStatus.COMPLETED, "56")
