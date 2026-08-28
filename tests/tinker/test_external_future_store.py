import asyncio

import pytest

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.external_future_store import (
    ExternalFutureStore,
    SequenceConflictError,
)


@pytest.mark.asyncio
async def test_external_future_store_completes_without_database_polling() -> None:
    store = ExternalFutureStore()
    request = {"prompt": [1, 2, 3]}
    request_id = store.create(types.RequestType.EXTERNAL, "model", request)

    waiter = asyncio.create_task(store.wait(request_id, timeout=1))
    assert store.complete(request_id, RequestStatus.COMPLETED, '{"sequences": []}')

    assert await waiter == (
        RequestStatus.COMPLETED,
        types.RequestType.EXTERNAL,
        '{"sequences": []}',
    )


@pytest.mark.asyncio
async def test_external_future_store_keeps_identical_sequence_retries_idempotent() -> None:
    store = ExternalFutureStore()
    request = {"prompt": [1, 2, 3]}

    first = store.create(types.RequestType.EXTERNAL, "model", request, seq_id=7)
    retry = store.create(types.RequestType.EXTERNAL, "model", request, seq_id=7)

    assert retry == first
    with pytest.raises(SequenceConflictError, match="sequence number was reused"):
        store.create(types.RequestType.EXTERNAL, "model", {"prompt": [4]}, seq_id=7)


@pytest.mark.asyncio
async def test_external_future_store_reports_missing_and_timed_out_futures() -> None:
    store = ExternalFutureStore()

    with pytest.raises(KeyError):
        await store.wait(123, timeout=0)

    request_id = store.create(types.RequestType.EXTERNAL, "model", {})
    with pytest.raises(asyncio.TimeoutError):
        await store.wait(request_id, timeout=0)


def test_external_future_store_drops_oldest_completed_entry_at_capacity() -> None:
    store = ExternalFutureStore(max_entries=1)
    first = store.create(types.RequestType.EXTERNAL, "model", {})
    assert store.complete(first, RequestStatus.COMPLETED, "{}")

    second = store.create(types.RequestType.EXTERNAL, "model", {})

    assert first not in store
    assert second in store


def test_external_future_store_expires_completed_sequence() -> None:
    store = ExternalFutureStore(terminal_retention_seconds=0)
    first = store.create(types.RequestType.EXTERNAL, "model", {}, seq_id=1)
    assert store.complete(first, RequestStatus.COMPLETED, "{}")

    retry = store.create(types.RequestType.EXTERNAL, "model", {}, seq_id=1)

    assert retry != first
