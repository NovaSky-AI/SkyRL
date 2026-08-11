"""Tests for waiting on asynchronous request results (api.wait_for_future/poll_futures)."""

import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine

from skyrl.tinker import types
from skyrl.tinker.api import poll_futures, wait_for_future
from skyrl.tinker.db_models import (
    FutureDB,
    RequestStatus,
    enable_sqlite_wal,
    get_async_database_url,
)


@pytest.fixture()
def db_url(tmp_path):
    """A file-backed SQLite database with the schema created.

    A file rather than :memory: so the sync and async engines below see the same
    database.
    """
    url = f"sqlite:///{tmp_path / 'tinker.db'}"
    sync_engine = create_engine(url)
    enable_sqlite_wal(sync_engine)
    SQLModel.metadata.create_all(sync_engine)
    sync_engine.dispose()
    return url


@pytest.fixture()
def sync_engine(db_url):
    engine = create_engine(db_url)
    enable_sqlite_wal(engine)
    yield engine
    engine.dispose()


@pytest_asyncio.fixture()
async def async_engine(db_url):
    engine = create_async_engine(get_async_database_url(db_url))
    enable_sqlite_wal(engine.sync_engine)
    yield engine
    await engine.dispose()


def insert_pending(sync_engine, count: int = 1) -> list[int]:
    """Insert ``count`` pending futures, returning their request_ids."""
    with Session(sync_engine) as session:
        rows = [
            FutureDB(
                request_type=types.RequestType.SAMPLE,
                model_id="model_a",
                request_data={"checkpoint_id": ""},
                status=RequestStatus.PENDING,
            )
            for _ in range(count)
        ]
        for row in rows:
            session.add(row)
        session.commit()
        return [row.request_id for row in rows]


def mark_completed(sync_engine, request_id: int, result_data: dict, status=RequestStatus.COMPLETED) -> None:
    with Session(sync_engine) as session:
        row = session.get(FutureDB, request_id)
        row.result_data = result_data
        row.status = status
        session.commit()


@pytest_asyncio.fixture()
async def waiters(async_engine):
    """A waiters registry with a poller running against it.

    Polls fast so tests do not have to wait on the production interval.
    """
    registry: dict[int, set[asyncio.Future]] = {}
    poller = asyncio.create_task(poll_futures(async_engine, registry, poll_interval_sec=0.01))
    yield registry
    poller.cancel()
    with suppress(asyncio.CancelledError):
        await poller


@pytest.mark.asyncio
async def test_resolves_once_the_request_completes(waiters, sync_engine):
    request_id = insert_pending(sync_engine)[0]

    async def complete_soon():
        await asyncio.sleep(0.05)
        mark_completed(sync_engine, request_id, {"sequences": []})

    asyncio.create_task(complete_soon())
    result = await wait_for_future(waiters, request_id, timeout=5)

    assert result == (RequestStatus.COMPLETED, {"sequences": []})


@pytest.mark.asyncio
async def test_surfaces_failed_status(waiters, sync_engine):
    request_id = insert_pending(sync_engine)[0]
    mark_completed(sync_engine, request_id, {"error": "boom"}, status=RequestStatus.FAILED)

    assert await wait_for_future(waiters, request_id, timeout=5) == (RequestStatus.FAILED, {"error": "boom"})


@pytest.mark.asyncio
async def test_abandoned_request_times_out_and_leaves_no_entry(waiters, sync_engine):
    """A caller giving up gets None, and must drop out of the poll set.

    Otherwise the registry, and so the poll query, grows without bound.
    """
    request_id = insert_pending(sync_engine)[0]

    assert await wait_for_future(waiters, request_id, timeout=0.05) is None
    assert request_id not in waiters


@pytest.mark.asyncio
async def test_one_waiter_giving_up_does_not_strand_the_others(waiters, sync_engine):
    """Concurrent waiters on one id are routine, since the SDK retries a slow
    retrieve_future against the same request_id. One timing out must not cancel
    the rest."""
    request_id = insert_pending(sync_engine)[0]

    quick = asyncio.create_task(wait_for_future(waiters, request_id, timeout=0.05))
    patient = asyncio.create_task(wait_for_future(waiters, request_id, timeout=5))

    assert await quick is None
    mark_completed(sync_engine, request_id, {"ok": True})

    assert await patient == (RequestStatus.COMPLETED, {"ok": True})


@pytest.mark.asyncio
async def test_query_count_does_not_scale_with_waiters(waiters, sync_engine, async_engine):
    """The whole point of the shared poller: load must not scale with waiters."""
    from sqlalchemy import event

    request_ids = insert_pending(sync_engine, count=50)
    statements = []

    @event.listens_for(async_engine.sync_engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    tasks = [asyncio.create_task(wait_for_future(waiters, request_id, timeout=5)) for request_id in request_ids]
    # Let several poll iterations run while every request is still pending.
    await asyncio.sleep(0.1)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # 50 waiters over several ticks would be hundreds of statements if each
    # polled on its own; batched it is one per tick.
    assert 0 < len(statements) < 50


def _stub_request(async_engine, waiters):
    from types import SimpleNamespace

    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_engine=async_engine, future_waiters=waiters)))


@pytest.mark.asyncio
async def test_wait_raises_for_unknown_request(waiters):
    """The poller reports ids with no row, since rows are never deleted."""
    with pytest.raises(KeyError):
        await wait_for_future(waiters, 123456, timeout=5)


@pytest.mark.asyncio
async def test_retrieve_future_returns_completed_result(waiters, async_engine, sync_engine):
    from skyrl.tinker import api

    request_id = insert_pending(sync_engine)[0]
    mark_completed(sync_engine, request_id, {"sequences": [1]})

    result = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)), _stub_request(async_engine, waiters)
    )

    assert result == {"sequences": [1]}


@pytest.mark.asyncio
async def test_retrieve_future_404s_for_unknown_request(waiters, async_engine):
    from fastapi import HTTPException

    from skyrl.tinker import api

    with pytest.raises(HTTPException) as excinfo:
        await api.retrieve_future(api.RetrieveFutureRequest(request_id="123456"), _stub_request(async_engine, waiters))

    assert excinfo.value.status_code == 404
