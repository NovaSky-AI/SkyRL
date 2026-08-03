"""Tests for the shared future waiter and the completion write path."""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine, select

from skyrl.tinker import types
from skyrl.tinker.db_models import (
    FutureDB,
    RequestStatus,
    enable_sqlite_wal,
    get_async_database_url,
    json_engine_kwargs,
)
from skyrl.tinker.futures import FutureWaiter, complete_future


@pytest.fixture()
def db_url(tmp_path):
    """A file-backed SQLite database with the schema created.

    A file rather than :memory: so the sync and async engines below see the same
    database.
    """
    url = f"sqlite:///{tmp_path / 'tinker.db'}"
    sync_engine = create_engine(url, **json_engine_kwargs())
    enable_sqlite_wal(sync_engine)
    SQLModel.metadata.create_all(sync_engine)
    sync_engine.dispose()
    return url


@pytest.fixture()
def sync_engine(db_url):
    engine = create_engine(db_url, **json_engine_kwargs())
    enable_sqlite_wal(engine)
    yield engine
    engine.dispose()


@pytest_asyncio.fixture()
async def async_engine(db_url):
    engine = create_async_engine(get_async_database_url(db_url), **json_engine_kwargs())
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
async def waiter(async_engine):
    # Poll fast so tests do not have to wait on the production interval.
    instance = FutureWaiter(async_engine, poll_interval_sec=0.01)
    await instance.start()
    yield instance
    await instance.stop()


@pytest.mark.asyncio
async def test_wait_returns_result_once_completed(waiter, sync_engine):
    request_id = insert_pending(sync_engine)[0]

    async def complete_soon():
        await asyncio.sleep(0.05)
        mark_completed(sync_engine, request_id, {"sequences": []})

    asyncio.create_task(complete_soon())
    result = await waiter.wait(request_id, timeout=5)

    assert result is not None
    assert result.status == RequestStatus.COMPLETED
    assert result.result_data == {"sequences": []}


@pytest.mark.asyncio
async def test_wait_surfaces_failed_status(waiter, sync_engine):
    request_id = insert_pending(sync_engine)[0]
    mark_completed(sync_engine, request_id, {"error": "boom"}, status=RequestStatus.FAILED)

    result = await waiter.wait(request_id, timeout=5)

    assert result.status == RequestStatus.FAILED
    assert result.result_data == {"error": "boom"}


@pytest.mark.asyncio
async def test_wait_raises_for_unknown_request(waiter):
    with pytest.raises(KeyError):
        await waiter.wait(123456, timeout=5)


@pytest.mark.asyncio
async def test_wait_returns_none_on_timeout(waiter, sync_engine):
    request_id = insert_pending(sync_engine)[0]

    assert await waiter.wait(request_id, timeout=0.05) is None


@pytest.mark.asyncio
async def test_many_waiters_share_one_query_per_poll(waiter, sync_engine):
    """The whole point of the shared poller: load must not scale with waiters."""
    request_ids = insert_pending(sync_engine, count=50)

    statements = []
    from sqlalchemy import event

    @event.listens_for(waiter._db_engine.sync_engine, "before_cursor_execute")
    def _count(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    tasks = [asyncio.create_task(waiter.wait(request_id, timeout=5)) for request_id in request_ids]
    # Let several poll iterations run while every request is still pending.
    await asyncio.sleep(0.1)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # 50 waiters over several ticks would be hundreds of statements if each
    # polled on its own; batched it is one per tick.
    assert 0 < len(statements) < 50


@pytest.mark.asyncio
async def test_multiple_waiters_on_same_request_all_resolve(waiter, sync_engine):
    request_id = insert_pending(sync_engine)[0]
    mark_completed(sync_engine, request_id, {"ok": True})

    results = await asyncio.gather(*(waiter.wait(request_id, timeout=5) for _ in range(3)))

    assert all(result.result_data == {"ok": True} for result in results)


@pytest.mark.asyncio
async def test_notify_resolves_without_polling(async_engine, sync_engine):
    """notify() must resolve waiters even when the poller is not running."""
    request_id = insert_pending(sync_engine)[0]
    waiter = FutureWaiter(async_engine, poll_interval_sec=3600)

    async def notify_soon():
        await asyncio.sleep(0.01)
        waiter.notify(request_id, RequestStatus.COMPLETED, {"fast": True})

    asyncio.create_task(notify_soon())
    result = await waiter.wait(request_id, timeout=5)

    assert result.result_data == {"fast": True}


@pytest.mark.asyncio
async def test_complete_future_writes_result(async_engine, sync_engine):
    request_id = insert_pending(sync_engine)[0]

    written = await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"sequences": [1]})

    assert written is True
    with Session(sync_engine) as session:
        row = session.get(FutureDB, request_id)
        assert row.status == RequestStatus.COMPLETED
        assert row.result_data == {"sequences": [1]}
        assert row.completed_at is not None


@pytest.mark.asyncio
async def test_complete_future_reports_missing_row(async_engine):
    assert await complete_future(async_engine, 999999, RequestStatus.COMPLETED, {}) is False


@pytest.mark.asyncio
async def test_complete_future_leaves_request_data_intact(async_engine, sync_engine):
    """The completion write updates by primary key and must not touch the payload."""
    request_id = insert_pending(sync_engine)[0]

    await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"done": True})

    with Session(sync_engine) as session:
        assert session.get(FutureDB, request_id).request_data == {"checkpoint_id": ""}


@pytest.mark.asyncio
async def test_complete_future_notifies_waiter(async_engine, sync_engine):
    request_id = insert_pending(sync_engine)[0]
    waiter = FutureWaiter(async_engine, poll_interval_sec=3600)
    wait_task = asyncio.create_task(waiter.wait(request_id, timeout=5))
    await asyncio.sleep(0)  # let the waiter register

    await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"x": 1}, waiter)

    result = await wait_task
    assert result.result_data == {"x": 1}


def test_json_codec_round_trips_payloads(sync_engine):
    """Whichever JSON codec is installed must round-trip payloads unchanged."""
    payload = {
        "tokens": [1, 2, 3],
        "logprobs": [-0.5, -1.25, 0.0],
        "nested": {"a": [True, False, None], "b": "text"},
    }
    with Session(sync_engine) as session:
        row = FutureDB(
            request_type=types.RequestType.FORWARD_BACKWARD,
            model_id="model_a",
            request_data=payload,
            status=RequestStatus.PENDING,
        )
        session.add(row)
        session.commit()
        request_id = row.request_id

    with Session(sync_engine) as session:
        stored = session.exec(select(FutureDB.request_data).where(FutureDB.request_id == request_id)).first()
    assert stored == payload


@pytest.mark.asyncio
async def test_complete_future_retries_transient_failures(async_engine, sync_engine, monkeypatch):
    """A pool-checkout timeout must not orphan the request.

    Nothing else can complete an EXTERNAL request, so a single transient failure
    here would leave it pending forever.
    """
    from sqlalchemy.exc import TimeoutError as SATimeoutError

    from skyrl.tinker import futures

    request_id = insert_pending(sync_engine)[0]
    monkeypatch.setattr(futures, "_RESULT_WRITE_BACKOFF_SEC", 0.001)

    real_exec = futures.AsyncSession.exec
    calls = []

    async def flaky_exec(self, statement, *args, **kwargs):
        calls.append(statement)
        if len(calls) <= 2:
            raise SATimeoutError("QueuePool limit reached")
        return await real_exec(self, statement, *args, **kwargs)

    monkeypatch.setattr(futures.AsyncSession, "exec", flaky_exec)

    assert await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"ok": 1}) is True
    assert len(calls) == 3  # two failures, then success

    monkeypatch.undo()
    with Session(sync_engine) as session:
        assert session.get(FutureDB, request_id).status == RequestStatus.COMPLETED


@pytest.mark.asyncio
async def test_complete_future_raises_after_exhausting_retries(async_engine, sync_engine, monkeypatch):
    """When retries run out the caller must find out, not fail silently."""
    from sqlalchemy.exc import TimeoutError as SATimeoutError

    from skyrl.tinker import futures

    request_id = insert_pending(sync_engine)[0]
    monkeypatch.setattr(futures, "_RESULT_WRITE_BACKOFF_SEC", 0.001)

    async def always_fails(self, statement, *args, **kwargs):
        raise SATimeoutError("QueuePool limit reached")

    monkeypatch.setattr(futures.AsyncSession, "exec", always_fails)

    with pytest.raises(SATimeoutError):
        await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"ok": 1})


@pytest.mark.asyncio
async def test_complete_future_does_not_retry_programming_errors(async_engine, sync_engine, monkeypatch):
    """Only transient failures are retried; real bugs surface immediately."""
    from skyrl.tinker import futures

    request_id = insert_pending(sync_engine)[0]
    calls = []

    async def raises_value_error(self, statement, *args, **kwargs):
        calls.append(statement)
        raise ValueError("bad payload")

    monkeypatch.setattr(futures.AsyncSession, "exec", raises_value_error)

    with pytest.raises(ValueError):
        await complete_future(async_engine, request_id, RequestStatus.COMPLETED, {"ok": 1})
    assert len(calls) == 1
