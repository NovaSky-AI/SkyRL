"""Tests for waiting on asynchronous request results (api.wait_for_future/poll_futures)."""

import asyncio
from contextlib import suppress

import pytest
import pytest_asyncio
from pydantic import BaseModel
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


# A completed sample request's payload. Stored and asserted on as the real type,
# so these tests break if the round trip through the column stops preserving it.
SAMPLE_RESULT = types.SampleOutput(sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1], logprobs=[-0.5])])


def mark_completed(sync_engine, request_id: int, result: BaseModel, status=RequestStatus.COMPLETED) -> None:
    """Complete a request the way the engine does: a model serialized to JSON text."""
    with Session(sync_engine) as session:
        row = session.get(FutureDB, request_id)
        row.result_data = result.model_dump_json()
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
        mark_completed(sync_engine, request_id, SAMPLE_RESULT)

    asyncio.create_task(complete_soon())
    status, request_type, result_data = await wait_for_future(waiters, request_id, timeout=5)

    # result_data is the stored JSON text, not a decoded object, so it takes a
    # parse to compare against the result that was stored.
    assert request_type == types.RequestType.SAMPLE
    assert (status, types.SampleOutput.model_validate_json(result_data)) == (RequestStatus.COMPLETED, SAMPLE_RESULT)


@pytest.mark.asyncio
async def test_surfaces_failed_status(waiters, sync_engine):
    request_id = insert_pending(sync_engine)[0]
    error = types.ErrorResponse(error="boom", status="failed")
    mark_completed(sync_engine, request_id, error, status=RequestStatus.FAILED)

    status, request_type, result_data = await wait_for_future(waiters, request_id, timeout=5)

    assert request_type == types.RequestType.SAMPLE
    assert (status, types.ErrorResponse.model_validate_json(result_data)) == (RequestStatus.FAILED, error)


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
    # Any request type's output can land here, not just a sample's.
    result = types.OptimStepOutput(metrics={"loss": 1.5})
    mark_completed(sync_engine, request_id, result)

    status, request_type, result_data = await patient
    assert request_type == types.RequestType.SAMPLE
    assert (status, types.OptimStepOutput.model_validate_json(result_data)) == (RequestStatus.COMPLETED, result)


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


def _stub_request(async_engine, waiters, external_future_store=None, headers: dict | None = None):
    from types import SimpleNamespace

    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_engine=async_engine,
                external_future_store=external_future_store,
                future_waiters=waiters,
            )
        ),
        headers=headers or {},
    )


@pytest.mark.asyncio
async def test_wait_raises_for_unknown_request(waiters):
    """The poller reports ids with no row, since rows are never deleted."""
    with pytest.raises(KeyError):
        await wait_for_future(waiters, 123456, timeout=5)


@pytest.mark.asyncio
async def test_retrieve_future_returns_completed_result(waiters, async_engine, sync_engine):
    from skyrl.tinker import api

    request_id = insert_pending(sync_engine)[0]
    mark_completed(sync_engine, request_id, SAMPLE_RESULT)

    response = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)), _stub_request(async_engine, waiters)
    )

    # The stored JSON text is returned as-is rather than re-encoded by FastAPI.
    assert response.media_type == "application/json"
    assert response.body == SAMPLE_RESULT.model_dump_json().encode()


@pytest.mark.asyncio
async def test_retrieve_future_returns_in_memory_external_result(waiters, async_engine):
    from skyrl.tinker import api
    from skyrl.tinker.extra import InMemoryFutureStore

    store = InMemoryFutureStore()
    request_id = store.create_future()
    store.complete_future(request_id, RequestStatus.COMPLETED, SAMPLE_RESULT.model_dump_json())

    response = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)),
        _stub_request(async_engine, waiters, store),
    )

    assert response.media_type == "application/json"
    assert response.body == SAMPLE_RESULT.model_dump_json().encode()


@pytest.mark.asyncio
async def test_retrieve_future_400s_with_the_stored_error(waiters, async_engine, sync_engine):
    """The failure path still decodes the payload, since it inspects the error."""
    from fastapi import HTTPException

    from skyrl.tinker import api

    request_id = insert_pending(sync_engine)[0]
    error = types.ErrorResponse(error="boom", status="failed")
    mark_completed(sync_engine, request_id, error, status=RequestStatus.FAILED)

    with pytest.raises(HTTPException) as excinfo:
        await api.retrieve_future(
            api.RetrieveFutureRequest(request_id=str(request_id)), _stub_request(async_engine, waiters)
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "boom"


@pytest.mark.asyncio
async def test_retrieve_future_serves_proto_when_accepted(waiters, async_engine, sync_engine):
    """A completed sample future is served as proto bytes when the client's
    Accept header asks for it (the JSON test above covers the default path)."""
    from tinker import SampleResponse
    from tinker.proto.response_conv import deserialize_proto_response

    from skyrl.tinker import api

    request_id = insert_pending(sync_engine)[0]
    sample_result = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1, 2], logprobs=[-0.5, -1.0])]
    )
    mark_completed(sync_engine, request_id, sample_result)

    result = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)),
        _stub_request(async_engine, waiters, headers={"accept": "Application/X-Protobuf, application/json"}),
    )

    assert result.media_type == "application/x-protobuf"
    response = deserialize_proto_response(result.body, SampleResponse)
    assert response.sequences[0].tokens == [1, 2]


@pytest.mark.asyncio
async def test_retrieve_future_404s_for_unknown_request(waiters, async_engine):
    from fastapi import HTTPException

    from skyrl.tinker import api

    with pytest.raises(HTTPException) as excinfo:
        await api.retrieve_future(api.RetrieveFutureRequest(request_id="123456"), _stub_request(async_engine, waiters))

    assert excinfo.value.status_code == 404
