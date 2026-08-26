"""Tests that retried training writes execute only once."""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import create_future, open_training_request_write_session
from skyrl.tinker.db_models import FutureDB, enable_sqlite_wal


class LargeTrainingInput(BaseModel):
    payload: str


def optim_input(learning_rate: float = 1e-4) -> types.OptimStepInput:
    return types.OptimStepInput(
        adam_params=types.AdamParams(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-12,
            weight_decay=0.0,
        )
    )


@pytest.mark.asyncio
async def test_training_request_retry_returns_original_future(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        original_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()

    async with AsyncSession(engine) as session:
        retry_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()
        futures = (await session.exec(select(FutureDB))).all()

    assert retry_id == original_id
    assert len(futures) == 1
    assert futures[0].seq_id == 7
    await engine.dispose()


@pytest.mark.asyncio
async def test_training_request_sequence_reuse_with_new_payload_fails(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7)
        await session.commit()

    async with AsyncSession(engine) as session:
        with pytest.raises(HTTPException, match="sequence number was reused") as error:
            await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input(2e-4), seq_id=7)

    assert error.value.status_code == 409
    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_training_request_types_cannot_reuse_sequence(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    start = asyncio.Event()

    async def create(request_type, request_data):
        async with AsyncSession(engine) as session:
            await start.wait()
            try:
                request_id = await create_future(session, request_type, "model_1", request_data, seq_id=7)
                await session.commit()
                return request_id
            except HTTPException as error:
                return error

    requests = [
        asyncio.create_task(
            create(
                types.RequestType.FORWARD_BACKWARD,
                types.ForwardBackwardInput(data=[], loss_fn="cross_entropy"),
            )
        ),
        asyncio.create_task(create(types.RequestType.OPTIM_STEP, optim_input())),
    ]
    start.set()
    results = await asyncio.gather(*requests)

    successes = [result for result in results if isinstance(result, int)]
    conflicts = [result for result in results if isinstance(result, HTTPException)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert conflicts[0].status_code == 409
    assert "sequence number was reused" in conflicts[0].detail

    async with AsyncSession(engine) as session:
        futures = (await session.exec(select(FutureDB))).all()
    assert len(futures) == 1
    assert futures[0].request_id == successes[0]
    assert futures[0].seq_id == 7
    await engine.dispose()


@pytest.mark.asyncio
async def test_training_requests_without_sequence_numbers_remain_distinct(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        first_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input())
        second_id = await create_future(session, types.RequestType.OPTIM_STEP, "model_1", optim_input())
        await session.commit()
        futures = (await session.exec(select(FutureDB))).all()

    # Two identical requests for one model, both with seq_id NULL, must not collide on
    # the futures (model_id, seq_id) constraint: SQL counts NULLs as distinct. That is
    # what keeps clients which send no seq_id on the pre-idempotency behaviour.
    assert second_id != first_id
    assert len(futures) == 2
    assert [future.seq_id for future in futures] == [None, None]
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlite_training_writes_wait_before_connection_checkout(tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}",
        pool_size=2,
        max_overflow=0,
        pool_timeout=0.05,
    )
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    app = FastAPI()
    app.state.db_engine = engine
    app.state.training_request_write_lock = asyncio.Lock()
    request = Request({"type": "http", "app": app})
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()

    async def write_training_request(seq_id: int) -> None:
        async with open_training_request_write_session(request) as session:
            await create_future(
                session,
                types.RequestType.FORWARD_BACKWARD,
                "model_1",
                LargeTrainingInput(payload="x" * 100_000),
                seq_id=seq_id,
            )
            if seq_id == 0:
                first_write_started.set()
                await release_first_write.wait()
            await session.commit()

    writes = [asyncio.create_task(write_training_request(seq_id)) for seq_id in range(32)]
    await first_write_started.wait()

    async with AsyncSession(engine) as heartbeat_session:
        count = await asyncio.wait_for(heartbeat_session.exec(select(func.count()).select_from(FutureDB)), timeout=0.1)
    assert count.one() == 1

    release_first_write.set()
    await asyncio.gather(*writes)

    async with AsyncSession(engine) as session:
        count = await session.exec(select(func.count()).select_from(FutureDB))
    assert count.one() == 32
    await engine.dispose()
