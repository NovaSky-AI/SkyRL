"""Tests that retried training writes execute only once."""

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import create_future
from skyrl.tinker.db_models import FutureDB, TrainingRequestDB, enable_sqlite_wal


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
        original_id = await create_future(
            session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7
        )
        await session.commit()

    async with AsyncSession(engine) as session:
        retry_id = await create_future(
            session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7
        )
        await session.commit()
        futures = (await session.exec(select(FutureDB))).all()

    assert retry_id == original_id
    assert len(futures) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_training_request_sequence_reuse_with_new_payload_fails(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    async with AsyncSession(engine) as session:
        await create_future(
            session, types.RequestType.OPTIM_STEP, "model_1", optim_input(), seq_id=7
        )
        await session.commit()

    async with AsyncSession(engine) as session:
        with pytest.raises(HTTPException, match="sequence number was reused") as error:
            await create_future(
                session, types.RequestType.OPTIM_STEP, "model_1", optim_input(2e-4), seq_id=7
            )

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
        mappings = (await session.exec(select(TrainingRequestDB))).all()
    assert len(futures) == 1
    assert len(mappings) == 1
    assert futures[0].request_id == successes[0] == mappings[0].request_id
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

    assert second_id != first_id
    await engine.dispose()
