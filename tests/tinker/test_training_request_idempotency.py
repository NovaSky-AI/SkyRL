"""Tests that retried training writes execute only once."""

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import create_future
from skyrl.tinker.db_models import FutureDB


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
