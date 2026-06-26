"""Fast tests for checkpoint deletion API behavior."""

import asyncio
from collections.abc import AsyncGenerator, Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.api import app, get_session
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import CheckpointDB, CheckpointStatus, ModelDB, SessionDB

MODEL_ID = "model_fast_delete"


@pytest.fixture
def checkpoint_api(tmp_path: Path) -> Iterator[tuple[TestClient, Path, AsyncEngine]]:
    db_path = tmp_path / "checkpoint_delete.db"
    checkpoint_base = tmp_path / "checkpoints"
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async def setup_db() -> None:
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with AsyncSession(db_engine) as session:
            yield session

    asyncio.run(setup_db())
    previous_state = app.state._state.copy()
    app.state.db_engine = db_engine
    app.state.engine_config = EngineConfig(
        base_model="test-model",
        checkpoints_base=checkpoint_base,
        database_url=f"sqlite:///{db_path}",
    )
    app.dependency_overrides[get_session] = override_get_session
    client = TestClient(app)

    try:
        yield client, checkpoint_base, db_engine
    finally:
        client.close()
        app.dependency_overrides.pop(get_session, None)
        app.state._state.clear()
        app.state._state.update(previous_state)
        asyncio.run(db_engine.dispose())


async def seed_model_and_checkpoints(
    db_engine: AsyncEngine,
    checkpoint_ids: list[str],
    checkpoint_type: types.CheckpointType = types.CheckpointType.TRAINING,
) -> None:
    async with AsyncSession(db_engine) as session:
        if await session.get(SessionDB, "session_fast_delete") is None:
            session.add(SessionDB(session_id="session_fast_delete", tags=[], sdk_version="test"))
        if await session.get(ModelDB, MODEL_ID) is None:
            session.add(
                ModelDB(
                    model_id=MODEL_ID,
                    base_model="test-model",
                    lora_config={"rank": 1},
                    status="created",
                    request_id=1,
                    session_id="session_fast_delete",
                )
            )
        for checkpoint_id in checkpoint_ids:
            session.add(
                CheckpointDB(
                    model_id=MODEL_ID,
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=checkpoint_type,
                    status=CheckpointStatus.COMPLETED,
                    completed_at=datetime.now(timezone.utc),
                )
            )
        await session.commit()


def write_training_checkpoint(checkpoint_base: Path, checkpoint_id: str, directory: bool = False) -> Path:
    checkpoint_path = checkpoint_base / MODEL_ID / f"{checkpoint_id}.tar.gz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        checkpoint_path.mkdir()
        (checkpoint_path / "checkpoint_0").write_text("tiny")
    else:
        checkpoint_path.write_text("tiny")
    return checkpoint_path


def write_sampler_checkpoint(checkpoint_base: Path, checkpoint_id: str) -> Path:
    checkpoint_path = checkpoint_base / MODEL_ID / "sampler_weights" / f"{checkpoint_id}.tar.gz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("tiny")
    return checkpoint_path


def listed_checkpoint_ids(client: TestClient) -> set[str]:
    response = client.get(f"/api/v1/training_runs/{MODEL_ID}/checkpoints")
    assert response.status_code == 200
    return {checkpoint["checkpoint_id"] for checkpoint in response.json()["checkpoints"]}


def test_delete_checkpoint_removes_saved_artifact_and_list_entry(
    checkpoint_api: tuple[TestClient, Path, AsyncEngine],
) -> None:
    client, checkpoint_base, db_engine = checkpoint_api
    asyncio.run(seed_model_and_checkpoints(db_engine, ["delete_training"]))
    training_checkpoint = write_training_checkpoint(checkpoint_base, "delete_training", directory=True)

    response = client.delete(f"/api/v1/training_runs/{MODEL_ID}/checkpoints/weights/delete_training")

    assert response.status_code == 204
    assert not training_checkpoint.exists()
    assert "delete_training" not in listed_checkpoint_ids(client)

    asyncio.run(
        seed_model_and_checkpoints(db_engine, ["delete_sampler"], checkpoint_type=types.CheckpointType.SAMPLER)
    )
    sampler_checkpoint = write_sampler_checkpoint(checkpoint_base, "delete_sampler")

    response = client.delete(f"/api/v1/training_runs/{MODEL_ID}/checkpoints/delete_sampler")

    assert response.status_code == 204
    assert not sampler_checkpoint.exists()
    assert "delete_sampler" not in listed_checkpoint_ids(client)


def test_delete_even_checkpoints_leaves_odd_checkpoints_listed(
    checkpoint_api: tuple[TestClient, Path, AsyncEngine],
) -> None:
    client, checkpoint_base, db_engine = checkpoint_api
    checkpoint_ids = ["1", "2", "3", "4", "5"]
    asyncio.run(seed_model_and_checkpoints(db_engine, checkpoint_ids))
    checkpoint_files = {
        checkpoint_id: write_training_checkpoint(checkpoint_base, checkpoint_id) for checkpoint_id in checkpoint_ids
    }

    assert listed_checkpoint_ids(client) == {"1", "2", "3", "4", "5"}

    assert client.delete(f"/api/v1/training_runs/{MODEL_ID}/checkpoints/2").status_code == 204
    assert client.delete(f"/api/v1/training_runs/{MODEL_ID}/checkpoints/4").status_code == 204

    assert checkpoint_files["1"].exists()
    assert not checkpoint_files["2"].exists()
    assert checkpoint_files["3"].exists()
    assert not checkpoint_files["4"].exists()
    assert checkpoint_files["5"].exists()
    assert listed_checkpoint_ids(client) == {"1", "3", "5"}
