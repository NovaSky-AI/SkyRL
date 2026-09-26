import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine

from skyrl.tinker import api, types
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    ModelDB,
    SessionDB,
    get_async_database_url,
)
from skyrl.tinker.engine import TinkerEngine


@pytest.fixture
def checkpoint_engine(tmp_path):
    engine = object.__new__(TinkerEngine)
    engine.config = SimpleNamespace(checkpoints_base=tmp_path / "checkpoints")
    engine.backend = MagicMock()
    engine.db_engine = create_engine(f"sqlite:///{tmp_path / 'checkpoints.db'}")
    SQLModel.metadata.create_all(engine.db_engine)
    with Session(engine.db_engine) as session:
        session.add(SessionDB(session_id="session", sdk_version="test"))
        session.add(
            ModelDB(
                model_id="model",
                base_model="test-model",
                lora_config={"rank": 8, "alpha": 16},
                status="ready",
                request_id=0,
                session_id="session",
            )
        )
        session.commit()
    yield engine
    engine.db_engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,backend_error",
    [
        pytest.param("training", False, id="unloaded-training"),
        pytest.param("sampler", False, id="unloaded-sampler"),
        pytest.param("training", True, id="backend-value-error"),
    ],
)
async def test_failed_save_finalizes_checkpoint_and_allows_delete(checkpoint_engine, monkeypatch, mode, backend_error):
    engine = checkpoint_engine
    engine.backend.has_model.return_value = backend_error
    if backend_error:
        engine.backend.save_checkpoint.side_effect = ValueError("checkpoint write failed")
    async_db = create_async_engine(get_async_database_url(str(engine.db_engine.url)))
    monkeypatch.setattr(api.app.state, "db_engine", async_db, raising=False)
    monkeypatch.setattr(api.app.state, "engine_config", engine.config, raising=False)
    monkeypatch.setattr(api.app.state, "sampler_checkpoint_validation_lock", asyncio.Lock(), raising=False)
    monkeypatch.setattr(api.app.state, "validated_sampler_checkpoints", set(), raising=False)

    training = mode == "training"
    endpoint = "/api/v1/save_weights" if training else "/api/v1/save_weights_for_sampler"
    checkpoint_type = types.CheckpointType.TRAINING if training else types.CheckpointType.SAMPLER
    checkpoint_key = ("model", "checkpoint", checkpoint_type)
    payload = {"model_id": "model", "path": "checkpoint"}
    path_kind = "weights" if training else "sampler_weights"
    delete_url = f"/api/v1/training_runs/model/checkpoints/{path_kind}/checkpoint"

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            saved = await client.post(endpoint, json=payload)
            assert saved.status_code == 200, saved.text
            with Session(engine.db_engine) as session:
                assert session.get(CheckpointDB, checkpoint_key).status == CheckpointStatus.PENDING
                requests = engine.find_single_requests(session)
            engine.process_single_requests(requests)
            with Session(engine.db_engine) as session:
                assert session.get(CheckpointDB, checkpoint_key).status == CheckpointStatus.FAILED

            deleted = await client.delete(delete_url)
            assert deleted.status_code == 204, deleted.text
    finally:
        await async_db.dispose()
