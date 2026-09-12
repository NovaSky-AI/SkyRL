import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import Session, SQLModel, create_engine

from skyrl.tinker import api, types
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    FutureDB,
    ModelDB,
    RequestStatus,
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


@pytest.mark.parametrize("outcome", ["success", "backend_error", "backend_value_error", "unloaded"])
@pytest.mark.parametrize("mode", ["training", "sampler", "ephemeral_sampler"])
def test_checkpoint_and_future_reach_matching_terminal_status(checkpoint_engine, mode, outcome):
    engine = checkpoint_engine
    training = mode == "training"
    request_type = types.RequestType.SAVE_WEIGHTS if training else types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER
    checkpoint_type = types.CheckpointType.TRAINING if training else types.CheckpointType.SAMPLER
    request_data = {"path": "checkpoint"}
    if mode == "ephemeral_sampler":
        request_data.update(sampling_session_seq_id=1, seq_id=1, sampling_session_id="sampling_session")
    with Session(engine.db_engine) as session:
        session.add(
            CheckpointDB(
                model_id="model",
                checkpoint_id="checkpoint",
                checkpoint_type=checkpoint_type,
                status=CheckpointStatus.PENDING,
            )
        )
        future = FutureDB(model_id="model", request_type=request_type, request_data=request_data)
        session.add(future)
        session.commit()
        request_id = future.request_id

    engine.backend.has_model.return_value = outcome != "unloaded"
    save = engine.backend.save_checkpoint if training else engine.backend.save_sampler_checkpoint
    if outcome == "backend_error":
        save.side_effect = OSError("checkpoint write failed")
    elif outcome == "backend_value_error":
        save.side_effect = ValueError("checkpoint write failed")

    with patch("skyrl.tinker.engine.logger") as captured_log:
        engine.process_single_requests({str(request_id): ("model", request_type, request_data)})

    if outcome == "unloaded":
        captured_log.exception.assert_not_called()
        captured_log.info.assert_called_once()
        assert "model not loaded" in captured_log.info.call_args.args[0]
        captured_log.warning.assert_not_called()
    elif outcome in ("backend_error", "backend_value_error"):
        assert captured_log.exception.call_count == 2
    else:
        captured_log.exception.assert_not_called()

    with Session(engine.db_engine) as session:
        future = session.get(FutureDB, request_id)
        checkpoint = session.get(CheckpointDB, ("model", "checkpoint", checkpoint_type))
        if outcome == "success":
            assert future.status == RequestStatus.COMPLETED
            assert checkpoint.status == CheckpointStatus.COMPLETED
            assert checkpoint.error_message is None
        else:
            assert future.status == RequestStatus.FAILED
            error = types.ErrorResponse.model_validate_json(future.result_data)
            expected_error = (
                "Model model not loaded (likely stale request from previous server)"
                if outcome == "unloaded"
                else "checkpoint write failed"
            )
            assert error.error == expected_error
            assert checkpoint.status == CheckpointStatus.FAILED
            assert checkpoint.error_message == expected_error
        assert future.completed_at is not None
        assert checkpoint.completed_at is not None

    if outcome == "unloaded":
        engine.backend.save_checkpoint.assert_not_called()
        engine.backend.save_sampler_checkpoint.assert_not_called()
    elif training:
        save.assert_called_once_with(engine.config.checkpoints_base / "model" / "checkpoint.tar.gz", "model")
    else:
        save.assert_called_once_with(
            engine.config.checkpoints_base / "model" / "sampler_weights" / "checkpoint.tar.gz",
            "model",
            persist=mode != "ephemeral_sampler",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["training", "sampler", "ephemeral_sampler"])
async def test_failed_save_can_be_deleted_and_retried(checkpoint_engine, monkeypatch, mode):
    engine = checkpoint_engine
    engine.backend.has_model.return_value = False
    async_db = create_async_engine(get_async_database_url(str(engine.db_engine.url)))
    monkeypatch.setattr(api.app.state, "db_engine", async_db, raising=False)
    monkeypatch.setattr(api.app.state, "engine_config", engine.config, raising=False)
    monkeypatch.setattr(api.app.state, "sampler_checkpoint_validation_lock", asyncio.Lock(), raising=False)
    monkeypatch.setattr(api.app.state, "validated_sampler_checkpoints", set(), raising=False)

    training = mode == "training"
    endpoint = "/api/v1/save_weights" if training else "/api/v1/save_weights_for_sampler"
    checkpoint_id = "ss1_seq1" if mode == "ephemeral_sampler" else "checkpoint"
    payload = {"model_id": "model"}
    if mode == "ephemeral_sampler":
        payload.update(sampling_session_seq_id=1, seq_id=1)
    else:
        payload["path"] = checkpoint_id
    path_kind = "weights" if training else "sampler_weights"
    delete_url = f"/api/v1/training_runs/model/checkpoints/{path_kind}/{checkpoint_id}"

    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
            saved = await client.post(endpoint, json=payload)
            assert saved.status_code == 200, saved.text
            request_id = saved.json()["request_id"]
            with Session(engine.db_engine) as session:
                requests = engine.find_single_requests(session)
            assert request_id in requests
            engine.process_single_requests(requests)
            with Session(engine.db_engine) as session:
                assert session.get(FutureDB, int(request_id)).status == RequestStatus.FAILED

            duplicate = await client.post(endpoint, json=payload)
            assert duplicate.status_code == 409
            deleted = await client.delete(delete_url)
            assert deleted.status_code == 204, deleted.text
            assert (await client.delete(delete_url)).status_code == 404
            retry = await client.post(endpoint, json=payload)
            assert retry.status_code == 200, retry.text
            assert retry.json()["request_id"] != request_id

        engine.backend.save_checkpoint.assert_not_called()
        engine.backend.save_sampler_checkpoint.assert_not_called()
    finally:
        await async_db.dispose()
