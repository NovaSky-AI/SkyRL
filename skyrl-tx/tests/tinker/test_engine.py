from cloudpathlib import AnyPath
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, SQLModel

from tx.tinker.engine import TinkerEngine, prepare_model_pass_batch
from tx.tinker.config import EngineConfig
from tx.tinker import types
from tx.tinker.db_models import SessionDB, ModelDB


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def test_process_unload_model():
    """Test that process_unload_model removes model from backend."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "test_model"
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    result = engine.process_unload_model(model_id, types.UnloadModelInput())
    assert result.status == "unloaded"
    assert not engine.backend.has_model(model_id)


def test_cleanup_stale_sessions():
    """Test that cleanup_stale_sessions unloads models from expired sessions."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
        session_timeout_sec=60,
        database_url="sqlite:///:memory:",  # Use in-memory DB for test isolation
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "stale_model"
    session_id = "stale_session"

    # Create model in backend
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    # Insert stale session and model into DB
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
    with Session(engine.db_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="test",
                status="active",
                last_heartbeat_at=stale_heartbeat,
            )
        )
        session.add(
            ModelDB(
                model_id=model_id,
                base_model=BASE_MODEL,
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="ready",
                request_id=1,
                session_id=session_id,
            )
        )
        session.commit()

    # Run cleanup and assert one model was unloaded
    assert engine.cleanup_stale_sessions() == 1
    assert not engine.backend.has_model(model_id)


def test_prepare_model_pass_batch_loss_fn_config():
    """Test that prepare_model_pass_batch extracts loss_fn_config from requests."""
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.ModelInputChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=[]),
            logprobs=types.TensorData(data=[]),
        ),
    )
    config = {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3}

    # With loss_fn_config
    requests_with_config = {
        "req1": (
            "model1",
            types.ForwardBackwardInput(
                data=[datum],
                loss_fn="ppo",
                loss_fn_config=config,
            ),
        ),
    }
    batch = prepare_model_pass_batch(requests_with_config)
    assert batch.all_loss_fns == ["ppo"]
    assert batch.all_loss_fn_configs == [config]

    # Without loss_fn_config (default None)
    requests_without_config = {
        "req2": (
            "model1",
            types.ForwardBackwardInput(
                data=[datum],
                loss_fn="cross_entropy",
            ),
        ),
    }
    batch_no_config = prepare_model_pass_batch(requests_without_config)
    assert batch_no_config.all_loss_fns == ["cross_entropy"]
    assert batch_no_config.all_loss_fn_configs == [None]
