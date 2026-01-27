from cloudpathlib import AnyPath
from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel

from tx.tinker.engine import TinkerEngine
from tx.tinker.config import EngineConfig
from tx.tinker import types
from tx.tinker.db_models import SessionDB, ModelDB, FutureDB, RequestStatus


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


class TestMaxMicroBatches:
    """Tests for max_micro_batches limiting in find_batchable_model_passes."""

    @staticmethod
    def _make_request_data(num_sequences: int) -> dict:
        """Create a ForwardBackwardInput request data with the given number of sequences."""
        data = []
        for _ in range(num_sequences):
            data.append(
                {
                    "model_input": {"chunks": [{"tokens": [1, 2, 3]}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": [2, 3, 4]},
                        "weights": {"data": [1.0, 1.0, 1.0]},
                        "advantages": {"data": [0.0, 0.0, 0.0]},
                        "logprobs": {"data": [0.0, 0.0, 0.0]},
                    },
                }
            )
        return {"data": data, "loss_fn": "cross_entropy"}

    @staticmethod
    def _create_engine(train_micro_batch_size: int, max_micro_batches: int) -> TinkerEngine:
        """Create an engine with the given micro batch configuration."""
        config = EngineConfig(
            base_model=BASE_MODEL,
            checkpoints_base=AnyPath(""),
            backend_config={
                "max_lora_adapters": 4,
                "max_lora_rank": 32,
                "train_micro_batch_size": train_micro_batch_size,
            },
            max_micro_batches=max_micro_batches,
            database_url="sqlite:///:memory:",
        )
        engine = TinkerEngine(config)
        SQLModel.metadata.create_all(engine.db_engine)
        return engine

    def _add_requests(self, engine: TinkerEngine, sequence_counts: list[int]):
        """Add FORWARD_BACKWARD requests with the given sequence counts."""
        with Session(engine.db_engine) as session:
            for num_sequences in sequence_counts:
                session.add(
                    FutureDB(
                        request_type=types.RequestType.FORWARD_BACKWARD,
                        model_id="model1",
                        request_data=self._make_request_data(num_sequences),
                        status=RequestStatus.PENDING,
                    )
                )
            session.commit()

    @pytest.mark.parametrize(
        "train_micro_batch_size,max_micro_batches,sequence_counts,expected_count",
        [
            # Gradient accumulation mode: ceil(16/4) + ceil(20/4) = 4 + 5 = 9 <= 10, ceil(8/4) = 2 would exceed
            (4, 10, [16, 20, 8], 2),
            # Full batch mode: each request counts as 1, so 3 requests fit in max_micro_batches=3
            (0, 3, [100, 200, 50, 75], 3),
            # Disabled: all requests included when max_micro_batches=0
            (4, 0, [50] * 10, 10),
        ],
        ids=["gradient_accumulation", "full_batch_mode", "disabled"],
    )
    def test_micro_batch_limiting(self, train_micro_batch_size, max_micro_batches, sequence_counts, expected_count):
        """Test that micro batches are limited correctly under different configurations."""
        engine = self._create_engine(train_micro_batch_size, max_micro_batches)
        self._add_requests(engine, sequence_counts)

        with Session(engine.db_engine) as session:
            result = engine.find_batchable_model_passes(session, types.RequestType.FORWARD_BACKWARD)

        assert len(result) == expected_count

    def test_always_includes_at_least_one_request(self):
        """Test that at least one request is always included even if it exceeds the limit."""
        # train_micro_batch_size=4, max_micro_batches=10
        # Request with 100 sequences = ceil(100/4) = 25 micro batches > 10
        # Should still be included to avoid starvation
        engine = self._create_engine(train_micro_batch_size=4, max_micro_batches=10)
        self._add_requests(engine, [100])

        with Session(engine.db_engine) as session:
            result = engine.find_batchable_model_passes(session, types.RequestType.FORWARD_BACKWARD)

        assert len(result) == 1
        _, req_data = list(result.values())[0]
        assert len(req_data.data) == 100
