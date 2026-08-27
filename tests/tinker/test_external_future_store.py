import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.requests import Request

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    FutureDB,
    ModelDB,
    RequestStatus,
    SamplingSessionDB,
    SessionDB,
    enable_sqlite_wal,
    get_async_database_url,
)
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


def _sample_input(seq_id: int) -> types.SampleInput:
    return types.SampleInput(
        base_model="model_a",
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[seq_id])]),
        sampling_params=types.SamplingParams(
            temperature=0.0, max_tokens=1, seed=seq_id
        ),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
        seq_id=seq_id,
    )


class _CompletingForwarder:
    def __init__(self, store: ExternalFutureStore):
        self.store = store

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        base_model: str | None = None,
    ) -> None:
        await self.store.complete(
            request_id,
            types.SampleOutput(sequences=[]),
            RequestStatus.COMPLETED,
        )


def _forward_backward_request(seq_id: int, db_write_lock: asyncio.Lock) -> Request:
    body = api.ForwardBackwardRequest(
        model_id="model_a",
        seq_id=seq_id,
        forward_backward_input=api.ForwardBackwardInput(
            data=[
                api.Datum(
                    model_input=api.ModelInput(
                        chunks=[api.EncodedTextChunk(tokens=[1, 2])]
                    ),
                    loss_fn_inputs={
                        "target_tokens": api.TensorData(data=[2, 3]),
                        "weights": api.TensorData(data=[1.0, 1.0]),
                    },
                )
            ],
            loss_fn="cross_entropy",
        ),
    ).model_dump_json().encode()
    body_sent = False

    async def receive():
        nonlocal body_sent
        if body_sent:
            return {"type": "http.disconnect"}
        body_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    app = SimpleNamespace(state=SimpleNamespace(db_write_lock=db_write_lock))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/forward_backward",
            "headers": [(b"content-type", b"application/json")],
            "app": app,
        },
        receive,
    )


@pytest_asyncio.fixture()
async def future_store(tmp_path):
    db_url = get_async_database_url(f"sqlite:///{tmp_path / 'tinker.db'}")
    engine = create_async_engine(
        db_url, pool_size=5, max_overflow=10, pool_timeout=0.1
    )
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    db_write_lock = asyncio.Lock()
    store = ExternalFutureStore(engine, db_write_lock)
    await store.start()
    yield store, engine, db_write_lock
    await store.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_two_full_rollout_waves_complete_and_persist(future_store):
    store, engine, db_write_lock = future_store
    forwarder = _CompletingForwarder(store)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_future_store=store,
                external_inference_client=forwarder,
                db_write_lock=db_write_lock,
                sampling_model_cache={},
                sampling_model_cache_lock=asyncio.Lock(),
            )
        )
    )
    result_data = types.SampleOutput(sequences=[]).model_dump_json()

    async with AsyncSession(engine) as session:
        session.add(
            SessionDB(
                session_id="session_a",
                tags=[],
                user_metadata={},
                sdk_version="test",
            )
        )
        session.add(
            SamplingSessionDB(
                sampling_session_id="session_a",
                session_id="session_a",
                sampling_session_seq_id=0,
                base_model="model_a",
            )
        )
        await session.commit()

    for wave in range(2):
        async def create_sample(index: int) -> int:
            async with AsyncSession(engine) as session:
                response = await api.asample(
                    api.SampleRequest(
                        prompt=api.ModelInput(
                            chunks=[api.EncodedTextChunk(tokens=[index])]
                        ),
                        sampling_params=api.SamplingParams(
                            temperature=0.0, max_tokens=1, seed=index
                        ),
                        sampling_session_id="session_a",
                        seq_id=wave * 512 + index,
                    ),
                    request,
                    session,
                )
            return int(response.request_id)

        async def heartbeat() -> None:
            async with AsyncSession(engine) as session:
                await api.session_heartbeat(
                    api.SessionHeartbeatRequest(session_id="session_a"),
                    request,
                    session,
                )

        responses = await asyncio.gather(
            *(create_sample(index) for index in range(512)),
            *(heartbeat() for _ in range(32)),
        )
        request_ids = responses[:512]
        waiters = [
            asyncio.create_task(store.wait(request_id, timeout=5))
            for request_id in request_ids
        ]

        async with AsyncSession(engine) as session:
            pending_after_creation = (
                await session.exec(
                    select(func.count())
                    .select_from(FutureDB)
                    .where(FutureDB.status == RequestStatus.PENDING)
                )
            ).one()
        assert pending_after_creation == 0

        results = await asyncio.gather(*waiters)

        assert all(
            result == (RequestStatus.COMPLETED, types.RequestType.EXTERNAL, result_data)
            for result in results
        )

    await store.flush()
    async with AsyncSession(engine) as session:
        persisted = (
            await session.exec(select(func.count()).select_from(FutureDB))
        ).one()
        pending = (
            await session.exec(
                select(func.count())
                .select_from(FutureDB)
                .where(FutureDB.status == RequestStatus.PENDING)
            )
        ).one()
        session_db = await session.get(SessionDB, "session_a")

    assert persisted == 1024
    assert pending == 0
    assert session_db is not None
    assert session_db.heartbeat_count == 64


@pytest.mark.asyncio
async def test_sustained_model_path_rollouts_training_futures_and_heartbeats(future_store):
    store, engine, db_write_lock = future_store
    forwarder = _CompletingForwarder(store)
    sample_request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_future_store=store,
                external_inference_client=forwarder,
                engine_config=EngineConfig(base_model="model_a"),
                db_write_lock=db_write_lock,
                sampling_model_cache={},
                sampling_model_cache_lock=asyncio.Lock(),
                validated_sampler_checkpoints=set(),
                sampler_checkpoint_validation_lock=asyncio.Lock(),
            )
        )
    )

    async with AsyncSession(engine) as session:
        session.add(
            SessionDB(
                session_id="session_a",
                tags=[],
                user_metadata={},
                sdk_version="test",
            )
        )
        session.add(
            SamplingSessionDB(
                sampling_session_id="session_a",
                session_id="session_a",
                sampling_session_seq_id=0,
                model_path="tinker://model_a/sampler_weights/weights_a",
            )
        )
        session.add(
            ModelDB(
                model_id="model_a",
                base_model="model_a",
                lora_config={},
                status="ready",
                request_id=0,
                session_id="session_a",
            )
        )
        session.add(
            CheckpointDB(
                model_id="model_a",
                checkpoint_id="weights_a",
                checkpoint_type=types.CheckpointType.SAMPLER,
                status=CheckpointStatus.COMPLETED,
            )
        )
        await session.commit()

    for wave in range(4):
        async def create_sample(index: int) -> None:
            async with AsyncSession(engine) as session:
                await api.asample(
                    api.SampleRequest(
                        prompt=api.ModelInput(
                            chunks=[api.EncodedTextChunk(tokens=[index])]
                        ),
                        sampling_params=api.SamplingParams(
                            temperature=0.0, max_tokens=1, seed=index
                        ),
                        sampling_session_id="session_a",
                        seq_id=wave * 512 + index,
                    ),
                    sample_request,
                    session,
                )

        async def create_training_future(index: int) -> None:
            async with AsyncSession(engine) as session:
                await api.forward_backward(
                    _forward_backward_request(wave * 512 + index, db_write_lock),
                    session,
                )

        async def heartbeat() -> None:
            async with AsyncSession(engine) as session:
                await api.session_heartbeat(
                    api.SessionHeartbeatRequest(session_id="session_a"),
                    sample_request,
                    session,
                )

        await asyncio.gather(
            *(create_sample(index) for index in range(512)),
            *(create_training_future(index) for index in range(512)),
            *(heartbeat() for _ in range(32)),
        )

    await store.flush()
    async with AsyncSession(engine) as session:
        persisted_by_type = dict(
            (
                await session.exec(
                    select(FutureDB.request_type, func.count())
                    .group_by(FutureDB.request_type)
                )
            ).all()
        )
        session_db = await session.get(SessionDB, "session_a")

    assert persisted_by_type[types.RequestType.EXTERNAL] == 2048
    assert persisted_by_type[types.RequestType.FORWARD_BACKWARD] == 2048
    assert session_db is not None
    assert session_db.heartbeat_count == 128
    assert sample_request.app.state.validated_sampler_checkpoints == {
        ("model_a", "weights_a")
    }


@pytest.mark.asyncio
async def test_forwarding_client_completes_in_memory_future(future_store, monkeypatch):
    store, engine, _ = future_store
    request_id = store.create("model_a", _sample_input(1))
    result = types.SampleOutput(
        sequences=[
            types.GeneratedSequence(
                stop_reason="stop", tokens=[1, 2], logprobs=[-0.5, -1.0]
            )
        ]
    )
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="model_a"), engine, store
    )

    async def forward(*args, **kwargs):
        return result

    monkeypatch.setattr(client, "_forward_with_retry", forward)
    try:
        await client.call_and_store_result(
            request_id,
            SimpleNamespace(),
            model_id="model_a",
            checkpoint_id="",
        )
        completed = await store.wait(request_id, timeout=1)
    finally:
        await client.aclose()

    assert completed == (
        RequestStatus.COMPLETED,
        types.RequestType.EXTERNAL,
        result.model_dump_json(),
    )


@pytest.mark.asyncio
async def test_retrieve_future_serializes_in_memory_result_as_proto(future_store):
    from tinker import SampleResponse
    from tinker.proto.response_conv import deserialize_proto_response

    store, engine, _ = future_store
    request_id = store.create("model_a", _sample_input(1))
    result = types.SampleOutput(
        sequences=[
            types.GeneratedSequence(
                stop_reason="stop", tokens=[1, 2], logprobs=[-0.5, -1.0]
            )
        ]
    )
    await store.complete(request_id, result, RequestStatus.COMPLETED)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_engine=engine, external_future_store=store, future_waiters={}
            )
        ),
        headers={"accept": "application/x-protobuf, application/json"},
    )
    response = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)), request
    )

    assert response.media_type == "application/x-protobuf"
    result = deserialize_proto_response(response.body, SampleResponse)
    assert result.sequences[0].tokens == [1, 2]
