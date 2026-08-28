import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession
from starlette.requests import Request

from skyrl.tinker import api, external_future_store, types
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
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=1, seed=seq_id),
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
    body = (
        api.ForwardBackwardRequest(
            model_id="model_a",
            seq_id=seq_id,
            forward_backward_input=api.ForwardBackwardInput(
                data=[
                    api.Datum(
                        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
                        loss_fn_inputs={
                            "target_tokens": api.TensorData(data=[2, 3]),
                            "weights": api.TensorData(data=[1.0, 1.0]),
                        },
                    )
                ],
                loss_fn="cross_entropy",
            ),
        )
        .model_dump_json()
        .encode()
    )
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
    engine = create_async_engine(db_url, pool_size=5, max_overflow=10, pool_timeout=0.1)
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
async def test_sustained_model_path_rollouts_training_futures_and_heartbeats(future_store):
    store, engine, db_write_lock = future_store
    forwarder = _CompletingForwarder(store)
    sample_request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_engine=engine,
                external_future_store=store,
                external_inference_client=forwarder,
                forwarding_tasks=set(),
                future_waiters={},
                engine_config=EngineConfig(base_model="model_a"),
                db_write_lock=db_write_lock,
                sampling_model_cache={},
                sampling_model_cache_lock=asyncio.Lock(),
                validated_sampler_checkpoints=set(),
                sampler_checkpoint_validation_lock=asyncio.Lock(),
            )
        ),
        headers={},
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

    future_poller = asyncio.create_task(
        api.poll_futures(engine, sample_request.app.state.future_waiters, poll_interval_sec=0.001)
    )
    expected_sample = types.SampleOutput(sequences=[]).model_dump_json().encode()
    try:
        request_ids = []
        for wave in range(4):

            async def create_sample(index: int) -> int:
                async with AsyncSession(engine) as session:
                    response = await api.asample(
                        api.SampleRequest(
                            prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[index])]),
                            sampling_params=api.SamplingParams(temperature=0.0, max_tokens=1, seed=index),
                            sampling_session_id="session_a",
                            seq_id=wave * 512 + index,
                        ),
                        sample_request,
                        session,
                    )
                return int(response.request_id)

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

            responses = await asyncio.gather(
                *(create_sample(index) for index in range(512)),
                *(create_training_future(index) for index in range(512)),
                *(heartbeat() for _ in range(32)),
            )
            request_ids = responses[:512]
            retrievals = await asyncio.gather(
                *(
                    api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), sample_request)
                    for request_id in request_ids
                )
            )
            assert all(response.body == expected_sample for response in retrievals)
            assert not store._entries

        repeated = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_ids[-1])), sample_request)
        assert repeated.body == expected_sample
    finally:
        future_poller.cancel()
        await asyncio.gather(future_poller, return_exceptions=True)

    await store.flush()
    async with AsyncSession(engine) as session:
        persisted_by_type = dict(
            (await session.exec(select(FutureDB.request_type, func.count()).group_by(FutureDB.request_type))).all()
        )
        session_db = await session.get(SessionDB, "session_a")

    assert persisted_by_type[types.RequestType.EXTERNAL] == 2048
    assert persisted_by_type[types.RequestType.FORWARD_BACKWARD] == 2048
    assert session_db is not None
    assert session_db.heartbeat_count == 128
    assert sample_request.app.state.validated_sampler_checkpoints == {("model_a", "weights_a")}
    assert not sample_request.app.state.forwarding_tasks


@pytest.mark.asyncio
async def test_terminal_futures_are_persisted_in_batches(future_store, monkeypatch):
    store, _, _ = future_store
    first_persistence_started = asyncio.Event()
    release_first_persistence = asyncio.Event()
    persisted_batch_sizes = []
    persist = store._persist

    async def record_batch(entries) -> None:
        persisted_batch_sizes.append(len(entries))
        if len(persisted_batch_sizes) == 1:
            first_persistence_started.set()
            await release_first_persistence.wait()
        await persist(entries)

    monkeypatch.setattr(store, "_persist", record_batch)
    result = types.SampleOutput(sequences=[])
    first_request_id = store.create("model_a", _sample_input(0))
    await store.complete(first_request_id, result, RequestStatus.COMPLETED)
    await first_persistence_started.wait()

    request_ids = [store.create("model_a", _sample_input(index)) for index in range(1, 65)]
    await asyncio.gather(*(store.complete(request_id, result, RequestStatus.COMPLETED) for request_id in request_ids))
    release_first_persistence.set()
    await store.flush()

    assert persisted_batch_sizes == [1, 64]


@pytest.mark.asyncio
async def test_shutdown_waits_for_forwarding_tasks_before_closing_store():
    release_forwarding = asyncio.Event()
    events = []

    class ClosingClient:
        async def aclose(self) -> None:
            events.append("client_closed")

    class ClosingStore:
        async def close(self) -> None:
            events.append("store_closed")

    app = SimpleNamespace(
        state=SimpleNamespace(
            external_inference_client=ClosingClient(),
            external_future_store=ClosingStore(),
            forwarding_tasks=set(),
        )
    )

    async def finish_forwarding() -> None:
        await release_forwarding.wait()
        events.append("future_completed")

    api._start_forwarding_task(app, finish_forwarding())
    shutdown = asyncio.create_task(api._close_external_inference(app))
    await asyncio.sleep(0)
    assert not shutdown.done()

    release_forwarding.set()
    await shutdown

    assert events == ["future_completed", "client_closed", "store_closed"]
    assert not app.state.forwarding_tasks


@pytest.mark.asyncio
async def test_shutdown_stops_engine_when_future_persistence_failed(monkeypatch):
    events = []

    class BackgroundEngine:
        pid = 123

        def terminate(self) -> None:
            events.append("engine_terminated")

        async def wait(self) -> int:
            events.append("engine_waited")
            return 0

    async def fail_external_close(_app) -> None:
        events.append("external_close_failed")
        raise RuntimeError("persistence failed")

    monkeypatch.setattr(api, "_close_external_inference", fail_external_close)

    with pytest.raises(RuntimeError, match="persistence failed"):
        await api._close_runtime(SimpleNamespace(), BackgroundEngine())

    assert events == ["external_close_failed", "engine_terminated", "engine_waited"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("dialect", "serializes"), [("sqlite", True), ("postgresql", False)])
async def test_db_write_context_serializes_only_sqlite(dialect, serializes):
    context = api._get_db_write_context(SimpleNamespace(dialect=SimpleNamespace(name=dialect)))
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first_writer() -> None:
        async with context:
            first_entered.set()
            await release_first.wait()

    async def second_writer() -> None:
        await first_entered.wait()
        async with context:
            second_entered.set()

    first = asyncio.create_task(first_writer())
    second = asyncio.create_task(second_writer())
    await first_entered.wait()
    await asyncio.sleep(0)
    assert second_entered.is_set() is not serializes

    release_first.set()
    await asyncio.gather(first, second)
    assert second_entered.is_set()


@pytest.mark.asyncio
async def test_sampler_checkpoint_delete_waits_for_validation_and_invalidates_cache(future_store, monkeypatch):
    _, engine, _ = future_store
    validation_started = asyncio.Event()
    release_validation = asyncio.Event()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_config=EngineConfig(base_model="model_a"),
                sampler_checkpoint_validation_lock=asyncio.Lock(),
                validated_sampler_checkpoints=set(),
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

    async def hold_validation(*args) -> None:
        validation_started.set()
        await release_validation.wait()

    monkeypatch.setattr(api, "validate_checkpoint", hold_validation)
    async with AsyncSession(engine) as validation_session, AsyncSession(engine) as deletion_session:
        validation = asyncio.create_task(
            api.validate_sampler_checkpoint_once(
                request,
                "model_a",
                "weights_a",
                validation_session,
            )
        )
        await validation_started.wait()
        deletion = asyncio.create_task(
            api.delete_checkpoint(
                request,
                "model_a",
                "weights_a",
                types.CheckpointType.SAMPLER,
                deletion_session,
            )
        )
        await asyncio.sleep(0)
        assert not deletion.done()

        release_validation.set()
        await asyncio.gather(validation, deletion)

    assert not request.app.state.validated_sampler_checkpoints
    async with AsyncSession(engine) as session:
        assert (
            await session.get(
                CheckpointDB,
                ("model_a", "weights_a", types.CheckpointType.SAMPLER),
            )
            is None
        )


@pytest.mark.asyncio
async def test_forwarding_client_completes_in_memory_future(future_store, monkeypatch):
    store, engine, _ = future_store
    request_id = store.create("model_a", _sample_input(1))
    result = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1, 2], logprobs=[-0.5, -1.0])]
    )
    client = SkyRLTrainInferenceForwardingClient(EngineConfig(base_model="model_a"), engine, store)

    async def forward(*args, **kwargs):
        return result

    monkeypatch.setattr(client, "_forward_with_retry", forward)
    sample_request = SimpleNamespace(
        prompt=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1])]),
        sampling_params=SimpleNamespace(max_tokens=1),
        sampling_session_id="sampling_a",
        seq_id=1,
    )
    try:
        await client.call_and_store_result(
            request_id,
            sample_request,
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
async def test_persistence_failure_is_reported_without_logging_values(future_store, monkeypatch):
    store, _, _ = future_store
    request_id = store.create("model_a", _sample_input(1))
    logger = Mock()

    async def fail_persistence(entries):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "_persist", fail_persistence)
    monkeypatch.setattr(external_future_store, "logger", logger)
    await store.complete(request_id, types.SampleOutput(sequences=[]), RequestStatus.COMPLETED)

    with pytest.raises(RuntimeError, match=f"Failed to persist external future {request_id}"):
        await store.wait(request_id, timeout=1)
    with pytest.raises(RuntimeError, match="External future persistence failed"):
        await store.flush()
    assert logger.error.call_count == 1
    assert logger.error.call_args.args[5] == "RuntimeError"
    assert "database unavailable" not in str(logger.error.call_args)


@pytest.mark.asyncio
async def test_close_stops_worker_after_persistence_failure(future_store, monkeypatch):
    store, _, _ = future_store
    request_id = store.create("model_a", _sample_input(1))

    async def fail_persistence(entries):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(store, "_persist", fail_persistence)
    await store.complete(request_id, types.SampleOutput(sequences=[]), RequestStatus.COMPLETED)
    worker = store._persist_worker
    assert worker is not None

    with pytest.raises(RuntimeError, match="External future persistence failed"):
        await store.close()

    assert worker.done()


@pytest.mark.asyncio
async def test_retrieve_future_serializes_in_memory_result_as_proto(future_store):
    from tinker import SampleResponse
    from tinker.proto.response_conv import deserialize_proto_response

    store, engine, _ = future_store
    request_id = store.create("model_a", _sample_input(1))
    result = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1, 2], logprobs=[-0.5, -1.0])]
    )
    await store.complete(request_id, result, RequestStatus.COMPLETED)

    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(db_engine=engine, external_future_store=store, future_waiters={})),
        headers={"accept": "application/x-protobuf, application/json"},
    )
    response = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), request)

    assert response.media_type == "application/x-protobuf"
    result = deserialize_proto_response(response.body, SampleResponse)
    assert result.sequences[0].tokens == [1, 2]
