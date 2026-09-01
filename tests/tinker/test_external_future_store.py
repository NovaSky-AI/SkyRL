import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
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
    store = ExternalFutureStore()
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
            assert all(store._entries[request_id].retrieved_at is not None for request_id in request_ids)

        repeated = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_ids[-1])), sample_request)
        assert repeated.body == expected_sample
    finally:
        future_poller.cancel()
        await asyncio.gather(future_poller, return_exceptions=True)

    async with AsyncSession(engine) as session:
        persisted_by_type = dict(
            (await session.exec(select(FutureDB.request_type, func.count()).group_by(FutureDB.request_type))).all()
        )
        session_db = await session.get(SessionDB, "session_a")

    # External sample futures live purely in memory — nothing reaches the DB.
    assert types.RequestType.EXTERNAL not in persisted_by_type
    assert persisted_by_type[types.RequestType.FORWARD_BACKWARD] == 2048
    assert session_db is not None
    assert session_db.heartbeat_count == 128
    assert sample_request.app.state.validated_sampler_checkpoints == {("model_a", "weights_a")}
    assert not store._forwarding_tasks


@pytest.mark.asyncio
async def test_retrieve_future_bounds_protobuf_serialization_off_event_loop(monkeypatch):
    event_loop_thread_id = threading.get_ident()
    serialization_thread_ids = []
    serialization_state_lock = threading.Lock()
    active_serializations = 0
    max_active_serializations = 0

    class CompletedStore:
        async def wait(self, request_id, timeout):
            return (
                RequestStatus.COMPLETED,
                types.RequestType.EXTERNAL,
                types.SampleOutput(sequences=[]).model_dump_json(),
            )

    def serialize_result_in_thread(request_type, result_data):
        nonlocal active_serializations, max_active_serializations
        serialization_thread_ids.append(threading.get_ident())
        with serialization_state_lock:
            active_serializations += 1
            max_active_serializations = max(max_active_serializations, active_serializations)
        try:
            time.sleep(0.01)
            return b"serialized"
        finally:
            with serialization_state_lock:
                active_serializations -= 1

    monkeypatch.setattr(api, "serialize_result", serialize_result_in_thread)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_future_store=CompletedStore(),
                future_waiters={},
                proto_serialization_lock=asyncio.Lock(),
            )
        ),
        headers={"accept": api.PROTO_CONTENT_TYPE},
    )

    responses = await asyncio.gather(
        *(
            api.retrieve_future(api.RetrieveFutureRequest(request_id=str(-request_id)), request)
            for request_id in range(1, 9)
        )
    )

    assert all(response.body == b"serialized" for response in responses)
    assert len(serialization_thread_ids) == len(responses)
    assert all(thread_id != event_loop_thread_id for thread_id in serialization_thread_ids)
    assert max_active_serializations == 1


@pytest.mark.asyncio
async def test_shutdown_waits_for_forwarding_tasks_before_closing_client():
    release_forwarding = asyncio.Event()
    events = []

    class ClosingClient:
        async def aclose(self) -> None:
            events.append("client_closed")

    store = ExternalFutureStore()
    await store.start()
    app = SimpleNamespace(
        state=SimpleNamespace(
            external_inference_client=ClosingClient(),
            external_future_store=store,
        )
    )

    async def finish_forwarding() -> None:
        await release_forwarding.wait()
        events.append("future_completed")

    store.spawn_forwarding_task(finish_forwarding())
    shutdown = asyncio.create_task(api._close_external_inference(app))
    await asyncio.sleep(0)
    assert not shutdown.done()

    release_forwarding.set()
    await shutdown

    assert events == ["future_completed", "client_closed"]
    assert not store._forwarding_tasks


@pytest.mark.asyncio
async def test_shutdown_cancels_hung_forwarding_tasks(monkeypatch):
    monkeypatch.setattr(ExternalFutureStore, "_FORWARDING_SHUTDOWN_TIMEOUT_SECONDS", 0.05)
    events = []

    store = ExternalFutureStore()
    await store.start()
    app = SimpleNamespace(
        state=SimpleNamespace(
            external_inference_client=None,
            external_future_store=store,
        )
    )

    async def hang_forever() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            events.append("forwarding_cancelled")
            raise

    store.spawn_forwarding_task(hang_forever())
    await asyncio.wait_for(api._close_external_inference(app), timeout=5)

    assert events == ["forwarding_cancelled"]
    assert not store._forwarding_tasks


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
async def test_sweep_evicts_entries_by_ttl(future_store):
    store, _, _ = future_store
    result = types.SampleOutput(sequences=[])

    retrieved_id = store.create("model_a", _sample_input(1))
    await store.complete(retrieved_id, result, RequestStatus.COMPLETED)
    await store.wait(retrieved_id, timeout=1)
    completed_id = store.create("model_a", _sample_input(2))
    await store.complete(completed_id, result, RequestStatus.COMPLETED)
    pending_id = store.create("model_a", _sample_input(3))

    now = datetime.now(timezone.utc)
    store._sweep(now)
    assert set(store._entries) == {retrieved_id, completed_id, pending_id}

    store._sweep(now + timedelta(seconds=ExternalFutureStore._RETRIEVED_TTL_SECONDS + 1))
    assert set(store._entries) == {completed_id, pending_id}

    store._sweep(now + timedelta(seconds=ExternalFutureStore._COMPLETED_TTL_SECONDS + 1))
    assert set(store._entries) == {pending_id}

    store._sweep(now + timedelta(seconds=ExternalFutureStore._PENDING_TTL_SECONDS + 1))
    assert not store._entries

    # A forwarding task finishing after its entry was swept is dropped, not an error.
    await store.complete(pending_id, result, RequestStatus.COMPLETED)


@pytest.mark.asyncio
async def test_request_ids_do_not_repeat_across_restarts(monkeypatch):
    import skyrl.tinker.external_future_store as store_module

    monkeypatch.setattr(store_module.time, "time", lambda: 1_000.0)
    first_boot = ExternalFutureStore()
    first_ids = [first_boot.create("model_a", _sample_input(index)) for index in range(3)]

    monkeypatch.setattr(store_module.time, "time", lambda: 1_000.001)
    second_boot = ExternalFutureStore()
    second_ids = [second_boot.create("model_a", _sample_input(index)) for index in range(3)]

    assert all(request_id < 0 for request_id in first_ids + second_ids)
    # A later boot starts below every id the earlier process handed out, so a
    # client polling a pre-restart id can never receive another request's result.
    assert max(second_ids) < min(first_ids)


@pytest.mark.asyncio
async def test_create_future_accepts_request_id_zero_after_negative_rows(future_store):
    # SQLite assigns max(rowid)+1, so when the table holds only negative
    # request_ids (written by a pre-memory-only server version) the first
    # autoincremented FutureDB row gets id 0.
    _, engine, _ = future_store
    async with AsyncSession(engine) as session:
        session.add(
            FutureDB(
                request_id=-1,
                request_type=types.RequestType.EXTERNAL,
                request_data={},
                status=RequestStatus.COMPLETED,
            )
        )
        await session.commit()

    async with AsyncSession(engine) as session:
        created_id = await api.create_future(
            session,
            types.RequestType.SAMPLE,
            model_id=None,
            request_data=_sample_input(2),
        )
        await session.commit()
    assert created_id == 0


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
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_engine=engine,
                external_future_store=store,
                future_waiters={},
                proto_serialization_lock=asyncio.Lock(),
            )
        ),
        headers={"accept": "application/x-protobuf, application/json"},
    )
    response = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), request)

    assert response.media_type == "application/x-protobuf"
    result = deserialize_proto_response(response.body, SampleResponse)
    assert result.sequences[0].tokens == [1, 2]
