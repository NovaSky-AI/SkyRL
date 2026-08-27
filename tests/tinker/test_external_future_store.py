import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    FutureDB,
    RequestStatus,
    SamplingSessionDB,
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
        self.gate = asyncio.Event()

    async def call_and_store_result(
        self,
        request_id: int,
        sample_req,
        model_id: str,
        checkpoint_id: str,
        base_model: str | None = None,
    ) -> None:
        await self.gate.wait()
        await self.store.complete(
            request_id,
            types.SampleOutput(sequences=[]),
            RequestStatus.COMPLETED,
        )


@pytest_asyncio.fixture()
async def future_store(tmp_path):
    db_url = get_async_database_url(f"sqlite:///{tmp_path / 'tinker.db'}")
    engine = create_async_engine(db_url, pool_size=1, max_overflow=0)
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    store = ExternalFutureStore(engine)
    await store.start()
    yield store, engine
    await store.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_two_full_rollout_waves_complete_and_persist(future_store):
    store, engine = future_store
    forwarder = _CompletingForwarder(store)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_future_store=store,
                external_inference_client=forwarder,
            )
        )
    )
    result_data = types.SampleOutput(sequences=[]).model_dump_json()

    async with AsyncSession(engine) as session:
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
        forwarder.gate.clear()

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

        request_ids = await asyncio.gather(
            *(create_sample(index) for index in range(512))
        )
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

        forwarder.gate.set()
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

    assert persisted == 1024
    assert pending == 0


@pytest.mark.asyncio
async def test_forwarding_client_completes_in_memory_future(future_store, monkeypatch):
    store, engine = future_store
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

    store, engine = future_store
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
