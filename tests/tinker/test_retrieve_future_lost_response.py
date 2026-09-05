"""A result whose delivery the client never received must survive for the SDK's retry.

Reproduces the 128x128 failure Chuck hit against PR j316chuck/SkyRL#18 (same
code as main): the SDK polls ``retrieve_future`` with a 45s client timeout and
gives up; the result lands afterwards; the abandoned handler still builds a
response and starts the short *retrieved* TTL clock even though nobody got the
bytes; the sweeper evicts the entry; the SDK's retry of the same request_id
gets ``404 Future not found``, which the SDK treats as fatal.

The server runs under a real uvicorn socket so the abandoned poll is a genuine
TCP disconnect, exactly as with the SDK. TTLs are shortened so the whole chain
takes a few seconds.
"""

import asyncio
import sys
from contextlib import suppress
from types import SimpleNamespace

import aiohttp
import pytest
import pytest_asyncio
import uvicorn
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

from skyrl.tinker import api, types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    RequestStatus,
    enable_sqlite_wal,
    get_async_database_url,
)
from skyrl.tinker.external_future_store import ExternalFutureStore

BASE_MODEL = "test-model"
RETRIEVED_TTL_SECONDS = 1.0
SWEEP_INTERVAL_SECONDS = 0.2


class _GatedForwarder:
    """Completes each forwarded sample only once the test releases it."""

    def __init__(self, store: ExternalFutureStore):
        self.store = store
        self.release = asyncio.Event()

    async def call_and_store_result(self, request_id, sample_req, model_id, checkpoint_id, *, base_model=None):
        await self.release.wait()
        result = types.SampleOutput(
            sequences=[types.GeneratedSequence(stop_reason="length", tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])]
        )
        await self.store.complete(request_id, result, RequestStatus.COMPLETED)


@pytest_asyncio.fixture()
async def served_app(tmp_path, monkeypatch):
    """The real API app on a real uvicorn socket, with app.state wired the way the lifespan does."""
    monkeypatch.setattr(ExternalFutureStore, "_RETRIEVED_TTL_SECONDS", RETRIEVED_TTL_SECONDS)
    monkeypatch.setattr(ExternalFutureStore, "_SWEEP_INTERVAL_SECONDS", SWEEP_INTERVAL_SECONDS)

    engine = create_async_engine(get_async_database_url(f"sqlite:///{tmp_path / 'tinker.db'}"))
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)

    store = ExternalFutureStore()
    await store.start()
    forwarder = _GatedForwarder(store)

    state = api.app.state
    state.engine_config = EngineConfig(base_model=BASE_MODEL)
    state.db_engine = engine
    state.future_waiters = {}
    state.future_poller = asyncio.create_task(api.poll_futures(engine, state.future_waiters, poll_interval_sec=0.01))
    state.proto_serialization_lock = asyncio.Lock()
    state.db_write_lock = asyncio.Lock()
    state.sampling_model_cache = {}
    state.sampling_model_cache_lock = asyncio.Lock()
    state.validated_sampler_checkpoints = set()
    state.sampler_checkpoint_validation_lock = asyncio.Lock()
    state.external_future_store = store
    state.external_inference_client = forwarder

    config = uvicorn.Config(api.app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]

    yield SimpleNamespace(url=f"http://127.0.0.1:{port}/api/v1", store=store, forwarder=forwarder)

    server.should_exit = True
    await serve_task
    state.future_poller.cancel()
    with suppress(asyncio.CancelledError):
        await state.future_poller
    await store.close()
    await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="relies on uvicorn disconnect handling over a real socket")
async def test_retry_after_client_abandoned_poll_is_served(served_app):
    payload = {
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "sampling_params": {"max_tokens": 3, "temperature": 1.0, "seed": 0},
        "base_model": BASE_MODEL,
    }
    async with aiohttp.ClientSession() as client:
        async with client.post(f"{served_app.url}/asample", json=payload) as resp:
            assert resp.status == 200
            request_id = (await resp.json())["request_id"]

        # The SDK's retrieve_future poll times out client-side (45s in the SDK)
        # while the result is still pending, and the connection is closed.
        with pytest.raises(asyncio.TimeoutError):
            await client.post(
                f"{served_app.url}/retrieve_future",
                json={"request_id": request_id},
                timeout=aiohttp.ClientTimeout(total=0.3),
            )
        await asyncio.sleep(0.2)  # let the server observe the disconnect

        # The result arrives after the client gave up. The abandoned handler
        # wakes, builds a response nobody will receive, and must NOT start the
        # short retrieved-TTL clock.
        served_app.forwarder.release.set()
        await asyncio.sleep(RETRIEVED_TTL_SECONDS + 3 * SWEEP_INTERVAL_SECONDS)

        # The SDK retries the same request_id once its backoff elapses.
        async with client.post(f"{served_app.url}/retrieve_future", json={"request_id": request_id}) as resp:
            body = await resp.text()
            assert resp.status == 200, f"retry of an undelivered result got {resp.status}: {body}"
            assert types.SampleOutput.model_validate_json(body).sequences[0].tokens == [1, 2, 3]


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="relies on uvicorn disconnect handling over a real socket")
async def test_delivered_result_still_expires_on_retrieved_ttl(served_app):
    """A result the client actually received is reclaimed on the short clock as before."""
    payload = {
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
        "sampling_params": {"max_tokens": 3, "temperature": 1.0, "seed": 0},
        "base_model": BASE_MODEL,
    }
    served_app.forwarder.release.set()
    async with aiohttp.ClientSession() as client:
        async with client.post(f"{served_app.url}/asample", json=payload) as resp:
            request_id = (await resp.json())["request_id"]
        async with client.post(f"{served_app.url}/retrieve_future", json={"request_id": request_id}) as resp:
            assert resp.status == 200
            await resp.read()
        assert int(request_id) in served_app.store._entries
        await asyncio.sleep(RETRIEVED_TTL_SECONDS + 3 * SWEEP_INTERVAL_SECONDS)
        assert int(request_id) not in served_app.store._entries
