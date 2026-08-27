import pytest

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_forwarding_timeout_uses_engine_config() -> None:
    config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=42.0)
    client = SkyRLTrainInferenceForwardingClient(config, db_engine=None)

    assert client._http_client.timeout.read == 42.0

    await client.aclose()


@pytest.mark.asyncio
async def test_proxy_resolution_waits_for_engine_readiness(monkeypatch) -> None:
    client = SkyRLTrainInferenceForwardingClient(EngineConfig(base_model="test-model"), db_engine=None)
    proxy_urls = iter([None, "http://inference-proxy"])

    async def read_proxy_url():
        return next(proxy_urls)

    monkeypatch.setattr(client, "_read_proxy_url_from_db", read_proxy_url)
    monkeypatch.setattr(client, "_PROXY_URL_POLL_INTERVAL_SEC", 0)

    assert await client._resolve_proxy_url() == "http://inference-proxy"

    await client.aclose()


@pytest.mark.asyncio
async def test_proxy_resolution_fails_after_forwarding_timeout(monkeypatch) -> None:
    config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=0.001)
    client = SkyRLTrainInferenceForwardingClient(config, db_engine=None)

    async def read_proxy_url():
        return None

    monkeypatch.setattr(client, "_read_proxy_url_from_db", read_proxy_url)

    with pytest.raises(RuntimeError, match="timed out waiting for a proxy URL"):
        await client._resolve_proxy_url()

    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_stores_external_result_without_database(monkeypatch) -> None:
    completed = []

    class FutureStore:
        def complete(self, request_id, status, result_data):
            completed.append((request_id, status, result_data))
            return True

    result = types.SampleOutput(sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1], logprobs=[-0.5])])
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="test-model"),
        db_engine=None,
        future_store=FutureStore(),
    )

    async def forward(*args, **kwargs):
        return result

    monkeypatch.setattr(client, "_forward_with_retry", forward)

    await client.call_and_store_result(123, object(), "model-a", "checkpoint-a")

    assert completed == [(123, RequestStatus.COMPLETED, result.model_dump_json())]
    await client.aclose()
