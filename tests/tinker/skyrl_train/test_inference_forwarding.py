from unittest.mock import AsyncMock

import pytest

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    InferenceForwardingError,
    SkyRLTrainInferenceForwardingClient,
)


def _create_client(config: EngineConfig) -> SkyRLTrainInferenceForwardingClient:
    return SkyRLTrainInferenceForwardingClient(config, db_engine=None, external_future_store=AsyncMock())


@pytest.mark.asyncio
async def test_forwarding_timeout_uses_engine_config() -> None:
    config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=42.0)
    client = _create_client(config)

    assert client._http_client.timeout.read == 42.0

    await client.aclose()


@pytest.mark.asyncio
async def test_proxy_resolution_waits_for_engine_readiness(monkeypatch) -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
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
    client = _create_client(config)

    async def read_proxy_url():
        return None

    monkeypatch.setattr(client, "_read_proxy_url_from_db", read_proxy_url)

    with pytest.raises(RuntimeError, match="timed out waiting for a proxy URL"):
        await client._resolve_proxy_url()

    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_retries_transient_no_worker_503(monkeypatch) -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
    expected = types.SampleOutput(sequences=[])
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(
        side_effect=[
            InferenceForwardingError(503, "No available workers"),
            expected,
        ]
    )
    monkeypatch.setattr(client, "_TRANSIENT_RETRY_INITIAL_DELAY_SEC", 0)

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._forward.await_count == 2
    assert client._resolve_proxy_url.await_args_list[1].kwargs == {"force_refresh": True}
    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_other_http_errors() -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(side_effect=InferenceForwardingError(500, "internal error"))

    with pytest.raises(InferenceForwardingError, match="returned 500"):
        await client._forward_with_retry(object(), "model", base_model=None)

    assert client._forward.await_count == 1
    await client.aclose()
