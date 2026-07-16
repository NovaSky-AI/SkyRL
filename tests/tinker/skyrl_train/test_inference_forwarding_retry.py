from unittest.mock import AsyncMock

import httpx
import pytest

from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


def forwarding_client() -> SkyRLTrainInferenceForwardingClient:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = None
    return client


@pytest.mark.asyncio
async def test_missing_proxy_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYRL_PROXY_RETRY_TIMEOUT_SEC", "1")
    monkeypatch.setenv("SKYRL_PROXY_RETRY_BACKOFF_SEC", "0")
    client = forwarding_client()
    client._resolve_proxy_url = AsyncMock(
        side_effect=[RuntimeError("inference engine not ready: no proxy URL published"), "http://proxy"]
    )
    expected = object()
    client._forward = AsyncMock(return_value=expected)

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._resolve_proxy_url.await_count == 2
    client._resolve_proxy_url.assert_awaited_with(force_refresh=True)


@pytest.mark.asyncio
async def test_proxy_network_error_refreshes_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYRL_PROXY_RETRY_TIMEOUT_SEC", "1")
    monkeypatch.setenv("SKYRL_PROXY_RETRY_BACKOFF_SEC", "0")
    client = forwarding_client()
    client._cached_proxy_url = "http://stale-proxy"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://stale-proxy", "http://fresh-proxy"])
    expected = object()
    client._forward = AsyncMock(side_effect=[httpx.ConnectError("connection lost"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_awaited_with(force_refresh=True)


@pytest.mark.asyncio
async def test_non_transient_runtime_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYRL_PROXY_RETRY_TIMEOUT_SEC", "1")
    monkeypatch.setenv("SKYRL_PROXY_RETRY_BACKOFF_SEC", "0")
    client = forwarding_client()
    client._resolve_proxy_url = AsyncMock(return_value="http://proxy")
    client._forward = AsyncMock(side_effect=RuntimeError("vLLM returned 400"))

    with pytest.raises(RuntimeError, match="vLLM returned 400"):
        await client._forward_with_retry(object(), "model", base_model=None)

    client._forward.assert_awaited_once()
