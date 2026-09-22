import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import aiohttp
import pytest

from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
    TransientInferenceError,
)


def test_forwarding_timeout_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", "1800")
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(["--base-model", "test-model"])
    config = EngineConfig.model_validate(vars(args))

    assert config.forwarding_inference_timeout_sec == 1800.0


@pytest.mark.asyncio
async def test_forwarding_client_uses_configured_timeout_and_connection_limit() -> None:
    config = EngineConfig(
        base_model="test-model",
        forwarding_inference_timeout_sec=1800.0,
        forwarding_inference_max_connections=64,
    )
    client = SkyRLTrainInferenceForwardingClient(config, db_engine=None)
    try:
        session = client._get_session()
        assert session.timeout.sock_connect == 60.0
        assert session.timeout.sock_read == 1800.0
        # No overall deadline: a request may wait in the connector queue for
        # as long as the engine takes to get to it.
        assert session.timeout.total is None
        assert session.connector.limit == 64
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_client_default_connection_limit_is_unlimited() -> None:
    client = SkyRLTrainInferenceForwardingClient(EngineConfig(base_model="test-model"), db_engine=None)
    try:
        assert client._get_session().connector.limit == 0
    finally:
        await client.aclose()


def _connect_error(message: str) -> aiohttp.ClientConnectorError:
    return aiohttp.ClientConnectorError(SimpleNamespace(ssl=None, host="inference", port=8000), OSError(message))


@pytest.mark.asyncio
async def test_forwarding_retries_connection_failure() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[_connect_error("unreachable"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_has_awaits([call(), call(force_refresh=True)])
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_retries_transient_5xx_once() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[TransientInferenceError("503 from router"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_4xx() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=RuntimeError("vLLM /v1/completions returned 400: bad request"))

    with pytest.raises(RuntimeError, match="returned 400"):
        await client._forward_with_retry(object(), "model", base_model=None)

    client._forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_read_timeout() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=123.0)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=aiohttp.SocketTimeoutError("slow response"))

    with pytest.raises(RuntimeError) as exc_info:
        await client._forward_with_retry(object(), "model", base_model=None)

    message = str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, aiohttp.SocketTimeoutError)
    assert "http://inference" in message
    assert "timed out after 123s" in message
    client._resolve_proxy_url.assert_awaited_once_with()
    client._forward.assert_awaited_once()
