import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    InferenceForwardingError,
    InferenceForwardingTimeoutError,
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
    forwarding_logger = Mock()
    monkeypatch.setattr(client, "_TRANSIENT_RETRY_INITIAL_DELAY_SEC", 0)
    monkeypatch.setattr("skyrl.tinker.extra.skyrl_train_inference_forwarding.logger", forwarding_logger)

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._forward.await_count == 2
    assert client._resolve_proxy_url.await_args_list[1].kwargs["force_refresh"] is True
    assert forwarding_logger.warning.call_args.args[1:3] == (1, client._TRANSIENT_503_MAX_ATTEMPTS)
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


@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ReadTimeout,
        httpx.WriteTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    ],
)
@pytest.mark.asyncio
async def test_forwarding_does_not_retry_ambiguous_transport_error(error_type) -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(side_effect=error_type("ambiguous transport failure"))

    with pytest.raises(error_type):
        await client._forward_with_retry(object(), "model", base_model=None)

    assert client._forward.await_count == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_retries_connect_error_once(monkeypatch) -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
    expected = types.SampleOutput(sequences=[])
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(side_effect=[httpx.ConnectError("connection refused"), expected])
    forwarding_logger = Mock()
    monkeypatch.setattr(client, "_TRANSIENT_RETRY_INITIAL_DELAY_SEC", 0)
    monkeypatch.setattr("skyrl.tinker.extra.skyrl_train_inference_forwarding.logger", forwarding_logger)

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._forward.await_count == 2
    assert forwarding_logger.warning.call_args.args[1:3] == (1, 2)
    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_deadline_clips_connect_retry_backoff(monkeypatch) -> None:
    timeout_sec = 0.1
    client = _create_client(EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=timeout_sec))
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    forwarding_logger = Mock()
    monkeypatch.setattr(client, "_TRANSIENT_RETRY_INITIAL_DELAY_SEC", 1.0)
    monkeypatch.setattr("skyrl.tinker.extra.skyrl_train_inference_forwarding.logger", forwarding_logger)

    with pytest.raises(InferenceForwardingTimeoutError, match="operation timeout"):
        await client._forward_with_retry(object(), "model", base_model=None)

    assert client._forward.await_count == 1
    assert forwarding_logger.warning.call_args.args[1:3] == (1, 2)
    assert 0 <= forwarding_logger.warning.call_args.args[3] <= timeout_sec
    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_deadline_clips_transient_503_backoff(monkeypatch) -> None:
    timeout_sec = 0.1
    client = _create_client(EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=timeout_sec))
    client._resolve_proxy_url = AsyncMock(return_value="http://inference-proxy")
    client._forward = AsyncMock(side_effect=InferenceForwardingError(503, "No available workers"))
    forwarding_logger = Mock()
    monkeypatch.setattr(client, "_TRANSIENT_RETRY_INITIAL_DELAY_SEC", 1.0)
    monkeypatch.setattr("skyrl.tinker.extra.skyrl_train_inference_forwarding.logger", forwarding_logger)
    started = asyncio.get_running_loop().time()

    with pytest.raises(InferenceForwardingTimeoutError, match="operation timeout"):
        await client._forward_with_retry(object(), "model", base_model=None)

    elapsed = asyncio.get_running_loop().time() - started
    assert client._forward.await_count == 1
    assert 0 <= forwarding_logger.warning.call_args.args[3] <= timeout_sec
    assert elapsed < 0.5
    await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_failure_persists_nonblank_typed_error() -> None:
    client = _create_client(EngineConfig(base_model="test-model"))
    client._forward_with_retry = AsyncMock(side_effect=httpx.ReadTimeout(""))
    sample_request = SimpleNamespace(
        prompt=SimpleNamespace(chunks=[]),
        sampling_params=SimpleNamespace(max_tokens=1),
        sampling_session_id="sampling-session",
        seq_id=1,
    )

    await client.call_and_store_result(7, sample_request, "model", "checkpoint")

    result = client.external_future_store.complete.await_args.args[1]
    assert result.error == "ReadTimeout: inference forwarding failed"
    assert result.error.strip()
    await client.aclose()
