import argparse
from unittest.mock import AsyncMock, call, patch

import httpx
import pytest

from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


def test_forwarding_timeout_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", "1800")
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(["--base-model", "test-model"])
    config = EngineConfig.model_validate(vars(args))

    assert config.forwarding_inference_timeout_sec == 1800.0


def test_forwarding_client_uses_configured_timeout() -> None:
    config = EngineConfig(
        base_model="test-model",
        forwarding_inference_timeout_sec=1800.0,
    )

    with patch("skyrl.tinker.extra.skyrl_train_inference_forwarding.httpx.AsyncClient") as async_client:
        SkyRLTrainInferenceForwardingClient(config, db_engine=None)

    timeout = async_client.call_args.kwargs["timeout"]
    assert timeout.connect == 10.0
    assert timeout.read == 1800.0
    assert timeout.write == 300.0
    assert timeout.pool == 300.0


@pytest.mark.asyncio
async def test_forwarding_retries_connection_failure() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[httpx.ConnectError("unreachable"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_has_awaits([call(), call(force_refresh=True)])
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_read_timeout() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=123.0)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=httpx.ReadTimeout("slow response"))

    with pytest.raises(RuntimeError) as exc_info:
        await client._forward_with_retry(object(), "model", base_model=None)

    message = str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
    assert "http://inference" in message
    assert "timed out after 123s" in message
    client._resolve_proxy_url.assert_awaited_once_with()
    client._forward.assert_awaited_once()
