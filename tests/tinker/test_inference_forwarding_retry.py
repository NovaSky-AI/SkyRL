from unittest.mock import AsyncMock, call, patch

import httpx
import pytest

from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_forwarding_retries_unavailable_worker() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(
        side_effect=[RuntimeError("vLLM /v1/completions returned 503: No available workers"), expected]
    )

    with patch("skyrl.tinker.extra.skyrl_train_inference_forwarding.asyncio.sleep", new=AsyncMock()):
        result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_has_awaits([call(), call(force_refresh=True)])
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_read_timeout() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=httpx.ReadTimeout("slow response"))

    with pytest.raises(httpx.ReadTimeout):
        await client._forward_with_retry(object(), "model", base_model=None)

    client._resolve_proxy_url.assert_awaited_once_with()
    client._forward.assert_awaited_once()
