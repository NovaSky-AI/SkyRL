import asyncio
from unittest.mock import AsyncMock

import pytest

from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_proxy_url_refreshes_after_inference_engine_restart():
    client = SkyRLTrainInferenceForwardingClient.__new__(SkyRLTrainInferenceForwardingClient)
    client._cached_proxy_url = None
    client._cache_lock = asyncio.Lock()
    client._read_proxy_url_from_db = AsyncMock(side_effect=["http://old-router", "http://new-router"])

    assert await client._resolve_proxy_url() == "http://old-router"
    assert await client._resolve_proxy_url() == "http://new-router"
