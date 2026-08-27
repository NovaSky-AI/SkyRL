import pytest

from skyrl.tinker.config import EngineConfig
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_forwarding_timeout_uses_engine_config() -> None:
    config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=42.0)
    client = SkyRLTrainInferenceForwardingClient(config, db_engine=None)

    assert client._http_client.timeout.read == 42.0

    await client.aclose()
