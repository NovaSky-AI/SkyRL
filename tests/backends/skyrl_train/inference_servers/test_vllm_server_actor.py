import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    _remove_lora_adapter,
)


@pytest.mark.asyncio
async def test_remove_lora_adapter_clears_engine_and_registry():
    engine_client = SimpleNamespace(remove_lora=AsyncMock(return_value=True))
    models = SimpleNamespace(
        engine_client=engine_client,
        lora_requests={"model-a": SimpleNamespace(lora_int_id=7)},
        lora_resolver_lock=defaultdict(asyncio.Lock),
    )

    lora_int_id = await _remove_lora_adapter(models, "model-a")

    assert lora_int_id == 7
    engine_client.remove_lora.assert_awaited_once_with(7)
    assert "model-a" not in models.lora_requests


@pytest.mark.asyncio
async def test_remove_lora_adapter_keeps_registry_when_engine_removal_fails():
    engine_client = SimpleNamespace(remove_lora=AsyncMock(side_effect=RuntimeError("engine failed")))
    models = SimpleNamespace(
        engine_client=engine_client,
        lora_requests={"model-a": SimpleNamespace(lora_int_id=7)},
        lora_resolver_lock=defaultdict(asyncio.Lock),
    )

    with pytest.raises(RuntimeError, match="engine failed"):
        await _remove_lora_adapter(models, "model-a")

    assert "model-a" in models.lora_requests


@pytest.mark.asyncio
async def test_remove_lora_adapter_rejects_unknown_adapter():
    engine_client = SimpleNamespace(remove_lora=AsyncMock())
    models = SimpleNamespace(
        engine_client=engine_client,
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
    )

    with pytest.raises(HTTPException) as exc_info:
        await _remove_lora_adapter(models, "missing")

    assert exc_info.value.status_code == 404
    engine_client.remove_lora.assert_not_awaited()
