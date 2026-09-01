import asyncio
from collections import defaultdict
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.inference_servers.lora_reload import (
    replace_lora_adapter,
)


class _Counter:
    def __init__(self):
        self.value = 0

    def inc(self, amount):
        self.value += amount
        return self.value


class _LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _EngineClient:
    def __init__(self):
        self.events = []

    async def add_lora(self, request):
        self.events.append(("add", request.lora_int_id, request.lora_path))

    async def remove_lora(self, lora_int_id):
        self.events.append(("remove", lora_int_id))


@pytest.mark.asyncio
async def test_reloading_lora_replaces_engine_id():
    engine_client = _EngineClient()
    models = SimpleNamespace(
        engine_client=engine_client,
        lora_id_counter=_Counter(),
        lora_requests={},
        lora_resolver_lock=defaultdict(asyncio.Lock),
    )

    first_id = await replace_lora_adapter(models, "policy", "/adapters/step-1", _LoRARequest)
    second_id = await replace_lora_adapter(models, "policy", "/adapters/step-2", _LoRARequest)

    assert (first_id, second_id) == (1, 2)
    assert engine_client.events == [
        ("add", 1, "/adapters/step-1"),
        ("remove", 1),
        ("add", 2, "/adapters/step-2"),
    ]
    assert models.lora_requests["policy"].lora_int_id == 2
    assert models.lora_requests["policy"].lora_path == "/adapters/step-2"
