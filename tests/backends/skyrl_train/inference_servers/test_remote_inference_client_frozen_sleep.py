"""CPU tests for ``RemoteInferenceClient.sleep_with_frozen_requests``."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


def _client(**kwargs) -> tuple[RemoteInferenceClient, list]:
    client = RemoteInferenceClient(proxy_url="http://router", server_urls=["http://s0"], data_parallel_size=1, **kwargs)
    calls = []

    async def _call(endpoint, json=None, method="POST", params=None):
        calls.append((endpoint, json, params))
        return {}

    client._call_all_servers = _call
    return client, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("offload_kv", [True, False])
async def test_sleep_with_frozen_requests_pauses_in_keep_mode_then_sleeps_the_allocator(offload_kv):
    """The client pauses in keep mode, sleeps the allocator, and aborts no request."""
    client, calls = _client()
    assert client.requests_frozen_asleep is False
    await client.sleep_with_frozen_requests(offload_kv=offload_kv)
    assert calls == [
        ("/pause", None, {"mode": "keep", "clear_cache": "false"}),
        ("/collective_rpc", {"method": "skyrl_sleep_for_weight_sync", "kwargs": {"offload_kv": offload_kv}}, None),
    ]
    assert client.requests_frozen_asleep is True
    await client.resume_generation()
    assert client.requests_frozen_asleep is False


@pytest.mark.asyncio
async def test_sleep_with_frozen_requests_keeps_the_weights_for_a_lora_sync():
    """A LoRA adapter sync does not send the base model again, so the sleep offloads the weights."""
    client, calls = _client(uses_lora_weight_sync=True)
    await client.sleep_with_frozen_requests(offload_kv=True)
    assert calls[-1] == (
        "/collective_rpc",
        {"method": "skyrl_sleep_for_weight_sync", "kwargs": {"offload_kv": True, "offload_weights": True}},
        None,
    )


@pytest.mark.asyncio
async def test_sleep_for_weight_sync_discards_the_weights_unless_asked():
    """Only ``sleep_with_frozen_requests`` keeps the weights for a LoRA sync. A direct call does not,
    so the fully-async weight sync never copies the base model to host memory."""
    client, calls = _client(uses_lora_weight_sync=True)
    await client.sleep_for_weight_sync(offload_kv=True)
    assert calls[-1] == (
        "/collective_rpc",
        {"method": "skyrl_sleep_for_weight_sync", "kwargs": {"offload_kv": True}},
        None,
    )
