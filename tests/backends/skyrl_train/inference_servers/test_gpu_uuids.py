"""CPU-only tests for physical GPU reports from the inference control plane."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.parametrize("uuid", ["GPU-physical-5", b"GPU-physical-5"])
def test_worker_reports_current_physical_gpu_uuid(monkeypatch, uuid):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 5)

    def get_device_properties(device):
        assert device == 5
        return SimpleNamespace(uuid=uuid)

    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    worker = NewInferenceWorkerWrap()

    assert worker.skyrl_get_gpu_uuid() == "GPU-physical-5"


@pytest.mark.parametrize("engines,dp,tp", [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 2, 2)])
def test_gpu_uuids_include_each_server_worker_once(monkeypatch, engines, dp, tp):
    urls = [f"http://engine-{engine}-dp-{replica}" for engine in range(engines) for replica in range(dp)]
    client = RemoteInferenceClient(proxy_url="http://unused", server_urls=urls, data_parallel_size=dp)
    expected = {url: [f"{url}-gpu-{rank}" for rank in range(tp)] for url in urls}
    call_server = AsyncMock(side_effect=lambda url, *args: (url, {"status": 200, "body": {"results": expected[url]}}))
    monkeypatch.setattr(client, "_call_server", call_server)

    assert asyncio.run(client.get_gpu_uuids()) == expected

    assert call_server.await_count == len(urls)
    for call in call_server.await_args_list:
        assert call.args[1:3] == ("/collective_rpc", {"method": "skyrl_get_gpu_uuid"})


def test_gpu_uuid_query_preserves_duplicates_for_validation(monkeypatch):
    client = RemoteInferenceClient(proxy_url="http://unused", server_urls=["http://server"], data_parallel_size=1)
    monkeypatch.setattr(
        client,
        "_call_all_servers",
        AsyncMock(return_value={"http://server": {"body": {"results": ["GPU-0", "GPU-0"]}}}),
    )

    assert asyncio.run(client.get_gpu_uuids()) == {"http://server": ["GPU-0", "GPU-0"]}


@pytest.mark.parametrize(
    "response", [None, {}, {"body": None}, {"body": {"results": []}}, {"body": {"results": [None]}}]
)
def test_gpu_uuid_query_rejects_missing_worker_reports(monkeypatch, response):
    client = RemoteInferenceClient(proxy_url="http://unused", server_urls=["http://server"], data_parallel_size=1)
    monkeypatch.setattr(client, "_call_all_servers", AsyncMock(return_value={"http://server": response}))

    with pytest.raises(RuntimeError, match="Missing or invalid GPU UUIDs from http://server"):
        asyncio.run(client.get_gpu_uuids())
