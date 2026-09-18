from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    pack_full_logprobs,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


@pytest.mark.asyncio
async def test_full_logprobs_use_skyrl_endpoint_and_decode_rows(monkeypatch):
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
        logprob_output="full",
    )
    captured = {}
    raw_rows = [
        {token_id: SimpleNamespace(logprob=-float(token_id + 1)) for token_id in range(3)},
        {token_id: SimpleNamespace(logprob=-float(token_id + 4)) for token_id in range(3)},
    ]

    async def return_full_rows(url, json, headers):
        captured.update(url=url, json=json, headers=headers)
        return {
            "choices": [
                {
                    "token_ids": [1, 2],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -2.0}, {"logprob": -6.0}]},
                    "full_logprobs": pack_full_logprobs([1, 2], raw_rows, vocab_size=3),
                }
            ]
        }

    monkeypatch.setattr(client, "_post", return_full_rows)
    sampling_params = {"logprobs": 1, "n": 1}

    result = await client._generate_single([10], sampling_params, None, "model")

    assert captured["url"] == "http://unused/skyrl/v1/generate"
    assert captured["json"]["sampling_params"]["logprobs"] == -1
    assert captured["json"]["return_full_logprobs"] is True
    assert sampling_params["logprobs"] == 1
    assert result["response_logprobs"] == [-2.0, -6.0]
    np.testing.assert_array_equal(
        result["response_full_logprobs"],
        np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32),
    )


@pytest.mark.asyncio
async def test_action_mode_logprobs_minus_one_does_not_request_full_payload(monkeypatch):
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
    )
    captured = {}

    async def return_sampled_rows(url, json, headers):
        captured.update(url=url, json=json, headers=headers)
        return {
            "choices": [
                {
                    "token_ids": [1],
                    "finish_reason": "stop",
                    "logprobs": {"content": [{"logprob": -2.0}]},
                }
            ]
        }

    monkeypatch.setattr(client, "_post", return_sampled_rows)

    result = await client._generate_single([10], {"logprobs": -1}, None, "model")

    assert captured["url"] == "http://unused/inference/v1/generate"
    assert "return_full_logprobs" not in captured["json"]
    assert result["response_logprobs"] == [-2.0]
    assert "response_full_logprobs" not in result


@pytest.mark.asyncio
async def test_isoexec_wake_verifies_applied_weights():
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
        uses_isoexec=True,
    )
    client._call_all_servers = AsyncMock(return_value={"http://unused": {"status": "ok"}})

    await client.wake_up()

    assert client._call_all_servers.await_args_list == [
        call("/wake_up", params={}),
        call("/collective_rpc", {"method": "isoexec_verify_after_wake"}),
    ]


@pytest.mark.asyncio
async def test_isoexec_weights_only_wake_defers_verification_until_after_sync():
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
        uses_isoexec=True,
    )
    client._call_all_servers = AsyncMock(return_value={"http://unused": {"status": "ok"}})

    await client.wake_up(tags=["weights"])
    await client.wake_up(tags=["kv_cache"])

    assert client._call_all_servers.await_args_list == [
        call("/wake_up", params={"tags": ["weights"]}),
        call("/wake_up", params={"tags": ["kv_cache"]}),
        call("/collective_rpc", {"method": "isoexec_verify_after_wake"}),
    ]


@pytest.mark.asyncio
async def test_isoexec_direct_weight_sync_wake_defers_verification_until_kv_restore():
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
        uses_isoexec=True,
    )
    client._call_all_servers = AsyncMock(return_value={"http://unused": {"status": "ok"}})

    await client.wake_for_weight_sync(tags=["weights"])
    await client.wake_for_weight_sync(tags=["kv_cache"])

    assert client._call_all_servers.await_args_list == [
        call(
            "/collective_rpc",
            {"method": "skyrl_wake_for_weight_sync", "kwargs": {"tags": ["weights"]}},
        ),
        call(
            "/collective_rpc",
            {"method": "skyrl_wake_for_weight_sync", "kwargs": {"tags": ["kv_cache"]}},
        ),
        call("/collective_rpc", {"method": "isoexec_verify_after_wake"}),
    ]


@pytest.mark.asyncio
async def test_isoexec_preserve_weights_forces_level_one_sleep():
    client = RemoteInferenceClient(
        proxy_url="http://unused",
        server_urls=["http://unused"],
        data_parallel_size=1,
        preserve_weights_on_sleep=True,
    )
    client._call_all_servers = AsyncMock(return_value={"http://unused": {"status": "ok"}})

    await client.sleep(level=2)

    client._call_all_servers.assert_awaited_once_with("/sleep", params={"level": "1"})
