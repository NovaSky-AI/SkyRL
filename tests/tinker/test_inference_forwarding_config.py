import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import aiohttp
import pytest
from aiohttp import web

from skyrl.tinker.api import (
    EncodedTextChunk,
    ModelInput,
    SampleRequest,
    SamplingParams,
    _should_forward_sample_requests,
)
from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
    TransientInferenceError,
)


def test_forwarding_timeout_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", "1800")
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(["--base-model", "test-model"])
    config = EngineConfig.model_validate(vars(args))

    assert config.forwarding_inference_timeout_sec == 1800.0


def test_base_checkpoint_path_parses() -> None:
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(["--base-model", "test-model", "--base-model-checkpoint-path", "/models/custom"])

    assert EngineConfig.model_validate(vars(args)).base_model_checkpoint_path == "/models/custom"


def test_runtime_role_flag_parses() -> None:
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(
        [
            "--base-model",
            "test-model",
            "--runtime-role",
            "trainer",
        ]
    )
    config = EngineConfig.model_validate(vars(args))

    assert config.runtime_role == "trainer"


@pytest.mark.parametrize(
    ("runtime_role", "colocate_all", "expected"),
    [
        ("trainer", False, False),
        ("inference", False, False),
        ("combined", False, True),
        ("combined", True, False),
    ],
)
def test_skyrl_train_forwarding_requires_non_colocated_combined_runtime(
    runtime_role: str, colocate_all: bool, expected: bool
) -> None:
    config = EngineConfig(
        base_model="test-model",
        backend="fsdp",
        runtime_role=runtime_role,
        backend_config={"trainer.placement.colocate_all": colocate_all},
    )

    assert _should_forward_sample_requests(config) is expected


@pytest.mark.asyncio
async def test_forwarding_client_uses_configured_timeout_and_connection_limit() -> None:
    config = EngineConfig(
        base_model="test-model",
        forwarding_inference_timeout_sec=1800.0,
        forwarding_inference_max_connections=64,
    )
    client = SkyRLTrainInferenceForwardingClient(config, db_engine=None, external_future_store=ExternalFutureStore())
    try:
        session = client._get_session()
        assert session.timeout.sock_connect == 60.0
        assert session.timeout.sock_read == 1800.0
        # The forwarding scope supplies one overall deadline across attempts.
        assert session.timeout.total is None
        assert session.connector.limit == 64
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_forwarding_client_default_connection_limit_is_unlimited() -> None:
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="test-model"), db_engine=None, external_future_store=ExternalFutureStore()
    )
    try:
        assert client._get_session().connector.limit == 0
    finally:
        await client.aclose()


def _connect_error(message: str) -> aiohttp.ClientConnectorError:
    return aiohttp.ClientConnectorError(SimpleNamespace(ssl=None, host="inference", port=8000), OSError(message))


@pytest.mark.asyncio
async def test_forwarding_retries_connection_failure() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model")
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[_connect_error("unreachable"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_has_awaits([call(), call(force_refresh=True)])
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_retries_transient_5xx_once() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model")
    client._cached_proxy_url = "http://old"
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[TransientInferenceError("503 from router"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_4xx() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model")
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=RuntimeError("vLLM /v1/completions returned 400: bad request"))

    with pytest.raises(RuntimeError, match="returned 400"):
        await client._forward_with_retry(object(), "model", base_model=None)

    client._forward.assert_awaited_once()


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_read_timeout() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=123.0)
    client._cached_proxy_url = "http://inference"
    client._resolve_proxy_url = AsyncMock(return_value="http://inference")
    client._forward = AsyncMock(side_effect=aiohttp.SocketTimeoutError("slow response"))

    with pytest.raises(RuntimeError) as exc_info:
        await client._forward_with_retry(object(), "model", base_model=None)

    message = str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, aiohttp.SocketTimeoutError)
    assert "http://inference" in message
    assert "timed out after 123s" in message
    client._resolve_proxy_url.assert_awaited_once_with()
    client._forward.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_status", "delay", "budget", "expected_status", "expected_calls"),
    [
        (200, 0.1, 0.5, RequestStatus.COMPLETED, 1),
        (200, 0.6, 0.4, RequestStatus.FAILED, 1),
        (500, 0.1, 0.5, RequestStatus.COMPLETED, 2),
        (500, 0.3, 0.5, RequestStatus.FAILED, 2),
    ],
)
async def test_forwarded_future_shares_deadline_across_http_attempts(
    first_status, delay, budget, expected_status, expected_calls
) -> None:
    calls = 0

    async def complete(request):
        nonlocal calls
        await request.read()
        calls += 1
        status = first_status if calls == 1 else 200
        await asyncio.sleep(delay)
        return web.json_response(
            {"choices": [{"token_ids": [2], "logprobs": {"token_logprobs": [-0.1]}, "finish_reason": "length"}]},
            status=status,
        )

    app = web.Application()
    app.router.add_post("/v1/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    store = ExternalFutureStore()
    client = SkyRLTrainInferenceForwardingClient(
        EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=budget),
        db_engine=None,
        external_future_store=store,
    )
    client._resolve_proxy_url = AsyncMock(return_value=f"http://127.0.0.1:{runner.addresses[0][1]}")
    request = SampleRequest(
        base_model="test-model",
        prompt=ModelInput(chunks=[EncodedTextChunk(tokens=[1])]),
        sampling_params=SamplingParams(max_tokens=1),
    )
    request_id = store.create("", request)
    try:
        await client.call_and_store_result(request_id, request, "", "", base_model="test-model")
        result = await store.wait(request_id, timeout=1)
        assert result is not None
        assert result[0] == expected_status
        assert calls == expected_calls
        if expected_status == RequestStatus.FAILED:
            assert f"timed out after {budget:g}s" in result[2]
        else:
            assert store.proto_result(request_id)
    finally:
        await client.aclose()
        await runner.cleanup()
