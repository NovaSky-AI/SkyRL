import argparse
from unittest.mock import AsyncMock, call, patch

import httpx
import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.db_models import EngineStateDB
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
    client._resolve_proxy_url = AsyncMock(side_effect=["http://old", "http://new"])
    expected = object()
    client._forward = AsyncMock(side_effect=[httpx.ConnectError("unreachable"), expected])

    result = await client._forward_with_retry(object(), "model", base_model=None)

    assert result is expected
    client._resolve_proxy_url.assert_has_awaits([call(), call()])
    assert client._forward.await_count == 2


@pytest.mark.asyncio
async def test_forwarding_does_not_retry_read_timeout() -> None:
    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.engine_config = EngineConfig(base_model="test-model", forwarding_inference_timeout_sec=123.0)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("next_url", ["http://new-router", None])
async def test_forwarding_observes_endpoint_change_after_model_teardown(tmp_path, next_url) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    client = SkyRLTrainInferenceForwardingClient(EngineConfig(base_model="test-model"), engine)
    sample_request = object()
    # The old port may be reused by a non-HTTP actor: no ConnectError triggers refresh.
    client._forward = AsyncMock(return_value=object())
    try:
        async with AsyncSession(engine) as session:
            session.add(EngineStateDB(singleton_id=1, inference_proxy_url="http://old-router"))
            await session.commit()
        await client._forward_with_retry(sample_request, "old-model", base_model=None)

        async with AsyncSession(engine) as session:
            row = await session.get(EngineStateDB, 1)
            row.inference_proxy_url = next_url
            await session.commit()

        if next_url is None:
            with pytest.raises(RuntimeError, match="no proxy URL published"):
                await client._forward_with_retry(sample_request, "new-model", base_model=None)
            client._forward.assert_awaited_once()
        else:
            await client._forward_with_retry(sample_request, "new-model", base_model=None)
            client._forward.assert_has_awaits(
                [
                    call(
                        "http://old-router",
                        sample_request,
                        "old-model",
                        base_model=None,
                    ),
                    call(next_url, sample_request, "new-model", base_model=None),
                ]
            )
    finally:
        await client.aclose()
        await engine.dispose()
