import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, db_observability, types
from skyrl.tinker.db_models import FutureDB, RequestStatus
from skyrl.tinker.extra import skyrl_train_inference_forwarding
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


@pytest.mark.asyncio
async def test_logs_slow_statement_with_pool_state(monkeypatch, tmp_path):
    timestamps = iter([10.0, 12.0])
    logger = Mock()
    monkeypatch.setattr(db_observability, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(db_observability, "logger", logger)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'slow.db'}")
    db_observability.enable_database_observability(engine)
    db_observability.enable_database_observability(engine)

    async with engine.connect() as connection:
        await connection.exec_driver_sql("SELECT 1")
    await engine.dispose()

    assert logger.warning.call_count == 1
    assert logger.warning.call_args.args[1:4] == ("SELECT", "sqlite", 2.0)
    assert "Pool size" in logger.warning.call_args.args[4]


@pytest.mark.asyncio
async def test_logs_failed_statement_without_query_values(monkeypatch, tmp_path):
    timestamps = iter([10.0, 10.25])
    logger = Mock()
    monkeypatch.setattr(db_observability, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(db_observability, "logger", logger)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'failed.db'}")
    db_observability.enable_database_observability(engine)

    with pytest.raises(OperationalError):
        async with engine.connect() as connection:
            await connection.exec_driver_sql("SELECT secret_value FROM missing_table")
    await engine.dispose()

    assert logger.error.call_count == 1
    assert logger.error.call_args.args[1:4] == ("SELECT", "sqlite", "0.250")
    assert "secret_value" not in str(logger.error.call_args)
    assert "missing_table" not in str(logger.error.call_args)


async def _create_forwarder_fixture(tmp_path, pool_size=5, pool_timeout=30.0):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'forwarder.db'}",
        pool_size=pool_size,
        max_overflow=0,
        pool_timeout=pool_timeout,
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as session:
        future = FutureDB(
            request_type=types.RequestType.EXTERNAL,
            model_id="model_a",
            seq_id=7,
            request_data={},
        )
        session.add(future)
        await session.commit()
        await session.refresh(future)

    client = object.__new__(SkyRLTrainInferenceForwardingClient)
    client.db_engine = engine
    sample_request = SimpleNamespace(
        prompt=SimpleNamespace(chunks=[SimpleNamespace(tokens=[1, 2, 3])]),
        sampling_params=SimpleNamespace(max_tokens=4),
        sampling_session_id="sampling_a",
        seq_id=7,
    )
    return engine, client, sample_request, future.request_id


@pytest.mark.asyncio
async def test_forwarding_failure_logs_request_dimensions(monkeypatch, tmp_path):
    engine, client, sample_request, request_id = await _create_forwarder_fixture(tmp_path)
    timestamps = iter([10.0, 12.0, 12.0, 12.0, 12.1])
    logger = Mock()
    client._forward_with_retry = AsyncMock(side_effect=httpx.ReadTimeout("connection reset"))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "logger", logger)

    await client.call_and_store_result(request_id, sample_request, "model_a", "weights_a")

    assert logger.exception.call_count == 1
    assert logger.exception.call_args.args[1:7] == (
        request_id,
        "model_a",
        "sampling_a",
        7,
        3,
        4,
    )
    async with AsyncSession(engine) as session:
        future = await session.get(FutureDB, request_id)
        assert future.status == RequestStatus.FAILED
        assert "connection reset" in future.result_data
    await engine.dispose()


@pytest.mark.asyncio
async def test_proxy_database_failure_has_distinct_stage(monkeypatch, tmp_path):
    engine, client, sample_request, request_id = await _create_forwarder_fixture(tmp_path)
    logger = Mock()
    client._forward_with_retry = AsyncMock(side_effect=SQLAlchemyTimeoutError("proxy pool exhausted"))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "logger", logger)

    await client.call_and_store_result(request_id, sample_request, "model_a", "weights_a")

    assert logger.error.call_count == 1
    assert "failure_stage=proxy_database" in logger.error.call_args.args[0]
    assert "Pool size" in logger.error.call_args.args[8]
    assert logger.error.call_args.args[9] == "TimeoutError"
    assert "proxy pool exhausted" not in str(logger.error.call_args)
    async with AsyncSession(engine) as session:
        future = await session.get(FutureDB, request_id)
        assert future.status == RequestStatus.FAILED
        assert "proxy pool exhausted" in future.result_data
    await engine.dispose()


@pytest.mark.asyncio
async def test_slow_completion_write_logs_pool_and_timing(monkeypatch, tmp_path):
    engine, client, sample_request, request_id = await _create_forwarder_fixture(tmp_path)
    timestamps = iter([10.0, 10.1, 10.1, 11.5])
    logger = Mock()
    client._forward_with_retry = AsyncMock(return_value=types.SampleOutput(sequences=[]))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "logger", logger)

    await client.call_and_store_result(request_id, sample_request, "model_a", "weights_a")

    assert logger.warning.call_count == 1
    assert logger.warning.call_args.args[1:8] == (
        request_id,
        "model_a",
        7,
        RequestStatus.COMPLETED.value,
        3,
        pytest.approx(0.1),
        pytest.approx(1.4),
    )
    assert "Pool size" in logger.warning.call_args.args[8]
    await engine.dispose()


@pytest.mark.asyncio
async def test_pool_exhaustion_logs_failing_future(monkeypatch, tmp_path):
    engine, client, sample_request, request_id = await _create_forwarder_fixture(
        tmp_path,
        pool_size=1,
        pool_timeout=0.01,
    )
    logger = Mock()
    client._forward_with_retry = AsyncMock(return_value=types.SampleOutput(sequences=[]))
    monkeypatch.setattr(skyrl_train_inference_forwarding, "logger", logger)

    with pytest.raises(SQLAlchemyTimeoutError):
        async with engine.connect():
            await client.call_and_store_result(
                request_id,
                sample_request,
                "model_a",
                "weights_a",
            )

    assert logger.error.call_count == 1
    assert logger.error.call_args.args[1:6] == (
        request_id,
        "model_a",
        7,
        RequestStatus.COMPLETED.value,
        3,
    )
    assert "Pool size: 1" in logger.error.call_args.args[8]
    assert logger.error.call_args.args[9] == "TimeoutError"
    await engine.dispose()


@pytest.mark.asyncio
async def test_poller_logs_pool_checkout_timeout(monkeypatch, tmp_path):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'poller.db'}",
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    logger = Mock()
    monkeypatch.setattr(api, "logger", logger)
    waiter = asyncio.get_running_loop().create_future()

    async with engine.connect():
        poller = asyncio.create_task(api.poll_futures(engine, {1: {waiter}}, 0.001))
        await asyncio.sleep(0.03)
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)

    messages = [call.args[0] for call in logger.error.call_args_list]
    expected_message = (
        "Future poller database failure failure_stage=future_poller awaited_futures=%s " "pool=%s error_type=%s"
    )
    assert expected_message in messages
    matching = next(call for call in logger.error.call_args_list if call.args[0] == expected_message)
    assert matching.args[1] == 1
    assert "Pool size: 1" in matching.args[2]
    assert matching.args[3] == "TimeoutError"
    waiter.cancel()
    await engine.dispose()


@pytest.mark.asyncio
async def test_api_boundary_logs_database_failure(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    logger = Mock()
    monkeypatch.setattr(api, "logger", logger)
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/asample"),
        app=SimpleNamespace(state=SimpleNamespace(db_engine=engine)),
    )

    async def fail_request(_request):
        raise SQLAlchemyTimeoutError("pool exhausted")

    with pytest.raises(SQLAlchemyTimeoutError):
        await api.log_database_request_failure(request, fail_request)

    assert logger.error.call_count == 1
    assert logger.error.call_args.args[1:3] == ("POST", "/api/v1/asample")
    assert "Pool size" in logger.error.call_args.args[4]
    assert logger.error.call_args.args[5] == "TimeoutError"
    assert "pool exhausted" not in str(logger.error.call_args)
    await engine.dispose()
