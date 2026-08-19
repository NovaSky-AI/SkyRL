import asyncio
import json
import os
import random
import time

import httpx
import psutil
import pytest
import pytest_asyncio
import tinker.types as sdk_types
import zstandard as zstd
from fastapi import Depends, FastAPI, Request
from google.protobuf.message import Message
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select
from sqlmodel.ext.asyncio.session import AsyncSession
from tinker.proto.request_conv import forward_backward_request_to_proto

from skyrl.tinker import api, types
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    FutureDB,
    ModelDB,
    RequestStatus,
    SessionDB,
    enable_sqlite_wal,
)
from skyrl.tinker.extra import InMemoryFutureStore
from skyrl.tinker.operation_transport import (
    OperationPayloadFormat,
    OperationTransportClient,
    OperationTransportServer,
    TrainingOperationQueue,
)


@pytest_asyncio.fixture()
async def operation_api(tmp_path):
    db_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tinker.db'}")
    async with db_engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(db_engine) as session:
        session.add(SessionDB(session_id="session_a", sdk_version="test"))
        session.add(
            ModelDB(
                model_id="model_a",
                base_model="test/model",
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="created",
                request_id=1,
                session_id="session_a",
            )
        )
        await session.commit()

    future_store = InMemoryFutureStore()
    api.app.state.db_engine = db_engine
    api.app.state.external_future_store = future_store
    api.app.state.training_operation_queue = TrainingOperationQueue(future_store)
    api.app.state.training_model_locks = {}
    api.app.state.training_control_locks = {}
    parse_concurrency = int(os.getenv("SKYRL_TRANSPORT_PARSE_CONCURRENCY", str(api.TRAINING_PARSE_CONCURRENCY)))
    api.app.state.training_parse_semaphore = asyncio.Semaphore(parse_concurrency)
    transport = httpx.ASGITransport(app=api.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, db_engine, api.app.state.training_operation_queue
    await db_engine.dispose()


def _forward_backward(seq_id: int) -> dict:
    return {
        "model_id": "model_a",
        "seq_id": seq_id,
        "forward_backward_input": {"data": [], "loss_fn": "cross_entropy"},
    }


def _proto_forward_backward(seq_id: int, *, forward_only: bool = False) -> bytes:
    request = sdk_types.ForwardBackwardRequest(
        model_id="model_a",
        seq_id=seq_id,
        forward_backward_input=sdk_types.ForwardBackwardInput(
            data=[
                sdk_types.Datum(
                    model_input=sdk_types.ModelInput.from_ints([1, 2, 3, 4]),
                    loss_fn_inputs={
                        "target_tokens": sdk_types.TensorData(data=[2, 3, 4, 5], dtype="int64", shape=[4]),
                        "weights": sdk_types.TensorData(data=[1.0, 0.5, 1.0, 0.0], dtype="float32", shape=[4]),
                    },
                )
            ],
            loss_fn="cross_entropy",
        ),
    )
    message = forward_backward_request_to_proto(request)
    message.forward_only = forward_only
    return message.SerializeToString()


def _realistic_forward_backward_body() -> bytes:
    """Build one valid native-GSPO request body near the production size."""
    token_count = 216_250
    tokens = [1] * token_count
    ones = [1.0] * token_count
    zeros = [0.0] * token_count
    request = {
        "model_id": "model_a",
        "seq_id": 0,
        "forward_backward_input": {
            "data": [
                {
                    "model_input": {"chunks": [{"tokens": tokens}]},
                    "loss_fn_inputs": {
                        "target_tokens": {"data": tokens},
                        "weights": {"data": ones},
                        "advantages": {"data": ones},
                        "logprobs": {"data": zeros},
                    },
                }
            ],
            "loss_fn": "gspo",
            "loss_fn_config": {
                "clip_low_threshold": 0.2,
                "clip_high_threshold": 1.2,
            },
        },
    }
    return json.dumps(request, separators=(",", ":")).encode()


def _realistic_proto_message() -> Message:
    """Build one valid native-GSPO protobuf request at production scale."""
    token_count = 276_000
    generator = random.Random(0)
    tokens = [generator.randrange(32_000) for _ in range(token_count)]
    weights = [0.0 if index % 7 == 0 else 1.0 for index in range(token_count)]
    advantages = [generator.uniform(-10, 10) for _ in range(token_count)]
    logprobs = [generator.uniform(-10, 0) for _ in range(token_count)]
    request = sdk_types.ForwardBackwardRequest(
        model_id="model_a",
        seq_id=0,
        forward_backward_input=sdk_types.ForwardBackwardInput(
            data=[
                sdk_types.Datum(
                    model_input=sdk_types.ModelInput.from_ints(tokens),
                    loss_fn_inputs={
                        "target_tokens": sdk_types.TensorData(
                            data=tokens,
                            dtype="int64",
                            shape=[token_count],
                        ),
                        "weights": sdk_types.TensorData(
                            data=weights,
                            dtype="float32",
                            shape=[token_count],
                        ),
                        "advantages": sdk_types.TensorData(
                            data=advantages,
                            dtype="float32",
                            shape=[token_count],
                        ),
                        "logprobs": sdk_types.TensorData(
                            data=logprobs,
                            dtype="float32",
                            shape=[token_count],
                        ),
                    },
                )
            ],
            loss_fn="gspo",
            loss_fn_config={
                "clip_low_threshold": 0.2,
                "clip_high_threshold": 1.2,
            },
        ),
    )
    return forward_backward_request_to_proto(request)


@pytest.mark.asyncio
async def test_http_training_burst_stays_out_of_sqlite_and_orders_optimizer(
    operation_api,
):
    client, db_engine, queue = operation_api
    requests = [client.post("/api/v1/forward_backward", json=_forward_backward(seq_id)) for seq_id in range(2, 209)]
    # The SDK deliberately submits the first chunk after the parallel tail.
    requests.append(client.post("/api/v1/forward_backward", json=_forward_backward(1)))
    requests.append(
        client.post(
            "/api/v1/optim_step",
            json={
                "model_id": "model_a",
                "seq_id": 209,
                "adam_params": {"learning_rate": 1e-4},
            },
        )
    )
    responses = await asyncio.gather(*requests)

    assert {response.status_code for response in responses} == {200}
    assert len({response.json()["request_id"] for response in responses}) == 209
    async with AsyncSession(db_engine) as session:
        assert (await session.exec(select(func.count()).select_from(FutureDB))).one() == 0

    model_passes = queue.claim_ready()
    assert [operation.seq_id for operation in model_passes] == list(range(1, 209))
    queue.acknowledge_claim([operation.request_id for operation in model_passes])
    for operation in model_passes:
        queue.complete(operation.request_id, api.RequestStatus.COMPLETED, "{}")
    (optimizer,) = queue.claim_ready()
    assert optimizer.request_type == types.RequestType.OPTIM_STEP
    assert optimizer.seq_id == 209


@pytest.mark.asyncio
@pytest.mark.parametrize("compressed", [False, True])
async def test_forward_backward_accepts_sdk_proto(operation_api, compressed):
    client, _, queue = operation_api
    body = _proto_forward_backward(1, forward_only=True)
    headers = {"content-type": api.PROTO_CONTENT_TYPE}
    if compressed:
        body = zstd.compress(body)
        headers["content-encoding"] = "zstd"

    response = await client.post("/api/v1/forward_backward", content=body, headers=headers)

    assert response.status_code == 200
    (operation,) = queue.claim_ready()
    assert operation.request_type == types.RequestType.FORWARD
    assert operation.seq_id == 1
    assert operation.payload == body
    assert operation.payload_format is (
        OperationPayloadFormat.PROTO_ZSTD if compressed else OperationPayloadFormat.PROTO
    )


@pytest.mark.asyncio
async def test_client_config_advertises_parallel_compressed_proto(operation_api):
    client, _, _ = operation_api

    response = await client.post("/api/v1/client/config")

    assert response.status_code == 200
    assert response.json() == {
        "pjwt_auth_enabled": False,
        "parallel_fwdbwd_chunks": True,
        "proto_write_fwdbwd": True,
        "proto_compress_fwdbwd": True,
        "fwd_via_fwdbwd": True,
    }


@pytest.mark.asyncio
async def test_forward_backward_rejects_invalid_or_oversized_zstd(operation_api, monkeypatch):
    client, _, _ = operation_api
    headers = {
        "content-type": api.PROTO_CONTENT_TYPE,
        "content-encoding": "zstd",
    }

    invalid = await client.post("/api/v1/forward_backward", content=b"not-zstd", headers=headers)
    monkeypatch.setattr(api, "MAX_TRAINING_REQUEST_BYTES", 1024)
    oversized = await client.post(
        "/api/v1/forward_backward",
        content=zstd.compress(b"x" * 1025),
        headers=headers,
    )

    assert invalid.status_code == 422
    assert oversized.status_code == 413


@pytest.mark.asyncio
async def test_training_retry_is_idempotent_and_conflicting_reuse_is_409(operation_api):
    client, _, _ = operation_api
    original = await client.post("/api/v1/forward_backward", json=_forward_backward(1))
    retry = await client.post("/api/v1/forward_backward", json=_forward_backward(1))
    conflict = await client.post(
        "/api/v1/forward",
        json={
            "model_id": "model_a",
            "seq_id": 1,
            "forward_input": {"data": [], "loss_fn": "cross_entropy"},
        },
    )

    assert original.status_code == retry.status_code == 200
    assert original.json()["request_id"] == retry.json()["request_id"]
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_checkpoint_request_shares_model_sequence_without_storing_payload(
    operation_api,
):
    client, db_engine, queue = operation_api
    first = await client.post("/api/v1/forward_backward", json=_forward_backward(1))
    save = await client.post(
        "/api/v1/save_weights",
        json={"model_id": "model_a", "seq_id": 2, "path": "checkpoint_a"},
    )
    later = await client.post("/api/v1/forward_backward", json=_forward_backward(3))

    assert first.status_code == save.status_code == later.status_code == 200
    async with AsyncSession(db_engine) as session:
        assert (await session.exec(select(func.count()).select_from(FutureDB))).one() == 0
        assert (await session.exec(select(func.count()).select_from(CheckpointDB))).one() == 1

    (first_operation,) = queue.claim_ready()
    queue.acknowledge_claim([first_operation.request_id])
    queue.complete(first_operation.request_id, api.RequestStatus.COMPLETED, "{}")
    (save_operation,) = queue.claim_ready()
    assert save_operation.request_type == types.RequestType.SAVE_WEIGHTS
    assert save_operation.seq_id == 2


@pytest.mark.asyncio
async def test_restart_marks_old_database_training_work_failed(operation_api):
    _, db_engine, _ = operation_api
    async with AsyncSession(db_engine) as session:
        session.add(
            FutureDB(
                request_type=types.RequestType.FORWARD_BACKWARD,
                model_id="model_a",
                seq_id=1,
                request_data={"old": "payload"},
            )
        )
        session.add(
            CheckpointDB(
                model_id="model_a",
                checkpoint_id="stale",
                checkpoint_type=types.CheckpointType.TRAINING,
                status=CheckpointStatus.PENDING,
            )
        )
        await session.commit()

    await api.fail_stale_training_operations(db_engine)

    async with AsyncSession(db_engine) as session:
        future = (await session.exec(select(FutureDB))).one()
        checkpoint = (await session.exec(select(CheckpointDB))).one()
    assert future.status == RequestStatus.FAILED
    assert "resume from a checkpoint" in future.result_data
    assert checkpoint.status == CheckpointStatus.FAILED


@pytest.mark.asyncio
async def test_restart_loses_in_memory_future_with_explicit_404(operation_api):
    client, _, _ = operation_api
    response = await client.post("/api/v1/forward_backward", json=_forward_backward(1))
    api.app.state.external_future_store = InMemoryFutureStore()

    retrieved = await client.post(
        "/api/v1/retrieve_future",
        json={"request_id": response.json()["request_id"]},
    )

    assert retrieved.status_code == 404


@pytest.mark.asyncio
async def test_queue_backpressure_is_retryable_http_429(operation_api):
    client, _, queue = operation_api
    queue.max_pending_per_model = 1
    first = await client.post("/api/v1/forward_backward", json=_forward_backward(1))
    rejected = await client.post("/api/v1/forward_backward", json=_forward_backward(2))

    assert first.status_code == 200
    assert rejected.status_code == 429
    assert rejected.headers["retry-after"] == "1"


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("SKYRL_RUN_TRANSPORT_LOAD_TEST") != "1",
    reason="set SKYRL_RUN_TRANSPORT_LOAD_TEST=1 for the 208-request production-size check",
)
async def test_production_size_training_update(operation_api, tmp_path, caplog):
    client, db_engine, operation_queue = operation_api
    request_count = int(os.getenv("SKYRL_TRANSPORT_LOAD_REQUESTS", "208"))
    operation_queue.register_model("model_a", 1)
    server = OperationTransportServer(operation_queue, str(tmp_path / "operations.sock"))
    await server.start()

    wire_format = os.getenv("SKYRL_TRANSPORT_LOAD_FORMAT", "json")
    proto_message = _realistic_proto_message() if wire_format == "proto-zstd" else None
    base_body = (
        zstd.compress(proto_message.SerializeToString())
        if proto_message is not None
        else _realistic_forward_backward_body()
    )
    assert 3.2 * 1024**2 <= len(base_body) <= 3.4 * 1024**2
    request_bodies = {}
    for seq_id in range(1, request_count + 1):
        if proto_message is not None:
            message = proto_message.__class__()
            message.CopyFrom(proto_message)
            message.seq_id = seq_id
            request_bodies[seq_id] = zstd.compress(message.SerializeToString())
        else:
            request_bodies[seq_id] = base_body.replace(
                b'"seq_id":0',
                f'"seq_id":{seq_id}'.encode(),
                1,
            )

    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    stop_memory_sample = asyncio.Event()
    submissions_done = asyncio.Event()
    request_latencies = []
    heartbeat_latencies = []
    heartbeat_statuses = []

    async def sample_memory():
        nonlocal peak_rss
        while not stop_memory_sample.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            await asyncio.sleep(0.02)

    async def send_heartbeats():
        while not submissions_done.is_set():
            started = time.perf_counter()
            response = await client.post("/api/v1/session_heartbeat", json={"session_id": "session_a"})
            heartbeat_latencies.append(time.perf_counter() - started)
            heartbeat_statuses.append(response.status_code)
            await asyncio.sleep(0.05)

    async def submit(seq_id: int):
        if proto_message is not None:
            headers = {
                "content-type": api.PROTO_CONTENT_TYPE,
                "content-encoding": "zstd",
            }
        else:
            headers = {"content-type": "application/json"}
        # ASGITransport is in-process. Copy the client-owned bytes once to
        # model the allocation that a real HTTP server makes while reading.
        body = bytes(bytearray(request_bodies[seq_id]))
        started = time.perf_counter()
        response = await client.post(
            "/api/v1/forward_backward",
            content=body,
            headers=headers,
        )
        request_latencies.append(time.perf_counter() - started)
        return response

    memory_task = asyncio.create_task(sample_memory())
    heartbeat_task = asyncio.create_task(send_heartbeats())
    submission_started = time.perf_counter()
    try:
        responses = await asyncio.gather(*(submit(seq_id) for seq_id in range(2, request_count + 1)), submit(1))
    finally:
        submissions_done.set()
        await heartbeat_task
    optimizer_response = await client.post(
        "/api/v1/optim_step",
        json={
            "model_id": "model_a",
            "seq_id": request_count + 1,
            "adam_params": {"learning_rate": 1e-4},
        },
    )
    submission_seconds = time.perf_counter() - submission_started
    status_codes = [response.status_code for response in responses]
    assert status_codes == [200] * request_count
    assert optimizer_response.status_code == 200, optimizer_response.text

    transport_client = await asyncio.to_thread(OperationTransportClient, str(tmp_path / "operations.sock"))
    transfer_started = time.perf_counter()
    operations = await asyncio.to_thread(transport_client.claim)
    claimed_seq_ids = [operation.seq_id for operation in operations]

    def complete_model_passes():
        result = types.ForwardBackwardOutput(loss_fn_output_type="per_token_loss", loss_fn_outputs=[], metrics={})
        for operation in operations:
            transport_client.complete(operation.request_id, result)

    await asyncio.to_thread(complete_model_passes)
    (optimizer,) = await asyncio.to_thread(transport_client.claim)
    await asyncio.to_thread(
        transport_client.complete,
        optimizer.request_id,
        types.OptimStepOutput(metrics={}),
    )
    transfer_seconds = time.perf_counter() - transfer_started

    first_future = responses[-1].json()["request_id"]
    first_result = await client.post("/api/v1/retrieve_future", json={"request_id": first_future})
    optimizer_result = await client.post(
        "/api/v1/retrieve_future",
        json={"request_id": optimizer_response.json()["request_id"]},
    )

    stop_memory_sample.set()
    await memory_task
    peak_rss = max(peak_rss, process.memory_info().rss)
    transport_client.close()
    await server.close()

    queue_pool_failures = sum("QueuePool" in record.getMessage() for record in caplog.records)
    async with AsyncSession(db_engine) as session:
        sqlite_future_rows = (await session.exec(select(func.count()).select_from(FutureDB))).one()

    metrics = {
        "accepted_forward_backward": status_codes.count(200),
        "http_4xx": sum(400 <= status < 500 for status in status_codes),
        "http_5xx": sum(status >= 500 for status in status_codes),
        "queue_pool_failures": queue_pool_failures,
        "sequence_gaps": len(set(range(1, request_count + 1)) - set(claimed_seq_ids)),
        "request_bytes": len(base_body),
        "wire_format": wire_format,
        "submission_seconds": round(submission_seconds, 3),
        "max_request_seconds": round(max(request_latencies), 3),
        "max_heartbeat_seconds": round(max(heartbeat_latencies), 3),
        "transport_and_completion_seconds": round(transfer_seconds, 3),
        "peak_rss_delta_mib": round((peak_rss - start_rss) / 1024**2, 1),
        "sqlite_future_rows": sqlite_future_rows,
    }
    print("\nTRAINING_TRANSPORT_METRICS=" + json.dumps(metrics, sort_keys=True))

    assert len({response.json()["request_id"] for response in responses}) == request_count
    assert claimed_seq_ids == list(range(1, request_count + 1))
    assert optimizer.seq_id == request_count + 1
    assert optimizer.request_type == types.RequestType.OPTIM_STEP
    assert heartbeat_statuses and heartbeat_statuses == [200] * len(heartbeat_statuses)
    assert max(request_latencies) < 60
    assert first_result.status_code == optimizer_result.status_code == 200
    assert queue_pool_failures == 0
    assert sqlite_future_rows == 0


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("SKYRL_RUN_LEGACY_SQLITE_LOAD_TEST") != "1",
    reason="set SKYRL_RUN_LEGACY_SQLITE_LOAD_TEST=1 for the previous SQLite path",
)
async def test_legacy_sqlite_production_size_training_update(tmp_path, caplog):
    """Measure the previous API path with the production pool and payload shape."""
    request_count = int(os.getenv("SKYRL_TRANSPORT_LOAD_REQUESTS", "208"))
    db_engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'legacy-tinker.db'}",
        pool_size=5,
        max_overflow=10,
    )
    enable_sqlite_wal(db_engine.sync_engine)
    async with db_engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(db_engine) as session:
        session.add(SessionDB(session_id="session_a", sdk_version="test"))
        session.add(
            ModelDB(
                model_id="model_a",
                base_model="test/model",
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="created",
                request_id=1,
                session_id="session_a",
            )
        )
        await session.commit()

    legacy_app = FastAPI()

    async def get_legacy_session() -> AsyncSession:
        async with AsyncSession(db_engine) as session:
            yield session

    @legacy_app.post("/api/v1/forward_backward")
    async def legacy_forward_backward(
        request: Request,
        session: AsyncSession = Depends(get_legacy_session),
    ):
        parsed = api.ForwardBackwardRequest.model_validate_json(await request.body())
        await api.get_model(session, parsed.model_id)
        request_id = await api.create_future(
            session=session,
            request_type=types.RequestType.FORWARD_BACKWARD,
            model_id=parsed.model_id,
            request_data=parsed.forward_backward_input.to_types(),
            seq_id=parsed.seq_id,
        )
        await session.commit()
        return {"request_id": str(request_id), "status": "pending"}

    @legacy_app.post("/api/v1/optim_step")
    async def legacy_optim_step(
        request: api.OptimStepRequest,
        session: AsyncSession = Depends(get_legacy_session),
    ):
        await api.get_model(session, request.model_id)
        request_id = await api.create_future(
            session=session,
            request_type=types.RequestType.OPTIM_STEP,
            model_id=request.model_id,
            request_data=types.OptimStepInput(adam_params=request.adam_params.to_types()),
            seq_id=request.seq_id,
        )
        await session.commit()
        return {"request_id": str(request_id), "status": "pending"}

    @legacy_app.post("/api/v1/session_heartbeat")
    async def legacy_heartbeat(
        request: api.SessionHeartbeatRequest,
        session: AsyncSession = Depends(get_legacy_session),
    ):
        session_db = await session.get(SessionDB, request.session_id)
        session_db.heartbeat_count += 1
        await session.commit()
        return {"type": "session_heartbeat"}

    base_body = _realistic_forward_backward_body()
    request_bodies = {
        seq_id: base_body.replace(b'"seq_id":0', f'"seq_id":{seq_id}'.encode(), 1)
        for seq_id in range(1, request_count + 1)
    }
    process = psutil.Process()
    start_rss = process.memory_info().rss
    peak_rss = start_rss
    stop_memory_sample = asyncio.Event()
    submissions_done = asyncio.Event()
    heartbeat_latencies = []
    heartbeat_statuses = []

    async def sample_memory():
        nonlocal peak_rss
        while not stop_memory_sample.is_set():
            peak_rss = max(peak_rss, process.memory_info().rss)
            await asyncio.sleep(0.02)

    transport = httpx.ASGITransport(app=legacy_app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def send_heartbeats():
            while not submissions_done.is_set():
                started = time.perf_counter()
                response = await client.post(
                    "/api/v1/session_heartbeat",
                    json={"session_id": "session_a"},
                )
                heartbeat_latencies.append(time.perf_counter() - started)
                heartbeat_statuses.append(response.status_code)
                await asyncio.sleep(0.05)

        async def submit(seq_id: int):
            body = bytes(bytearray(request_bodies[seq_id]))
            return await client.post(
                "/api/v1/forward_backward",
                content=body,
                headers={"content-type": "application/json"},
            )

        memory_task = asyncio.create_task(sample_memory())
        heartbeat_task = asyncio.create_task(send_heartbeats())
        submission_started = time.perf_counter()
        try:
            responses = await asyncio.gather(
                *(submit(seq_id) for seq_id in range(2, request_count + 1)),
                submit(1),
            )
        finally:
            submissions_done.set()
            await heartbeat_task
        optimizer_response = await client.post(
            "/api/v1/optim_step",
            json={
                "model_id": "model_a",
                "seq_id": request_count + 1,
                "adam_params": {"learning_rate": 1e-4},
            },
        )
        submission_seconds = time.perf_counter() - submission_started

    stop_memory_sample.set()
    await memory_task
    peak_rss = max(peak_rss, process.memory_info().rss)
    status_codes = [response.status_code for response in responses]
    async with AsyncSession(db_engine) as session:
        stored_seq_ids = set(
            (
                await session.exec(
                    select(FutureDB.seq_id)
                    .where(FutureDB.model_id == "model_a")
                    .where(FutureDB.request_type == types.RequestType.FORWARD_BACKWARD)
                )
            ).all()
        )
    queue_pool_failures = sum("QueuePool" in record.getMessage() for record in caplog.records)
    metrics = {
        "accepted_forward_backward": status_codes.count(200),
        "http_4xx": sum(400 <= status < 500 for status in status_codes),
        "http_5xx": sum(status >= 500 for status in status_codes),
        "heartbeat_4xx": sum(400 <= status < 500 for status in heartbeat_statuses),
        "heartbeat_5xx": sum(status >= 500 for status in heartbeat_statuses),
        "max_heartbeat_seconds": round(max(heartbeat_latencies), 3),
        "optimizer_status": optimizer_response.status_code,
        "peak_rss_delta_mib": round((peak_rss - start_rss) / 1024**2, 1),
        "queue_pool_failures": queue_pool_failures,
        "request_bytes": len(base_body),
        "sequence_gaps": len(set(range(1, request_count + 1)) - stored_seq_ids),
        "submission_seconds": round(submission_seconds, 3),
        "wire_format": "json",
    }
    print("\nLEGACY_SQLITE_METRICS=" + json.dumps(metrics, sort_keys=True))
    await db_engine.dispose()

    assert len(status_codes) == request_count
