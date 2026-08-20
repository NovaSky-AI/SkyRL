import asyncio
from types import SimpleNamespace

import pytest

from skyrl.tinker import types
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.engine import TinkerEngine
from skyrl.tinker.extra.in_memory_future_store import InMemoryFutureStore
from skyrl.tinker.operation_transport import (
    ClaimedOperation,
    OperationBackpressure,
    OperationExpired,
    OperationTransportClient,
    OperationTransportServer,
    TrainingOperationQueue,
)


def _payload(value: str = "cross_entropy") -> bytes:
    return types.ForwardBackwardInput(data=[], loss_fn=value).model_dump_json().encode()


def test_gap_buffering_idempotency_and_optimizer_barrier():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=1)

    second = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 2, _payload())
    third = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 3, _payload())
    assert queue.claim_ready() == []

    first = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, _payload())
    assert queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, _payload()) == first
    with pytest.raises(ValueError, match="sequence number was reused"):
        queue.enqueue(types.RequestType.FORWARD, "model_a", 1, _payload())

    save = queue.enqueue(
        types.RequestType.SAVE_WEIGHTS,
        "model_a",
        4,
        types.SaveWeightsInput(path="checkpoint").model_dump_json().encode(),
    )
    optim = queue.enqueue(
        types.RequestType.OPTIM_STEP,
        "model_a",
        5,
        types.OptimStepInput(
            adam_params=types.AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
                weight_decay=0.0,
            )
        )
        .model_dump_json()
        .encode(),
    )
    assert [operation.request_id for operation in queue.claim_ready()] == [
        first,
        second,
        third,
    ]
    assert queue.claim_ready() == []

    queue.acknowledge_claim([first, second, third])
    for request_id in (first, second, third):
        queue.complete(request_id, RequestStatus.COMPLETED, "{}")
    assert [operation.request_id for operation in queue.claim_ready()] == [save]
    queue.acknowledge_claim([save])
    queue.complete(save, RequestStatus.COMPLETED, "{}")
    assert [operation.request_id for operation in queue.claim_ready()] == [optim]


def test_forward_forward_backward_and_optimizer_keep_sequence_order():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=1)
    forward_backward = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 2, _payload())
    optimizer = queue.enqueue(
        types.RequestType.OPTIM_STEP,
        "model_a",
        3,
        types.OptimStepInput(
            adam_params=types.AdamParams(
                learning_rate=1e-4,
                beta1=0.9,
                beta2=0.95,
                eps=1e-12,
                weight_decay=0.0,
            )
        )
        .model_dump_json()
        .encode(),
    )
    forward = queue.enqueue(types.RequestType.FORWARD, "model_a", 1, _payload())

    for expected_request_id in (forward, forward_backward, optimizer):
        (operation,) = queue.claim_ready()
        assert operation.request_id == expected_request_id
        queue.acknowledge_claim([expected_request_id])
        queue.complete(expected_request_id, RequestStatus.COMPLETED, "{}")


def test_gap_filler_bypasses_backpressure_but_tail_does_not():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(
        futures,
        max_pending_per_model=2,
        max_payload_bytes=8,
        gap_reserve_bytes=4,
    )
    queue.register_model("model_a", next_seq_id=1)

    queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 2, b"tail")
    with pytest.raises(OperationBackpressure):
        queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 3, b"later")

    # Refusing the missing ordinal could never free capacity: seq=2 cannot run
    # until seq=1 arrives, so the exact hole filler is admitted over the cap.
    queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, b"gap")


def test_unsequenced_operation_runs_when_no_sequenced_request_is_pending():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=1)

    request_id = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", None, _payload())

    assert [operation.request_id for operation in queue.claim_ready()] == [request_id]


def test_registered_sequence_rejects_retry_from_older_database_history():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=10)

    with pytest.raises(OperationExpired):
        queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 9, _payload())


def test_terminal_retry_history_is_bounded_and_old_sequence_is_not_reused():
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures, max_terminal_records=1)
    queue.register_model("model_a", next_seq_id=1)
    for seq_id in (1, 2):
        request_id = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", seq_id, _payload())
        queue.claim_ready()
        queue.acknowledge_claim([request_id])
        queue.complete(request_id, RequestStatus.COMPLETED, "{}")

    with pytest.raises(OperationExpired):
        queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, _payload())


def test_retry_expires_when_its_completed_result_is_released():
    futures = InMemoryFutureStore(max_terminal_futures=1)
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=1)
    for seq_id in (1, 2):
        request_id = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", seq_id, _payload())
        queue.claim_ready()
        queue.acknowledge_claim([request_id])
        queue.complete(request_id, RequestStatus.COMPLETED, "{}")

    with pytest.raises(OperationExpired):
        queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, _payload())


@pytest.mark.asyncio
async def test_unix_transport_releases_payload_after_claim_and_completes_future(
    tmp_path,
):
    futures = InMemoryFutureStore()
    queue = TrainingOperationQueue(futures)
    queue.register_model("model_a", next_seq_id=1)
    payload = _payload()
    request_id = queue.enqueue(types.RequestType.FORWARD_BACKWARD, "model_a", 1, payload)

    socket_path = str(tmp_path / "operations.sock")
    server = OperationTransportServer(queue, socket_path)
    await server.start()
    client = await asyncio.to_thread(OperationTransportClient, socket_path)
    try:
        operations = await asyncio.to_thread(client.claim)
        assert len(operations) == 1
        assert operations[0].request_id == request_id
        assert operations[0].payload == payload
        assert queue.payload_bytes == 0

        result = types.ForwardBackwardOutput(loss_fn_outputs=[], loss_fn_output_type="", metrics={})
        await asyncio.to_thread(client.complete, request_id, result)
        assert await futures.wait_for_future(request_id, 1) == (
            RequestStatus.COMPLETED,
            result.model_dump_json(),
        )
    finally:
        client.close()
        await server.close()


def test_engine_executes_transport_batch_without_database_futures():
    completed = {}

    class FakeTransport:
        def complete(self, request_id, result):
            completed[request_id] = result

    def forward_backward(batch):
        return {
            request_id: types.ForwardBackwardOutput(loss_fn_outputs=[], loss_fn_output_type="", metrics={})
            for request_id, *_ in batch.request_batch_slices
        }

    engine = object.__new__(TinkerEngine)
    engine.backend = SimpleNamespace(
        has_model=lambda model_id: model_id == "model_a",
        forward_backward=forward_backward,
    )
    engine.operation_transport = FakeTransport()
    operations = [
        ClaimedOperation(
            request_id=-index,
            request_type=types.RequestType.FORWARD_BACKWARD,
            model_id="model_a",
            seq_id=index,
            payload=_payload(),
        )
        for index in (1, 2, 3)
    ]

    engine.process_transport_operations(operations)

    assert set(completed) == {-1, -2, -3}
    assert all(isinstance(result, types.ForwardBackwardOutput) for result in completed.values())
