"""Forwarded sample results are encoded to proto once and served as-is.

The forwarding client decodes the vLLM body and encodes straight to the
``SampleResponse`` wire form; no pydantic model or JSON text is built. These
tests pin the fast path to the validated path byte for byte.
"""

import asyncio
from types import SimpleNamespace

import pytest

from skyrl.tinker import api, types
from skyrl.tinker.db_models import RequestStatus
from skyrl.tinker.external_future_store import ExternalFutureStore
from skyrl.tinker.proto_serialization import (
    PROTO_CONTENT_TYPE,
    serialize_result,
    serialize_sample_output,
)

SEQUENCES = [
    ("length", [1, 2, 3, 40000], [-0.5, -1.25, -0.03125, -7.0]),
    ("stop", [7], [-2.0]),
]
PROMPT_LOGPROBS = [None, -0.75, -3.5]
TOPK = [None, [(11, -0.5), (12, -1.5)], [(13, -0.25)]]


def _validated_bytes(prompt_logprobs=None, topk=None) -> bytes:
    output = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason=s, tokens=t, logprobs=lp) for s, t, lp in SEQUENCES],
        prompt_logprobs=prompt_logprobs,
        topk_prompt_logprobs=topk,
    )
    return serialize_result(types.RequestType.SAMPLE, output.model_dump())


def test_fast_path_matches_validated_serialization_bytes():
    assert serialize_sample_output(SEQUENCES, None, None) == _validated_bytes()
    assert serialize_sample_output(SEQUENCES, PROMPT_LOGPROBS, TOPK) == _validated_bytes(PROMPT_LOGPROBS, TOPK)


@pytest.mark.asyncio
async def test_store_keeps_proto_bytes_as_is():
    store = ExternalFutureStore()
    request_id = store.create("model_a", SimpleNamespace())
    proto = serialize_sample_output(SEQUENCES, None, None)

    await store.complete(request_id, proto, RequestStatus.COMPLETED)

    status, request_type, result_data = await store.wait(request_id, timeout=1)
    assert (status, request_type, result_data) == (RequestStatus.COMPLETED, types.RequestType.EXTERNAL, None)
    assert store.proto_result(request_id) is proto


@pytest.mark.asyncio
async def test_store_still_accepts_pydantic_results():
    store = ExternalFutureStore()
    request_id = store.create("model_a", SimpleNamespace())
    output = types.SampleOutput(sequences=[])

    await store.complete(request_id, output, RequestStatus.COMPLETED)

    assert (await store.wait(request_id, timeout=1))[2] == output.model_dump_json()
    assert store.proto_result(request_id) is None


def test_store_ttl_overrides_apply_per_instance():
    store = ExternalFutureStore(retrieved_ttl_sec=5.0, completed_ttl_sec=7.0)
    assert (store._RETRIEVED_TTL_SECONDS, store._COMPLETED_TTL_SECONDS) == (5.0, 7.0)
    assert ExternalFutureStore._RETRIEVED_TTL_SECONDS == 300.0
    assert ExternalFutureStore()._RETRIEVED_TTL_SECONDS == 300.0


async def _connected() -> bool:
    return False


def _request(store: ExternalFutureStore, accept: str, serialize_calls: list) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_future_store=store,
                future_waiters={},
                proto_serialization_lock=asyncio.Lock(),
            )
        ),
        headers={"accept": accept},
        is_disconnected=_connected,
    )


@pytest.mark.asyncio
async def test_retrieve_future_passes_stored_proto_through_without_reencoding(monkeypatch):
    store = ExternalFutureStore()
    request_id = store.create("model_a", SimpleNamespace())
    proto = serialize_sample_output(SEQUENCES, None, None)
    await store.complete(request_id, proto, RequestStatus.COMPLETED)
    serialize_calls: list = []
    monkeypatch.setattr(api, "_serialize_proto_result", lambda *a: serialize_calls.append(a) or b"unexpected")

    response = await api.retrieve_future(
        api.RetrieveFutureRequest(request_id=str(request_id)), _request(store, PROTO_CONTENT_TYPE, serialize_calls)
    )

    assert response.media_type == PROTO_CONTENT_TYPE
    assert response.body == proto
    assert serialize_calls == []


@pytest.mark.asyncio
async def test_retrieve_future_encodes_json_stored_result_once_for_proto_clients(monkeypatch):
    store = ExternalFutureStore()
    request_id = store.create("model_a", SimpleNamespace())
    output = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason=s, tokens=t, logprobs=lp) for s, t, lp in SEQUENCES]
    )
    await store.complete(request_id, output, RequestStatus.COMPLETED)
    calls: list = []
    real = api._serialize_proto_result
    monkeypatch.setattr(api, "_serialize_proto_result", lambda *a: calls.append(a) or real(*a))
    request = _request(store, PROTO_CONTENT_TYPE, calls)

    first = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), request)
    second = await api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), request)

    assert first.body == second.body == serialize_sample_output(SEQUENCES, None, None)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_retrieve_future_overlapping_proto_waiters_encode_once(monkeypatch):
    """Concurrent polls for one JSON-stored result must not each re-encode it."""
    store = ExternalFutureStore()
    request_id = store.create("model_a", SimpleNamespace())
    output = types.SampleOutput(
        sequences=[types.GeneratedSequence(stop_reason=s, tokens=t, logprobs=lp) for s, t, lp in SEQUENCES]
    )
    await store.complete(request_id, output, RequestStatus.COMPLETED)
    calls: list = []
    real = api._serialize_proto_result
    monkeypatch.setattr(api, "_serialize_proto_result", lambda *a: calls.append(a) or real(*a))
    request = _request(store, PROTO_CONTENT_TYPE, calls)

    responses = await asyncio.gather(
        *(api.retrieve_future(api.RetrieveFutureRequest(request_id=str(request_id)), request) for _ in range(8))
    )

    assert {r.body for r in responses} == {serialize_sample_output(SEQUENCES, None, None)}
    assert len(calls) == 1
