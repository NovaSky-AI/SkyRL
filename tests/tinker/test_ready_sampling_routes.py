from types import SimpleNamespace

import pytest
from starlette.requests import Request

from skyrl.tinker import api, types


class _CapturingFutureStore:
    def __init__(self) -> None:
        self.created = None

    def create(self, request_type, model_id, request_data, *, seq_id=None):
        self.created = (request_type, model_id, request_data, seq_id)
        return 42


class _NoDatabaseSession:
    async def commit(self) -> None:
        raise AssertionError("a cached sampling route must not check out a database connection")


def test_ready_sampling_routes_evict_the_least_recently_used_entry() -> None:
    routes = api.ReadySamplingRoutes(max_entries=2)
    routes.remember("sampling-a", "model-a", "checkpoint-a")
    routes.remember("sampling-b", "model-b", "checkpoint-b")
    assert routes.get("sampling-a") == api.ReadySamplingRoute("model-a", "checkpoint-a")

    routes.remember("sampling-c", "model-c", "checkpoint-c")

    assert routes.get("sampling-b") is None
    assert routes.get("sampling-a") is not None
    assert routes.get("sampling-c") is not None


@pytest.mark.asyncio
async def test_cached_sampling_route_keeps_asample_off_the_database() -> None:
    routes = api.ReadySamplingRoutes()
    routes.remember("sampling-ready", "model-ready", "checkpoint-ready")
    futures = _CapturingFutureStore()
    app = SimpleNamespace(
        state=SimpleNamespace(
            ready_sampling_routes=routes,
            external_future_store=futures,
            external_inference_client=None,
        )
    )
    req = Request({"type": "http", "app": app})
    sample = api.SampleRequest(
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=api.SamplingParams(max_tokens=4, seed=0),
        sampling_session_id="sampling-ready",
        seq_id=7,
    )

    response = await api.asample(sample, req, _NoDatabaseSession())

    assert response.request_id == "42"
    request_type, model_id, sample_input, seq_id = futures.created
    assert request_type == types.RequestType.EXTERNAL
    assert model_id == "model-ready"
    assert sample_input.checkpoint_id == "checkpoint-ready"
    assert seq_id == 7
