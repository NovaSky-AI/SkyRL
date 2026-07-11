"""CPU tests for the R3 stash endpoints on the vLLM server actor.

Exercises the custom FastAPI endpoints (`/skyrl/v1/completions`,
`/skyrl/v1/routed_experts/{fetch,weight_sync,clear}`) against a fake
completions handler, without starting a vLLM engine: the wrapper must stash
each choice's routing keyed by (model, prompt + response tokens), strip it
from the response, and the fetch endpoint must return exactly the requested
digests as an npz payload. Requires vLLM importable (no GPU). Run:
  uv run --extra dev --extra fsdp pytest tests/backends/skyrl_train/inference_servers/test_routed_experts_stash_endpoints.py
"""

from __future__ import annotations

import base64
import io
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

vllm_server_actor = pytest.importorskip("skyrl.backends.skyrl_train.inference_servers.vllm_server_actor")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from vllm.entrypoints.openai.completion.protocol import (  # noqa: E402
    CompletionResponse,
    CompletionResponseChoice,
)
from vllm.entrypoints.openai.engine.protocol import (  # noqa: E402
    ErrorInfo,
    ErrorResponse,
    UsageInfo,
)

from skyrl.backends.skyrl_train.inference_servers.routed_experts_stash import (  # noqa: E402
    load_arrays_npz,
    sequence_digest,
)

PROMPT_TOKENS = [10, 11, 12]


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _routing(num_tokens: int, num_layers: int = 2, topk: int = 3) -> np.ndarray:
    return np.arange(num_tokens * num_layers * topk, dtype=np.uint8).reshape(num_tokens, num_layers, topk)


def _completion_response(model: str, choices: list[CompletionResponseChoice]) -> CompletionResponse:
    return CompletionResponse(
        model=model,
        choices=choices,
        usage=UsageInfo(prompt_tokens=len(PROMPT_TOKENS), completion_tokens=0, total_tokens=len(PROMPT_TOKENS)),
    )


def _choice(index: int, tokens: list[int], routed_experts_b64: str | None) -> CompletionResponseChoice:
    return CompletionResponseChoice(
        index=index,
        text="",
        finish_reason="stop",
        prompt_token_ids=list(PROMPT_TOKENS),
        token_ids=tokens,
        routed_experts=routed_experts_b64,
    )


class _FakeServingCompletion:
    """Stands in for app.state.openai_serving_completion."""

    def __init__(self, result):
        self.result = result
        self.requests = []

    async def create_completion(self, request, raw_request):
        self.requests.append(request)
        # Fresh object per call, like the real handler (the wrapper strips
        # routing from the response it returns).
        return self.result.model_copy(deep=True)


def _make_client(result, enable_return_routed_experts: bool = True) -> tuple[TestClient, FastAPI]:
    app = FastAPI()
    cli_args = Namespace(enable_return_routed_experts=enable_return_routed_experts, enable_lora=False)
    vllm_server_actor.VLLMServerActor._add_custom_endpoints(app, engine=SimpleNamespace(), cli_args=cli_args)
    app.state.openai_serving_completion = _FakeServingCompletion(result)
    return TestClient(app), app


def test_completions_wrapper_stashes_and_strips_routing():
    routing_a = _routing(len(PROMPT_TOKENS) + 2 - 1)
    routing_b = _routing(len(PROMPT_TOKENS) + 3 - 1) + 1
    result = _completion_response(
        "model_a",
        [_choice(0, [20, 21], _npy_b64(routing_a)), _choice(1, [30, 31, 32], _npy_b64(routing_b))],
    )
    client, app = _make_client(result)

    resp = client.post("/skyrl/v1/completions", json={"model": "model_a", "prompt": PROMPT_TOKENS, "n": 2})
    assert resp.status_code == 200
    body = resp.json()

    # Response carries token ids but no routing.
    assert [c["token_ids"] for c in body["choices"]] == [[20, 21], [30, 31, 32]]
    assert all(c["routed_experts"] is None for c in body["choices"])
    # return_token_ids was forced so the stash could be keyed.
    assert app.state.openai_serving_completion.requests[0].return_token_ids is True

    # Each choice's routing is stashed under (model, prompt + response tokens).
    stash = app.state.skyrl_routed_experts_stash
    hits = stash.get_many(
        "model_a",
        [sequence_digest(PROMPT_TOKENS + [20, 21]).hex(), sequence_digest(PROMPT_TOKENS + [30, 31, 32]).hex()],
    )
    assert len(hits) == 2
    np.testing.assert_array_equal(hits[sequence_digest(PROMPT_TOKENS + [20, 21]).hex()], routing_a)
    np.testing.assert_array_equal(hits[sequence_digest(PROMPT_TOKENS + [30, 31, 32]).hex()], routing_b)


def test_completions_wrapper_passthrough_when_capture_disabled():
    result = _completion_response("m", [_choice(0, [20, 21], None)])
    client, app = _make_client(result, enable_return_routed_experts=False)

    resp = client.post("/skyrl/v1/completions", json={"model": "m", "prompt": PROMPT_TOKENS})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["token_ids"] == [20, 21]
    assert app.state.skyrl_routed_experts_stash is None


def test_completions_wrapper_survives_bad_routing_payload():
    result = _completion_response("m", [_choice(0, [20, 21], "!!! not npy !!!")])
    client, app = _make_client(result)

    resp = client.post("/skyrl/v1/completions", json={"model": "m", "prompt": PROMPT_TOKENS})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["routed_experts"] is None
    assert len(app.state.skyrl_routed_experts_stash) == 0


def test_completions_wrapper_rejects_streaming_and_bad_requests():
    result = _completion_response("m", [_choice(0, [20], None)])
    client, _ = _make_client(result)
    assert client.post("/skyrl/v1/completions", json={"model": "m", "prompt": [1], "stream": True}).status_code == 400
    assert client.post("/skyrl/v1/completions", json={"model": "m", "prompt": [1], "n": "bogus"}).status_code == 400


def test_completions_wrapper_forwards_handler_errors():
    error = ErrorResponse(error=ErrorInfo(message="model not found", type="NotFoundError", code=404))
    client, _ = _make_client(error)
    resp = client.post("/skyrl/v1/completions", json={"model": "missing", "prompt": [1]})
    assert resp.status_code == 404
    assert resp.json()["error"]["message"] == "model not found"


def test_fetch_returns_requested_digests_as_npz():
    routing = _routing(4)
    result = _completion_response("model_a", [_choice(0, [20, 21], _npy_b64(routing))])
    client, app = _make_client(result)
    client.post("/skyrl/v1/completions", json={"model": "model_a", "prompt": PROMPT_TOKENS})

    digest_hex = sequence_digest(PROMPT_TOKENS + [20, 21]).hex()
    resp = client.post(
        "/skyrl/v1/routed_experts/fetch",
        json={"model": "model_a", "digests": [digest_hex, sequence_digest([9]).hex()]},
    )
    assert resp.status_code == 200
    hits = load_arrays_npz(resp.content)
    assert set(hits) == {digest_hex}
    np.testing.assert_array_equal(hits[digest_hex], routing)

    # Wrong model name misses (stash is scoped per model).
    resp = client.post("/skyrl/v1/routed_experts/fetch", json={"model": "other", "digests": [digest_hex]})
    assert load_arrays_npz(resp.content) == {}


def test_weight_sync_and_clear_endpoints_drive_stash_lifecycle():
    routing = _routing(4)
    result = _completion_response("model_a", [_choice(0, [20, 21], _npy_b64(routing))])
    client, app = _make_client(result)
    client.post("/skyrl/v1/completions", json={"model": "model_a", "prompt": PROMPT_TOKENS})
    digest_hex = sequence_digest(PROMPT_TOKENS + [20, 21]).hex()

    # First sync with max_staleness=1: entry survives; second sync drops it.
    resp = client.post("/skyrl/v1/routed_experts/weight_sync", json={"model": "model_a", "max_staleness": 1})
    assert resp.json() == {"removed": 0}
    resp = client.post("/skyrl/v1/routed_experts/weight_sync", json={"model": "model_a", "max_staleness": 1})
    assert resp.json() == {"removed": 1}
    assert (
        load_arrays_npz(
            client.post("/skyrl/v1/routed_experts/fetch", json={"model": "model_a", "digests": [digest_hex]}).content
        )
        == {}
    )

    # Clear drops everything for the model.
    client.post("/skyrl/v1/completions", json={"model": "model_a", "prompt": PROMPT_TOKENS})
    assert client.post("/skyrl/v1/routed_experts/clear", json={"model": "model_a"}).json() == {"removed": 1}


def test_r3_endpoints_are_noops_without_capture():
    result = _completion_response("m", [_choice(0, [20], None)])
    client, _ = _make_client(result, enable_return_routed_experts=False)
    assert client.post("/skyrl/v1/routed_experts/fetch", json={"model": "m", "digests": []}).status_code == 200
    assert client.post("/skyrl/v1/routed_experts/weight_sync", json={"model": "m"}).json() == {"removed": 0}
    assert client.post("/skyrl/v1/routed_experts/clear", json={"model": "m"}).json() == {"removed": 0}
