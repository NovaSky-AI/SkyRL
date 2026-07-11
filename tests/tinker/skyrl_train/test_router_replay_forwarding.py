"""CPU tests for R3 routing capture on the API-side sample-forwarding path.

In non-colocated mode the API process forwards sample requests directly to
the SkyRL-Train-managed vLLM (``SkyRLTrainInferenceForwardingClient``), so
the rollout routing vLLM attaches to ``/v1/completions`` choices surfaces in
the API process rather than in the training backend. These tests verify the
forwarding client decodes each choice's base64 ``.npy`` routing and spools it
keyed by ``(model_id, prompt + response tokens)`` — the exact key the backend
reconstructs in forward_backward — without touching the client-facing sample
output. No inference engines or GPUs needed. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_router_replay_forwarding.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest

forwarding = pytest.importorskip("skyrl.tinker.extra.skyrl_train_inference_forwarding")

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.config import EngineConfig  # noqa: E402
from skyrl.tinker.routed_experts_spool import RoutedExpertsSpool  # noqa: E402

PROMPT_TOKENS = [10, 11, 12]


def _npy_b64(arr: np.ndarray) -> str:
    """Encode an array the way vLLM's /v1/completions does (base64 .npy)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _routing(num_tokens: int, num_layers: int = 2, topk: int = 3) -> np.ndarray:
    return np.arange(num_tokens * num_layers * topk, dtype=np.int32).reshape(num_tokens, num_layers, topk)


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def json(self):
        return self._payload


class _FakeHTTPClient:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    async def post(self, url, json=None, headers=None):
        self.requests.append((url, json, headers))
        return _FakeResponse(self.payload)

    async def aclose(self):
        pass


def _make_client(tmp_path, payload) -> forwarding.SkyRLTrainInferenceForwardingClient:
    engine_config = EngineConfig(
        base_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
        backend="megatron",
        database_url=f"sqlite:///{tmp_path / 'tinker.db'}",
        routed_experts_spool_dir=str(tmp_path / "spool"),
    )
    client = forwarding.SkyRLTrainInferenceForwardingClient(engine_config, db_engine=None)
    # Swap the real pool for a canned-response fake (close the real one first).
    asyncio.run(client._http_client.aclose())
    client._http_client = _FakeHTTPClient(payload)
    return client


def _sample_req(num_samples: int = 1):
    prompt = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=list(PROMPT_TOKENS))])
    return SimpleNamespace(
        prompt=SimpleNamespace(to_types=lambda: prompt),
        sampling_params=SimpleNamespace(seed=0, max_tokens=8, temperature=1.0, top_p=1.0, top_k=-1, stop=None),
        num_samples=num_samples,
        sampling_session_id=None,
        seq_id=None,
    )


def _choice(tokens: list[int], routed_experts_b64: str | None) -> dict:
    return {
        "token_ids": tokens,
        "logprobs": {"token_logprobs": [-0.1] * len(tokens)},
        "finish_reason": "stop",
        "routed_experts": routed_experts_b64,
    }


def test_forward_spools_routing_per_choice(tmp_path):
    """Each choice's routing lands in the spool under (model_id, prompt + choice tokens)."""
    # Routing covers prompt + response minus the last sampled token.
    routing_a = _routing(num_tokens=len(PROMPT_TOKENS) + 2 - 1)
    routing_b = _routing(num_tokens=len(PROMPT_TOKENS) + 3 - 1) + 1
    payload = {"choices": [_choice([20, 21], _npy_b64(routing_a)), _choice([30, 31, 32], _npy_b64(routing_b))]}
    client = _make_client(tmp_path, payload)

    out = asyncio.run(client._forward("http://proxy", _sample_req(num_samples=2), "model_a", base_model=None))

    # Client-facing output is unchanged by R3 (no new fields, tokens intact).
    assert [seq.tokens for seq in out.sequences] == [[20, 21], [30, 31, 32]]
    assert all(isinstance(seq, types.GeneratedSequence) for seq in out.sequences)

    # The backend-visible spool holds one file per choice, keyed by the full
    # sampled sequence (prompt + response) exactly as forward_backward keys it.
    spool = RoutedExpertsSpool(str(tmp_path / "spool"))
    np.testing.assert_array_equal(spool.consume("model_a", PROMPT_TOKENS + [20, 21]), routing_a)
    np.testing.assert_array_equal(spool.consume("model_a", PROMPT_TOKENS + [30, 31, 32]), routing_b)
    assert os.listdir(spool.root) == []


def test_forward_skips_spool_for_base_model_samples(tmp_path):
    """Base-model samples have no training model to replay into — nothing is spooled."""
    payload = {"choices": [_choice([20, 21], _npy_b64(_routing(4)))]}
    client = _make_client(tmp_path, payload)

    out = asyncio.run(
        client._forward(
            "http://proxy", _sample_req(), model_id="", base_model="trl-internal-testing/tiny-Qwen3ForCausalLM"
        )
    )

    assert len(out.sequences) == 1
    assert not os.path.exists(str(tmp_path / "spool"))


def test_forward_without_routing_writes_nothing(tmp_path):
    """When vLLM was launched without routing capture, choices carry no routing."""
    payload = {"choices": [_choice([20, 21], None)]}
    client = _make_client(tmp_path, payload)

    out = asyncio.run(client._forward("http://proxy", _sample_req(), "model_a", base_model=None))

    assert len(out.sequences) == 1
    assert not os.path.exists(str(tmp_path / "spool"))


def test_forward_survives_bad_routing_payload(tmp_path):
    """A malformed routing payload must never fail the sample response."""
    payload = {"choices": [_choice([20, 21], "!!! not base64 !!!")]}
    client = _make_client(tmp_path, payload)

    out = asyncio.run(client._forward("http://proxy", _sample_req(), "model_a", base_model=None))

    assert [seq.tokens for seq in out.sequences] == [[20, 21]]
    # Whatever ended up on disk (if anything) is rejected at consume time.
    spool = RoutedExpertsSpool(str(tmp_path / "spool"))
    assert spool.consume("model_a", PROMPT_TOKENS + [20, 21]) is None
