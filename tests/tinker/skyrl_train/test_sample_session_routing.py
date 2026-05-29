"""Verifies the new-inference sample path forwards the routing key.

``_sample_with_remote_client`` puts the derived routing session id into the
request body as ``session_id``, which ``RemoteInferenceClient.sample`` lifts
into the ``X-Session-ID`` header. No inference engines are brought up, so this
runs on CPU. Requires the SkyRL-Train backend deps (ray/vllm). Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_sample_session_routing.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_sample_batch  # noqa: E402

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


class _SpyClient:
    """Captures the request payloads passed to RemoteInferenceClient.sample."""

    def __init__(self):
        self.payloads = []

    async def sample(self, request_payload):
        self.payloads.append(request_payload)
        return {}  # ignored; _aggregate_sample_results is stubbed below

    async def aclose(self):
        pass


def _sample_input(**kwargs) -> types.SampleInput:
    return types.SampleInput(
        base_model=BASE_MODEL,
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=4, seed=0),
        num_samples=1,
        checkpoint_id="",
        prompt_logprobs=False,
        **kwargs,
    )


def test_sample_with_remote_client_sets_session_id(monkeypatch):
    """Test that session_id is set correctly if routing key is present"""
    monkeypatch.setattr(skyrl_train_backend, "resolve_policy_model_name", lambda cfg: BASE_MODEL)

    spy = _SpyClient()
    fake_self = SimpleNamespace(
        _cfg=None,
        _base_lora_signature=None,
        _model_ids_to_role={},
        _inference_engine_client=spy,
        _aggregate_sample_results=lambda prepared_batch, outputs: {},
    )
    sample = skyrl_train_backend.SkyRLTrainBackend._sample_with_remote_client

    batch_with_session = prepare_sample_batch(
        {"req": ("", _sample_input(sampling_session_id="sampling_abcd", seq_id=7))}
    )
    sample(fake_self, batch_with_session)
    assert len(spy.payloads) == 1
    assert spy.payloads[0]["json"]["session_id"] == "sampling_abcd:7"

    spy.payloads.clear()
    batch_without_session = prepare_sample_batch({"req": ("", _sample_input())})
    sample(fake_self, batch_without_session)
    assert len(spy.payloads) == 1
    assert "session_id" not in spy.payloads[0]["json"]
