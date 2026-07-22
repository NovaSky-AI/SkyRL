"""Verifies the training (forward/forward_backward) path does not bring up
inference engines for text-only batches.

``_to_training_batch`` only needs vLLM's render endpoint for image chunks;
text-only batches — the SFT code path — must render locally so pure training
runs never pay for inference-engine startup. No inference engines are brought
up, so this runs on CPU. Requires the SkyRL-Train backend deps (ray/vllm). Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest pytest tests/tinker/skyrl_train/test_text_only_batch_no_inference.py
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.tinker import types  # noqa: E402
from skyrl.tinker.engine import prepare_model_pass_batch  # noqa: E402

PAD_TOKEN_ID = 0


class _EngineInitCalled(Exception):
    """Sentinel raised by the stubbed _ensure_inference_engines."""


class _RenderServerUsed(Exception):
    """Sentinel raised by the stubbed _create_render_client."""


def _fake_backend() -> SimpleNamespace:
    def _ensure_inference_engines():
        raise _EngineInitCalled

    def _create_render_client():
        raise _RenderServerUsed

    return SimpleNamespace(
        _renderer=None,
        _inference_engine_client=None,
        _cfg=None,
        _tokenizer=SimpleNamespace(pad_token_id=PAD_TOKEN_ID),
        _ensure_inference_engines=_ensure_inference_engines,
        _create_render_client=_create_render_client,
    )


def _prepared_batch(model_input: types.ModelInput) -> types.PreparedModelPassBatch:
    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=[]),
            logprobs=types.TensorData(data=[]),
        ),
    )
    requests = {"req1": ("model1", types.ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"))}
    return prepare_model_pass_batch(requests)


def test_text_only_batch_skips_inference_engines():
    """Text-only (SFT) batches render locally without touching inference engines."""
    fake_self = _fake_backend()
    prepared_batch = _prepared_batch(types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]))

    batch = skyrl_train_backend.SkyRLTrainBackend._to_training_batch(fake_self, prepared_batch, role="policy")

    # Full sequence = input tokens + last target token (SkyRL-Train shifts internally).
    assert batch["sequences"].tolist() == [[1, 2, 3, 4]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert fake_self._renderer is None


def test_image_batch_uses_render_server_not_engines():
    """Batches with image chunks go to the CPU render server, never the engines."""
    fake_self = _fake_backend()
    image_chunk = types.ImageChunk(data=base64.b64encode(b"not-a-real-png"), format="png")
    prepared_batch = _prepared_batch(types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3]), image_chunk]))

    with pytest.raises(_RenderServerUsed):
        skyrl_train_backend.SkyRLTrainBackend._to_training_batch(fake_self, prepared_batch, role="policy")


def test_render_client_prefers_engine_client_when_initialized():
    """Once engines are up (RL), rendering reuses the engine client instead of a CPU server."""
    engine_client = object()
    fake_self = SimpleNamespace(
        _inference_engines_initialized=True,
        _inference_engine_client=engine_client,
        _render_server=None,
        _cfg=None,
    )
    client = skyrl_train_backend.SkyRLTrainBackend._create_render_client(fake_self)
    assert client is engine_client
    assert fake_self._render_server is None


def test_engine_init_invalidates_cpu_render_state():
    """When engines come up, the CPU-render-backed renderer and server are dropped
    so the next image batch rebuilds against the engine client."""

    class _RenderServerStub:
        def __init__(self):
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    render_server = _RenderServerStub()
    fake_self = SimpleNamespace(
        _inference_engines_initialized=False,
        _inference_engine_client=object(),
        _create_new_inference_client=lambda: None,
        _dispatch=SimpleNamespace(set_inference_engine_client=lambda client: None),
        init_weight_sync_state=lambda: None,
        _renderer=object(),
        _render_server=render_server,
    )

    skyrl_train_backend.SkyRLTrainBackend._ensure_inference_engines(fake_self)

    assert fake_self._inference_engines_initialized
    assert fake_self._renderer is None
    assert render_server.shutdown_called
    assert fake_self._render_server is None
