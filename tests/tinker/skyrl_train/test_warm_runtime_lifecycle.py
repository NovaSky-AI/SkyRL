"""Unit tests for the multi-tenant warm-runtime lifecycle
(``keep_runtime_warm_on_last_unload``).

Covers issue #1654 items 1 and 3: the last LoRA unload can keep the shared
Ray runtime (workers, inference engines, base model) alive instead of calling
``ray.shutdown()``, full-parameter fine-tuning keeps the teardown behavior,
and the inference proxy URL published at engine bring-up is never cleared or
re-published in the warm lifecycle (a single upsert for the server lifetime).

CPU-only: backend state is constructed directly (no Ray, no GPU). Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest \\
    pytest tests/tinker/skyrl_train/test_warm_runtime_lifecycle.py
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    MegatronBackendOverrides,
    SkyRLTrainBackend,
)
from skyrl.tinker import types  # noqa: E402

LORA_CONFIG = types.LoraConfig(rank=8, alpha=16, seed=0)


def _backend(
    keep_runtime_warm: bool,
    model_ids: tuple[str, ...] = ("model-a",),
    lora: bool = True,
) -> SkyRLTrainBackend:
    """Build a backend in the post-create_model state without running __init__."""
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(keep_runtime_warm_on_last_unload=keep_runtime_warm)
    backend._model_ids_to_role = {model_id: "policy" for model_id in model_ids}
    backend._model_metadata = {
        model_id: types.ModelMetadata(adapter_index=0, lora_config=LORA_CONFIG) for model_id in model_ids
    }
    backend._cfg = Mock()
    backend._dispatch = Mock()
    backend._colocate_pg = None
    backend._inference_engine_client = Mock()
    backend._inference_engine_client.unload_lora_adapter = AsyncMock()
    backend._inference_engines_initialized = True
    backend._inference_adapter_ids = set()
    backend._renderer = None
    backend._render_server = None
    backend._base_lora_signature = (LORA_CONFIG.rank, int(LORA_CONFIG.alpha)) if lora else None
    backend._server_groups = []
    backend._inference_router = None
    backend._inference_state_publisher = Mock()
    return backend


def test_last_lora_unload_keeps_runtime_warm_when_enabled():
    backend = _backend(keep_runtime_warm=True)
    dispatch = backend._dispatch

    with patch("skyrl.backends.skyrl_train_backend.ray.shutdown") as shutdown:
        backend.delete_model("model-a")

    shutdown.assert_not_called()
    dispatch.delete_adapter.assert_called_once_with("policy", "model-a")
    assert backend._dispatch is dispatch
    assert backend._model_ids_to_role == {}
    assert backend._base_lora_signature == (8, 16)
    assert backend._inference_engines_initialized
    # Item 3: the proxy URL published at engine bring-up stays valid — the
    # warm path never clears (None) or re-publishes it.
    backend._inference_state_publisher.assert_not_called()


def test_keep_runtime_warm_defaults_to_true():
    assert MegatronBackendOverrides().keep_runtime_warm_on_last_unload is True


def test_last_lora_unload_tears_down_when_disabled():
    backend = _backend(keep_runtime_warm=False)

    with patch("skyrl.backends.skyrl_train_backend.ray.shutdown") as shutdown:
        backend.delete_model("model-a")

    shutdown.assert_called_once_with()
    assert backend._dispatch is None
    assert backend._base_lora_signature is None
    backend._inference_state_publisher.assert_called_once_with(None)


def test_fft_unload_tears_down_even_when_warm_enabled():
    backend = _backend(keep_runtime_warm=True, lora=False)

    with patch("skyrl.backends.skyrl_train_backend.ray.shutdown") as shutdown:
        backend.delete_model("model-a")

    shutdown.assert_called_once_with()
    assert backend._dispatch is None


def test_delete_unloads_synced_adapter_from_inference_engines():
    backend = _backend(keep_runtime_warm=True)
    backend._inference_adapter_ids.add("model-a")

    backend.delete_model("model-a")

    backend._inference_engine_client.unload_lora_adapter.assert_awaited_once_with("model-a")
    assert backend._inference_adapter_ids == set()


def test_delete_skips_inference_unload_for_unsynced_adapter():
    backend = _backend(keep_runtime_warm=True)

    backend.delete_model("model-a")

    backend._inference_engine_client.unload_lora_adapter.assert_not_awaited()


def test_delete_proceeds_when_inference_unload_fails():
    backend = _backend(keep_runtime_warm=True)
    backend._inference_adapter_ids.add("model-a")
    backend._inference_engine_client.unload_lora_adapter.side_effect = RuntimeError("vLLM unreachable")

    backend.delete_model("model-a")

    backend._dispatch.delete_adapter.assert_called_once_with("policy", "model-a")
    assert backend._model_ids_to_role == {}
    assert backend._inference_adapter_ids == set()


def test_create_model_registers_fresh_adapter_against_warm_runtime():
    backend = _backend(keep_runtime_warm=True, model_ids=())

    backend.create_model("model-b", LORA_CONFIG)

    backend._dispatch.register_adapter.assert_called_once_with("policy", "model-b")
    assert backend._model_ids_to_role == {"model-b": "policy"}


def test_create_model_signature_mismatch_rejected_against_warm_runtime():
    backend = _backend(keep_runtime_warm=True, model_ids=())

    with pytest.raises(ValueError, match="LoRA signature mismatch"):
        backend.create_model("model-b", types.LoraConfig(rank=16, alpha=32, seed=0))
