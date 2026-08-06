from unittest.mock import AsyncMock, Mock, patch

from skyrl.backends.skyrl_train_backend import (
    MegatronBackendOverrides,
    SkyRLTrainBackend,
)
from skyrl.tinker import types


def _warm_backend(keep_runtime_warm: bool, model_ids: tuple[str, ...] = ("model-a",)) -> SkyRLTrainBackend:
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(keep_runtime_warm_on_last_unload=keep_runtime_warm)
    backend._model_ids_to_role = {model_id: "policy" for model_id in model_ids}
    backend._model_metadata = {
        model_id: types.ModelMetadata(
            adapter_index=0,
            lora_config=types.LoraConfig(rank=8, alpha=16, seed=0),
        )
        for model_id in model_ids
    }
    backend._cfg = Mock()
    backend._dispatch = Mock()
    backend._colocate_pg = None
    backend._inference_engine_client = Mock()
    backend._inference_engine_client.unload_lora_adapter = AsyncMock()
    backend._inference_engines_initialized = True
    backend._keep_inference_warm = False
    backend._inference_adapter_ids = set()
    backend._renderer = Mock()
    backend._render_server = None
    backend._base_lora_signature = (8, 16)
    backend._server_groups = []
    backend._inference_router = None
    backend._inference_state_publisher = None
    return backend


def test_last_lora_unload_can_keep_shared_runtime_warm():
    backend = _warm_backend(keep_runtime_warm=True)
    dispatch = backend._dispatch

    with patch("skyrl.backends.skyrl_train_backend.ray.shutdown") as shutdown:
        backend.delete_model("model-a")

    dispatch.delete_adapter.assert_called_once_with("policy", "model-a")
    shutdown.assert_not_called()
    assert backend._dispatch is dispatch
    assert backend._model_ids_to_role == {}
    assert backend._base_lora_signature == (8, 16)


def test_model_unload_removes_adapter_from_warm_inference_runtime():
    backend = _warm_backend(keep_runtime_warm=True)
    backend._inference_adapter_ids.add("model-a")

    backend.delete_model("model-a")

    backend._inference_engine_client.unload_lora_adapter.assert_awaited_once_with("model-a")
    assert backend._inference_adapter_ids == set()


def test_create_model_registers_fresh_adapter_against_warm_runtime():
    backend = _warm_backend(keep_runtime_warm=True, model_ids=())
    lora_config = types.LoraConfig(rank=8, alpha=16, seed=0)

    backend.create_model("model-b", lora_config)

    backend._dispatch.register_adapter.assert_called_once_with("policy", "model-b")
    assert backend._model_ids_to_role == {"model-b": "policy"}


def test_last_lora_unload_still_shuts_down_runtime_by_default():
    backend = _warm_backend(keep_runtime_warm=False)

    with patch("skyrl.backends.skyrl_train_backend.ray.shutdown") as shutdown:
        backend.delete_model("model-a")

    shutdown.assert_called_once_with()
    assert backend._dispatch is None
    assert backend._base_lora_signature is None
