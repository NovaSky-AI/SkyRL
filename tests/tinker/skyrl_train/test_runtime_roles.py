from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    MegatronBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
)
from skyrl.tinker import types  # noqa: E402


def test_trainer_runtime_rejects_sampling():
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(runtime_role="trainer")
    backend._inference_engines_initialized = False

    with pytest.raises(RuntimeError, match="trainer-only"):
        backend._ensure_inference_engines()


@pytest.mark.parametrize("runtime_role", ["trainer", "inference"])
def test_single_role_does_not_create_colocated_gpu_pool(runtime_role):
    config = _build_skyrl_train_config("Qwen/Qwen3-0.6B", MegatronBackendOverrides(runtime_role=runtime_role))

    assert config.trainer.placement.colocate_all is False


def test_inference_runtime_starts_without_trainer_dispatch():
    backend = object.__new__(SkyRLTrainBackend)
    backend.base_model = "Qwen/Qwen3-0.6B"
    backend.config = MegatronBackendOverrides(runtime_role="inference")
    backend._cfg = None
    backend._inference_engines_initialized = False
    backend._dispatch = None
    backend._create_new_inference_client = Mock()
    backend.init_weight_sync_state = Mock()
    backend._renderer = None
    backend._render_server = None

    cfg = Mock()
    with (
        patch("skyrl.backends.skyrl_train_backend._build_skyrl_train_config", return_value=cfg),
        patch("skyrl.backends.skyrl_train_backend.ray.is_initialized", return_value=True),
    ):
        backend._ensure_inference_engines()

    assert backend._inference_engines_initialized
    assert backend._cfg is cfg
    backend.init_weight_sync_state.assert_not_called()


def test_inference_runtime_rejects_training():
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(runtime_role="inference")

    with pytest.raises(RuntimeError, match="inference-only"):
        backend.forward(SimpleNamespace(all_model_inputs=[]))


def test_inference_runtime_rejects_lora_models():
    backend = object.__new__(SkyRLTrainBackend)
    backend.config = MegatronBackendOverrides(runtime_role="inference")
    backend._model_ids_to_role = {}

    with pytest.raises(ValueError, match="unavailable"):
        backend.create_model("adapter-a", types.LoraConfig(rank=8, alpha=16, seed=0))
