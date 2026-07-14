from unittest.mock import AsyncMock, MagicMock

import pytest

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")
SkyRLTrainBackend = skyrl_train_backend.SkyRLTrainBackend


def test_training_only_export_does_not_start_inference():
    backend = object.__new__(SkyRLTrainBackend)
    backend._validate_model_state = MagicMock()
    backend._get_role = MagicMock(return_value="policy")
    backend._base_lora_signature = (32, 32.0)
    backend._dispatch = MagicMock()
    backend._dispatch.export_adapter = AsyncMock(return_value="/shared/adapters/model_step_4")
    backend._ensure_inference_engines = MagicMock()

    adapter_path = backend.export_adapter("model", "model_step_4")

    backend._dispatch.export_adapter.assert_awaited_once_with("model", "model_step_4")
    backend._ensure_inference_engines.assert_not_called()
    assert adapter_path == "/shared/adapters/model_step_4"


def test_loading_adapter_advances_inference_weight_version():
    backend = object.__new__(SkyRLTrainBackend)
    backend._ensure_inference_engines = MagicMock()
    backend._inference_engine_client = MagicMock()
    backend._inference_engine_client.load_lora_adapter = AsyncMock()

    backend.load_adapter("rollout_model", "/shared/adapters/model_step_4")

    backend._ensure_inference_engines.assert_called_once_with()
    backend._inference_engine_client.load_lora_adapter.assert_awaited_once_with(
        "rollout_model", "/shared/adapters/model_step_4"
    )
    backend._inference_engine_client.increment_weight_version.assert_called_once_with()
