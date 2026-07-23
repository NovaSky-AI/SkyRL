"""Regression coverage for retaining service-owned inference after model unload."""

import pytest

backend_module = pytest.importorskip("skyrl.backends.skyrl_train_backend")


def test_delete_model_keeps_prewarmed_inference():
    backend = object.__new__(backend_module.SkyRLTrainBackend)

    class Dispatch:
        stopped = False

        def shutdown(self):
            self.stopped = True

    dispatch = Dispatch()
    backend._model_ids_to_role = {"model": "policy"}
    backend._model_metadata = {"model": object()}
    backend._keep_inference_warm = True
    backend._dispatch = dispatch
    backend._cfg = object()
    backend._renderer = None
    backend._colocate_pg = object()
    backend._base_lora_signature = (8, 16)
    backend._inference_engine_client = object()
    backend._inference_engines_initialized = True
    backend._server_groups = []
    backend._inference_router = None

    backend.delete_model("model")

    assert dispatch.stopped
    assert backend._model_ids_to_role == {}
    assert backend._inference_engines_initialized
    assert backend._inference_engine_client is not None
