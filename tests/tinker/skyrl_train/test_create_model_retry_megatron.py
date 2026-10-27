"""A first create_model that fails after the policy weights load leaves no GPU reserved, so a retry on the
same backend succeeds (#2482).

GPU-gated. Runs the Megatron backend in-process rather than behind a Tinker server, so the test can make
the first build fail.

Run with:
  uv run --extra tinker --extra megatron --with pytest --with pytest-timeout \\
    pytest -s tests/tinker/skyrl_train/test_create_model_retry_megatron.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

cuda_available = False
try:  # pragma: no cover - import guard
    import torch

    cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
except Exception:
    cuda_available = False

pytestmark = pytest.mark.skipif(not cuda_available, reason="needs at least one CUDA GPU")

skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

import ray  # noqa: E402

from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup  # noqa: E402
from skyrl.backends.skyrl_train_backend import (  # noqa: E402
    MegatronBackendOverrides,
    SkyRLTrainBackend,
)
from skyrl.tinker import types  # noqa: E402
from tests.tinker.skyrl_train.test_multi_lora_megatron import BASE_MODEL  # noqa: E402

LORA_CONFIG = types.LoraConfig(rank=8, alpha=16, seed=0)


def test_create_model_succeeds_after_a_failed_first_build(monkeypatch):
    num_gpus = torch.cuda.device_count()
    # The colocated placement group takes every GPU, so if the failed attempt's were still reserved,
    # the retry's could not be granted and would time out.
    config = MegatronBackendOverrides(
        **{
            "trainer.placement.policy_num_gpus_per_node": num_gpus,
            "trainer.placement.policy_num_nodes": 1,
            "trainer.placement.colocate_all": True,
            "trainer.policy.megatron_config.tensor_model_parallel_size": 1,
            "trainer.policy.megatron_config.pipeline_model_parallel_size": 1,
            "generator.inference_engine.num_engines": num_gpus,
            "generator.inference_engine.tensor_parallel_size": 1,
        }
    )
    monkeypatch.setattr(skyrl_train_backend, "SKYRL_RAY_PG_TIMEOUT_IN_S", 60)
    backend = SkyRLTrainBackend(BASE_MODEL, config)

    try:
        # Under colocate_all, the offload comes right after init_model has loaded the weights.
        with patch.object(PPORayActorGroup, "offload_to_cpu", side_effect=RuntimeError("injected build failure")):
            with pytest.raises(RuntimeError, match="injected build failure"):
                backend.create_model("model-a", LORA_CONFIG)
        # Sampling answers as if no create_model had run.
        with pytest.raises(RuntimeError, match="Create a model before sampling"):
            backend._ensure_inference_engines()

        backend.create_model("model-a", LORA_CONFIG)
        assert backend.has_model("model-a")
    finally:
        ray.shutdown()
