"""CPU tests for the process-local ``GPUModelRunner`` registry.

vLLM 0.30 defaults to Model Runner V2 (``vllm.v1.worker.gpu.model_runner``), a
separate class from V1. Recording only V1 left ``current_model_runner()`` at
None under V2, which silently skipped the post-sync FP8 KV scale normalization
(NaN generations under ``kv_cache_dtype=fp8_*``). Fake runner modules stand in
for vLLM so no install or GPU is needed.
"""

import sys
import types

import pytest

from skyrl.backends.skyrl_train.patches.vllm import (
    patch_model_runner_registry as registry,
)


class _FakeRunner:
    def load_model(self, *args, **kwargs):
        return "loaded"


@pytest.fixture
def fake_runners(monkeypatch):
    """Install fake V1/V2 runner modules and reset the registry's state."""
    v1 = type("GPUModelRunner", (_FakeRunner,), {})
    v2 = type("GPUModelRunner", (_FakeRunner,), {})
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", types.SimpleNamespace(GPUModelRunner=v1))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", types.SimpleNamespace(GPUModelRunner=v2))
    monkeypatch.setattr(registry, "_PATCHED", False)
    monkeypatch.setattr(registry, "_CURRENT_MODEL_RUNNER", None)
    return v1, v2


@pytest.mark.parametrize("which", [0, 1], ids=["v1", "v2"])
def test_records_runner_after_load(fake_runners, which):
    registry.apply_model_runner_registry_patch()
    runner = fake_runners[which]()
    assert registry.current_model_runner() is None
    assert runner.load_model() == "loaded"
    assert registry.current_model_runner() is runner


def test_apply_is_idempotent(fake_runners):
    v1, _ = fake_runners
    registry.apply_model_runner_registry_patch()
    wrapped = v1.load_model
    registry.apply_model_runner_registry_patch()
    assert v1.load_model is wrapped


def test_v2_module_missing_still_patches_v1(fake_runners, monkeypatch):
    v1, _ = fake_runners
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", None)
    registry.apply_model_runner_registry_patch()
    runner = v1()
    runner.load_model()
    assert registry.current_model_runner() is runner


@pytest.mark.vllm
def test_both_installed_vllm_runners_have_load_model():
    # Pins the wrapped targets to the installed vLLM: an upstream rename must
    # fail here rather than silently leave current_model_runner() at None.
    v1 = pytest.importorskip("vllm.v1.worker.gpu_model_runner")
    v2 = pytest.importorskip("vllm.v1.worker.gpu.model_runner")
    assert callable(v1.GPUModelRunner.load_model)
    assert callable(v2.GPUModelRunner.load_model)
