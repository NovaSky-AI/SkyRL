import sys
import types

import pytest

from examples.train.flash_rl import runtime


@pytest.fixture(autouse=True)
def _clear_flashrl_runtime_cache(monkeypatch):
    runtime.load_flashrl_patch_fn.cache_clear()
    monkeypatch.delenv(runtime.FLASHRL_CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(runtime.FLASHRL_PATCH_FN_ENV_VAR, raising=False)
    yield
    runtime.load_flashrl_patch_fn.cache_clear()


def _register_module_chain(monkeypatch, module_name: str):
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        current_name = ".".join(parts[:index])
        module = sys.modules.get(current_name, types.ModuleType(current_name))
        monkeypatch.setitem(sys.modules, current_name, module)
        if index > 1:
            parent_name = ".".join(parts[: index - 1])
            setattr(sys.modules[parent_name], parts[index - 1], module)
    return sys.modules[module_name]


def test_validate_flashrl_environment_requires_flashrl_config():
    with pytest.raises(RuntimeError, match="FLASHRL_CONFIG"):
        runtime.validate_flashrl_environment(validate_patch_fn=False)


def test_load_flashrl_patch_fn_requires_module_callable_format(monkeypatch):
    monkeypatch.setenv(runtime.FLASHRL_CONFIG_ENV_VAR, "fp8_vllm")
    monkeypatch.setenv(runtime.FLASHRL_PATCH_FN_ENV_VAR, "invalid-target")

    with pytest.raises(RuntimeError, match="<module>:<callable>"):
        runtime.load_flashrl_patch_fn()


def test_load_flashrl_patch_fn_uses_default_patch_target(monkeypatch):
    monkeypatch.setenv(runtime.FLASHRL_CONFIG_ENV_VAR, "fp8_vllm")
    patch_module = _register_module_chain(monkeypatch, "vllm.model_executor.layers.patch")

    calls = {"count": 0}

    def apply_patch():
        calls["count"] += 1

    patch_module.apply_patch = apply_patch

    patch_fn = runtime.load_flashrl_patch_fn()
    assert patch_fn is apply_patch
    assert runtime.validate_flashrl_environment() == "fp8_vllm"

    patch_fn()
    assert calls["count"] == 1


def test_load_flashrl_patch_fn_supports_env_override(monkeypatch):
    monkeypatch.setenv(runtime.FLASHRL_CONFIG_ENV_VAR, "fp8_vllm")
    monkeypatch.setenv(runtime.FLASHRL_PATCH_FN_ENV_VAR, "flashrl_custom.patch:install")
    patch_module = _register_module_chain(monkeypatch, "flashrl_custom.patch")

    def install():
        return None

    patch_module.install = install

    assert runtime.load_flashrl_patch_fn() is install
