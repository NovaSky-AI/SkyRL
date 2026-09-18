from unittest.mock import patch

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    rocm_extra_engine_env_vars,
    rocm_visible_device_env,
)


def test_rocm_visible_device_env():
    assert rocm_visible_device_env([0, 2]) == {
        "HIP_VISIBLE_DEVICES": "0,1",
        "ROCR_VISIBLE_DEVICES": "0,2",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }


def test_rocm_extra_engine_env_vars_on_cuda():
    with patch(
        "skyrl.backends.skyrl_train.inference_servers.engine_utils.is_rocm_platform",
        return_value=False,
    ):
        assert rocm_extra_engine_env_vars() == {}


def test_engine_runtime_env_sets_ray_accel_override():
    from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
        build_engine_runtime_env,
    )

    runtime_env = build_engine_runtime_env()
    assert runtime_env is not None
    assert runtime_env["env_vars"]["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "0"


def test_rocm_extra_engine_env_vars_on_rocm(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V1", "0")
    with patch(
        "skyrl.backends.skyrl_train.inference_servers.engine_utils.is_rocm_platform",
        return_value=True,
    ):
        assert rocm_extra_engine_env_vars() == {
            "VLLM_USE_V1": "0",
            "VLLM_TARGET_DEVICE": "rocm",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
