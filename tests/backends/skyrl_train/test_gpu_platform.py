"""CPU tests for the AMD ROCm gates in ``skyrl.backends.skyrl_train.gpu_platform``.

ROCm is simulated by setting ``torch.version.hip``, the build attribute ``is_rocm`` reads.

uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/test_gpu_platform.py
"""

import logging

import pytest
import torch

from skyrl.backends.skyrl_train.gpu_platform import (
    check_driver_device_masks,
    is_rocm,
    ray_masks_amd_gpus,
    rocm_unsupported,
)


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)


@pytest.fixture
def not_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)


def test_is_rocm_follows_the_torch_build(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert not is_rocm()
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)
    assert is_rocm()


@pytest.mark.parametrize(
    "env, masks",
    [
        ({}, True),
        ({"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1"}, False),
        ({"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "True"}, False),
        ({"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "0"}, True),
        # Ray reads the HIP variable alone on an AMD node.
        ({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}, True),
    ],
)
def test_ray_masks_amd_gpus_follows_rays_own_variable(env, masks):
    assert ray_masks_amd_gpus(env) is masks


def test_ray_masks_amd_gpus_defaults_to_the_live_process_environment(monkeypatch):
    monkeypatch.delenv("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", raising=False)
    assert ray_masks_amd_gpus()
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    assert not ray_masks_amd_gpus()


def test_check_driver_device_masks_rejects_a_cuda_only_mask_on_rocm(rocm):
    with pytest.raises(RuntimeError, match="Unset CUDA_VISIBLE_DEVICES"):
        check_driver_device_masks({"CUDA_VISIBLE_DEVICES": "0,1"})


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"HIP_VISIBLE_DEVICES": "0,1"},
        {"HIP_VISIBLE_DEVICES": "0,1", "CUDA_VISIBLE_DEVICES": "0,1"},
        {"CUDA_VISIBLE_DEVICES": ""},
    ],
)
def test_check_driver_device_masks_accepts_what_ray_honours_on_rocm(rocm, env):
    check_driver_device_masks(env)


def test_check_driver_device_masks_is_silent_on_a_cuda_build(not_rocm):
    check_driver_device_masks({"CUDA_VISIBLE_DEVICES": "0,1"})


@pytest.mark.parametrize("feature", ["sharded_rdt", "fp8", "mp_executor", "nccl_multi_gpu_engine"])
def test_rocm_unsupported_names_the_platform(feature):
    assert rocm_unsupported(feature).startswith("Not supported on AMD ROCm: ")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_lm_head_kernel_asks_for_ieee_dot_precision_on_rocm(rocm, dtype):
    # Triton on ROCm has no tf32: it fails to compile the kernel when asked for it.
    from skyrl.backends.skyrl_train.distributed.megatron import (
        fused_linear_logprob_triton as kernel,
    )

    assert kernel._dot_input_precision(torch.empty(1, dtype=dtype)) == "ieee"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_lm_head_kernel_keeps_tf32_dot_precision_on_a_cuda_build(not_rocm, monkeypatch, dtype):
    from skyrl.backends.skyrl_train.distributed.megatron import (
        fused_linear_logprob_triton as kernel,
    )

    monkeypatch.setattr(kernel, "FORCE_FP32_IEEE_PRECISION", False)
    assert kernel._dot_input_precision(torch.empty(1, dtype=dtype)) == "tf32"


def test_unregistered_bridges_warn_unless_megatron_itself_is_missing(caplog):
    from skyrl.backends.skyrl_train.workers.megatron import model_bridges

    def levels():
        # A set: another handler on the way to the root logger can deliver the same record twice.
        return {r.levelno for r in caplog.records if r.name == model_bridges.__name__}

    with caplog.at_level(logging.DEBUG, logger=model_bridges.__name__):
        model_bridges._report_unregistered_bridges(ModuleNotFoundError("No module named 'megatron'", name="megatron"))
        assert levels() == {logging.DEBUG}
        caplog.clear()
        model_bridges._report_unregistered_bridges(
            ImportError("cannot import name 'Qwen35Bridge'", name="megatron.bridge.models.qwen.qwen35_bridge")
        )
        assert levels() == {logging.WARNING}
        assert "extra Megatron bridges are not registered" in caplog.records[0].getMessage()
