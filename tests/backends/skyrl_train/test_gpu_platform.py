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


def test_unregistered_bridges_warn_unless_megatron_itself_is_missing(caplog):
    from skyrl.backends.skyrl_train.workers.megatron import model_bridges

    with caplog.at_level(logging.DEBUG, logger=model_bridges.__name__):
        model_bridges._report_unregistered_bridges(ModuleNotFoundError("No module named 'megatron'", name="megatron"))
        assert [r.levelno for r in caplog.records] == [logging.DEBUG]
        caplog.clear()
        model_bridges._report_unregistered_bridges(
            ImportError("cannot import name 'Qwen35Bridge'", name="megatron.bridge.models.qwen.qwen35_bridge")
        )
        assert [r.levelno for r in caplog.records] == [logging.WARNING]
        assert "extra Megatron bridges are not registered" in caplog.records[0].getMessage()
