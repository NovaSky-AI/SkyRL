"""Tests for build_vllm_cli_args on GPU-less hosts."""

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.vllm
def test_build_vllm_cli_args_succeeds_on_gpu_less_host(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    # Simulate the GPU-less Ray head-node case: vLLM resolves current_platform
    # to UnspecifiedPlatform (device_type == ""), so AsyncEngineArgs.add_cli_args
    # walks VllmConfig defaults, instantiates DeviceConfig() and its
    # __post_init__ raises "Failed to infer device type" during arg parsing.
    # With the fix in build_vllm_cli_args, current_platform.device_type is
    # pinned to "cuda" before add_cli_args runs.
    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    args = build_vllm_cli_args(cfg)

    assert args is not None
    assert args.model == cfg.trainer.policy.model.path
    assert args.tensor_parallel_size == cfg.generator.inference_engine.tensor_parallel_size
    assert vllm.platforms.current_platform.device_type == "cuda"


@pytest.mark.vllm
def test_build_vllm_cli_args_fp32_lm_head_off_by_default():
    """The fp32 lm-head flag is not injected into additional_config by default."""
    cfg = SkyRLTrainConfig()
    assert cfg.generator.inference_engine.enable_fp32_lm_head is False

    args = build_vllm_cli_args(cfg)

    additional_config = getattr(args, "additional_config", None) or {}
    assert "skyrl_enable_fp32_lm_head" not in additional_config


@pytest.mark.vllm
def test_build_vllm_cli_args_fp32_lm_head_propagated_via_additional_config():
    """When enabled, the flag is propagated through additional_config so it
    reaches every vLLM worker."""
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.enable_fp32_lm_head = True

    args = build_vllm_cli_args(cfg)

    additional_config = getattr(args, "additional_config", None) or {}
    assert additional_config.get("skyrl_enable_fp32_lm_head") is True
