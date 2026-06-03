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
def test_build_vllm_cli_args_speculative_config_mtp(monkeypatch):
    """speculative_config is passed through to vLLM for MTP draft decoding."""
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    # Default: no speculative decoding.
    args = build_vllm_cli_args(cfg)
    assert getattr(args, "speculative_config", None) is None

    # Enable MTP speculative decoding.
    spec = {"method": "mtp", "num_speculative_tokens": 1}
    cfg.generator.inference_engine.speculative_config = spec
    args = build_vllm_cli_args(cfg)
    assert args.speculative_config == spec
