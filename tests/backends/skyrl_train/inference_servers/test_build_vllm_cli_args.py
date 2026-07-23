"""Tests for build_vllm_cli_args on GPU-less hosts."""

from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    _apply_serialized_fp8_weight_sync_defaults,
    build_vllm_cli_args,
    resolve_policy_model_name,
)
from skyrl.train.config import SkyRLTrainConfig


def test_serialized_fp8_weight_sync_defaults_configure_vllm_checkpoint_fp8(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_serialized_fp8_ignored_layers", lambda _model_path: [])
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "serialized_blockwise"
    engine_kwargs = {"hf_overrides": {"rope_theta": 10000.0}}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs, model_path="qwen35-test")

    assert engine_kwargs["quantization"] == "fp8"
    assert engine_kwargs["load_format"] == "dummy"
    assert engine_kwargs["hf_overrides"]["rope_theta"] == 10000.0
    assert engine_kwargs["hf_overrides"]["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }


def test_serialized_mxfp8_weight_sync_configures_modelopt_experts(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(
        inference_utils,
        "_serialized_mxfp8_quantization_config",
        lambda _model_path: {
            "quant_method": "modelopt",
            "quant_algo": "MXFP8",
            "ignore": ["*.self_attn.*"],
        },
    )
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "serialized_mxfp8"
    engine_kwargs = {}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs, model_path="qwen3-moe-test")

    assert engine_kwargs == {
        "quantization": "modelopt_mxfp8",
        "load_format": "dummy",
        "hf_overrides": {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MXFP8",
                "ignore": ["*.self_attn.*"],
            }
        },
    }


def test_serialized_mxfp8_requires_bfloat16():
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "serialized_mxfp8"
    ie_cfg.model_dtype = "float16"

    with pytest.raises(ValueError, match="model_dtype='bfloat16'"):
        _apply_serialized_fp8_weight_sync_defaults(ie_cfg, {}, model_path="qwen3-moe-test")


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {"quantization": "awq"},
        {"load_format": "safetensors"},
        {"hf_overrides": {"quantization_config": {"weight_block_size": [64, 128]}}},
    ],
)
def test_serialized_fp8_weight_sync_rejects_conflicting_vllm_settings(engine_kwargs, monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_serialized_fp8_ignored_layers", lambda _model_path: [])
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"

    with pytest.raises(ValueError, match="serialized FP8 weight sync"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            engine_kwargs,
            model_path="qwen35-test",
        )


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {"hf_overrides": []},
        {"hf_overrides": {"quantization_config": []}},
    ],
)
def test_serialized_fp8_weight_sync_rejects_non_mapping_overrides(engine_kwargs):
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"

    with pytest.raises(ValueError, match="must be a dict"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            engine_kwargs,
            model_path="qwen35-test",
        )


def test_serialized_fp8_requires_model_path():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"

    with pytest.raises(ValueError, match="model path is required"):
        _apply_serialized_fp8_weight_sync_defaults(cfg.generator.inference_engine, {})


def test_serialized_fp8_fails_when_model_config_cannot_be_inspected(monkeypatch):
    import transformers

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"

    def fail_config_load(*_args, **_kwargs):
        raise OSError("missing config")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fail_config_load)
    with pytest.raises(RuntimeError, match="Could not inspect the model config"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {},
            model_path="missing-model",
        )


def test_serialized_fp8_rejects_unsupported_model_layout(monkeypatch):
    import transformers

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="llama"),
    )

    with pytest.raises(ValueError, match="only Qwen3.5"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {},
            model_path="unsupported-model",
        )


def test_serialized_mxfp8_rejects_unsupported_model_layout(monkeypatch):
    import transformers

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_mxfp8"
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="llama"),
    )

    with pytest.raises(ValueError, match="does not support model_type"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {},
            model_path="unsupported-model",
        )


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
    cfg.generator.inference_engine.served_model_name = "served-alias"
    cfg.generator.inference_engine.engine_init_kwargs = {
        "hf_overrides": {
            "rope_parameters": {
                "rope_type": "linear",
                "factor": 2.0,
                "rope_theta": 10000.0,
            }
        }
    }
    args = build_vllm_cli_args(cfg)

    assert args is not None
    assert args.model == cfg.trainer.policy.model.path
    assert args.served_model_name == ["served-alias"]
    assert args.tensor_parallel_size == cfg.generator.inference_engine.tensor_parallel_size
    assert args.hf_overrides["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10000.0,
    }
    assert vllm.platforms.current_platform.device_type == "cuda"

    # NOTE: the MTP speculative_config wiring test lives in
    # tests/backends/skyrl_train/mtp/test_build_vllm_cli_args_mtp.py


def test_resolve_policy_model_name_uses_served_model_name():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "base-model"
    cfg.generator.inference_engine.served_model_name = "served-alias"

    assert resolve_policy_model_name(cfg) == "served-alias"
