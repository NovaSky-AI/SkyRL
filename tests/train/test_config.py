"""
uv run --isolated --extra dev pytest -s tests/train/test_config.py
"""

import pathlib
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Optional

import pytest
from omegaconf import OmegaConf

from skyrl.backends.skyrl_train.distributed.megatron import quantization_utils
from skyrl.train.config.config import (
    BaseConfig,
    DeltaWeightSyncConfig,
    SkyRLTrainConfig,
    TrainerConfig,
    _resolve_class_type,
    build_nested_dataclass,
    overrides_dict_to_dotlist,
)
from skyrl.train.utils import utils as train_utils
from skyrl.train.utils.utils import (
    prepare_runtime_environment,
    validate_cfg,
    validate_inference_engine_cfg,
)
from tests.train.util import example_dummy_config


def _get_nested_attr(cfg, dotted_path: str):
    """Resolve a dot-notation config path to its value."""
    node = cfg
    for part in dotted_path.split("."):
        node = getattr(node, part)
    return node


def _make_validated_test_config():
    """Return a small config that passes validate_batch_sizes()."""
    cfg = example_dummy_config()
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.critic_mini_batch_size = cfg.trainer.train_batch_size
    return cfg


# Helper dataclasses for testing
@dataclass
class _SimpleConfig(BaseConfig):
    a: int = 0


class SimpleEnum(Enum):
    A = "a"


@dataclass
class _NestedConfig(BaseConfig):
    b: int = 1
    c: Annotated[_SimpleConfig, "test"] = field(default_factory=_SimpleConfig)
    d: Optional[_SimpleConfig] = None
    e: Optional[SimpleEnum] = SimpleEnum.A


def test_build_nested_dataclass():
    # not all fields are present
    d = {"b": 4, "c": {"a": 2}}
    cfg = build_nested_dataclass(_NestedConfig, d)
    assert cfg.b == 4
    assert cfg.c.a == 2

    # all fields are present
    d = {"b": 4, "c": {"a": 2}, "d": {"a": 3}}
    cfg = build_nested_dataclass(_NestedConfig, d)
    assert cfg.b == 4
    assert cfg.c.a == 2
    assert cfg.d.a == 3


def test_build_nested_dataclass_full_config():
    d = {"trainer": {"policy": {"model": {"path": "path/to/model"}}}}
    cfg = build_nested_dataclass(SkyRLTrainConfig, d)
    assert cfg.trainer.policy.model.path == "path/to/model"


def test_build_nested_dataclass_invalid_config():
    d = {"path": "path/to/model"}
    with pytest.raises(ValueError):
        build_nested_dataclass(SkyRLTrainConfig, d)


def test_build_config_from_dict_config():
    cfg = OmegaConf.create({"a": 1})
    cfg = _SimpleConfig.from_dict_config(cfg)
    assert cfg.a == 1

    cfg = OmegaConf.create({"b": 1, "c": {"a": 2}})
    cfg = _NestedConfig.from_dict_config(cfg)
    assert cfg.b == 1
    assert cfg.c.a == 2

    cfg = OmegaConf.create({"b": 1, "c": {"a": 2}, "e": "a"})
    cfg = _NestedConfig.from_dict_config(cfg)
    assert cfg.b == 1
    assert cfg.c.a == 2
    assert isinstance(cfg.e, SimpleEnum)


def test_build_config_from_dict_config_invalid_config():
    cfg = OmegaConf.create({"path": "path/to/model"})
    with pytest.raises(ValueError):
        _SimpleConfig.from_dict_config(cfg)


def test_dtype_resolution():
    assert not _resolve_class_type(typing.Optional[int])
    assert _resolve_class_type(typing.Optional[_SimpleConfig]) is _SimpleConfig
    assert _resolve_class_type(typing.Union[None, _SimpleConfig]) is _SimpleConfig
    assert _resolve_class_type(typing.Annotated[_SimpleConfig, "test"]) is _SimpleConfig
    assert _resolve_class_type(Optional[SimpleEnum]) is SimpleEnum


def test_cli_overrides():
    # Basic overrides - str, int and dict fields
    overrides = [
        "trainer.policy.model.path=path/to/model",
        "trainer.seed=123",
        "generator.inference_engine.engine_init_kwargs.field=value",
        "generator.sampling_params.temperature=0.7",
    ]
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    assert cfg.trainer.policy.model.path == "path/to/model"
    assert cfg.trainer.seed == 123
    assert cfg.generator.inference_engine.engine_init_kwargs["field"] == "value"
    assert cfg.generator.sampling_params.temperature == 0.7

    # check that temperature is propagated to algorithm config
    assert cfg.trainer.algorithm.temperature == 0.7


def test_cli_overrides_empty_args():
    cfg = SkyRLTrainConfig.from_cli_overrides([])
    assert cfg.trainer.policy.model.path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.trainer.seed == 42


def test_cli_overrides_fp8_param_gather():
    cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.policy.megatron_config.ddp_config.fp8_param_gather=true"])
    assert cfg.trainer.policy.megatron_config.ddp_config.fp8_param_gather is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("vocab_entropy_chunk_size", -1),
        ("vocab_entropy_chunk_size", True),
        ("vocab_entropy_chunk_memory_mb", 0),
        ("vocab_entropy_chunk_memory_mb", True),
    ],
)
def test_trainer_config_rejects_invalid_vocab_entropy_chunking(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        TrainerConfig(**{field_name: value})


def test_runtime_env_forwards_te_block_scale_mode(monkeypatch):
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)

    env_vars = prepare_runtime_environment(example_dummy_config())

    assert env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"


def test_runtime_env_supports_fsdp_without_megatron_configs(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("NVTE_FP8_BLOCK_AMAX_EPSILON", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    cfg = example_dummy_config()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.megatron_config = None
    cfg.trainer.ref.megatron_config = None

    env_vars = prepare_runtime_environment(cfg)

    assert "NVTE_FP8_BLOCK_SCALING_FP32_SCALES" not in env_vars
    assert "VLLM_USE_DEEP_GEMM_E8M0" not in env_vars


def test_serialized_fp8_runtime_defaults_to_fp32_scales(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: False)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"
    assert env_vars["VLLM_USE_DEEP_GEMM_E8M0"] == "0"


def test_serialized_fp8_runtime_defaults_to_pow2_scales_on_blackwell(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: True)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"
    assert env_vars["VLLM_USE_DEEP_GEMM_E8M0"] == "1"


def test_serialized_fp8_pow2_scales_reject_disabled_e8m0_on_blackwell(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "0")
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: True)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="VLLM_USE_DEEP_GEMM_E8M0=1"):
        prepare_runtime_environment(cfg)


def test_serialized_fp8_requires_an_explicit_scale_mode_without_a_driver_gpu(monkeypatch):
    """The contract is baked into the runtime env before ray.init, so a GPU-less
    head cannot infer it from the workers; guessing Hopper would hand FP32 block
    scales to Blackwell workers."""
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: False)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="NVTE_FP8_BLOCK_SCALING_FP32_SCALES"):
        prepare_runtime_environment(cfg)


def test_serialized_fp8_pow2_scales_set_e8m0_without_a_driver_gpu(monkeypatch):
    """E8M0 follows the wire scale format, not the driver's device: vLLM picks the
    per-device form itself, so the default must survive a GPU-less head."""
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: False)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: False)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "0"
    assert env_vars["VLLM_USE_DEEP_GEMM_E8M0"] == "1"


@pytest.mark.parametrize("backend", ["sharded_rdt", "delta"])
def test_serialized_fp8_weight_sync_rejects_backends_without_a_chunk_channel(backend):
    """Neither backend carries payload + scale pairs, and both would otherwise
    fail only at the first sync -- after vLLM has loaded as FP8."""
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    cfg.generator.inference_engine.weight_sync_backend = backend

    with pytest.raises(ValueError, match=backend):
        validate_inference_engine_cfg(cfg)


def test_serialized_fp8_weight_sync_requires_megatron():
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "fsdp"
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="requires trainer.strategy='megatron'"):
        validate_inference_engine_cfg(cfg)


def test_serialized_fp8_weight_sync_rejects_adapter_only_megatron_lora():
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

    with pytest.raises(ValueError, match="requires full-weight updates"):
        validate_inference_engine_cfg(cfg)


def test_megatron_fp8_compute_defaults_to_fp32_scales_without_serialized_sync(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    monkeypatch.setattr(train_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: False)
    cfg = example_dummy_config()
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8"] = "hybrid"

    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] == "1"


def test_power_2_mode_rejects_persistent_fp8_without_serialized_sync(monkeypatch):
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    cfg = example_dummy_config()
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_param"] = True

    with pytest.raises(ValueError, match="fp8_param=false on Blackwell"):
        prepare_runtime_environment(cfg)


def test_megatron_validation_requires_fp8_param_gather_for_training():
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_param"] = True
    cfg.trainer.policy.megatron_config.ddp_config.fp8_param_gather = False

    with pytest.raises(ValueError, match="fp8_param_gather=true"):
        train_utils.validate_megatron_cfg(cfg)


def test_megatron_top_level_fp8_fields_fold_into_transformer_config_kwargs():
    from skyrl.train.config.config import MegatronConfig

    cfg = MegatronConfig(fp8="e4m3", fp8_recipe="auto", fp8_param=True, fp8_amax_compute_algo="most_recent")
    assert cfg.transformer_config_kwargs["fp8"] == "e4m3"
    assert cfg.transformer_config_kwargs["fp8_recipe"] == "auto"
    assert cfg.transformer_config_kwargs["fp8_param"] is True
    assert cfg.transformer_config_kwargs["fp8_amax_compute_algo"] == "most_recent"
    # Defaults stay off: no FP8 keys appear unless requested.
    assert "fp8" not in MegatronConfig().transformer_config_kwargs


def test_megatron_explicit_transformer_config_kwargs_override_top_level_fp8_fields():
    from skyrl.train.config.config import MegatronConfig

    cfg = MegatronConfig(fp8="e4m3", fp8_recipe="blockwise", transformer_config_kwargs={"fp8_recipe": "mxfp8"})
    assert cfg.transformer_config_kwargs["fp8_recipe"] == "mxfp8"
    assert cfg.transformer_config_kwargs["fp8"] == "e4m3"


def test_megatron_validation_allows_inference_only_fp8_param_without_gather():
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_param"] = True
    cfg.trainer.policy.megatron_config.ddp_config.fp8_param_gather = False

    train_utils.validate_megatron_cfg(cfg)


@pytest.mark.parametrize(("blackwell", "expected_recipe"), [(True, "mxfp8"), (False, "blockwise")])
def test_megatron_validation_resolves_auto_fp8_recipe(monkeypatch, blackwell, expected_recipe):
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: blackwell)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: blackwell)
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8"] = "e4m3"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_recipe"] = "auto"

    train_utils.validate_megatron_cfg(cfg)

    assert cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_recipe"] == expected_recipe


def test_megatron_validation_rejects_mxfp8_before_blackwell(monkeypatch):
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: False)
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: False)
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8"] = "e4m3"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_recipe"] = "mxfp8"

    with pytest.raises(ValueError, match="requires SM100"):
        train_utils.validate_megatron_cfg(cfg)


def test_megatron_validation_rejects_mxfp8_with_fp8_param(monkeypatch):
    monkeypatch.setattr(train_utils, "is_blackwell_or_newer", lambda: True)
    cfg = _make_validated_test_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8"] = "e4m3"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_recipe"] = "mxfp8"
    cfg.trainer.policy.megatron_config.transformer_config_kwargs["fp8_param"] = True
    cfg.trainer.policy.megatron_config.ddp_config.fp8_param_gather = True

    with pytest.raises(ValueError, match="not supported with fp8_recipe=mxfp8"):
        train_utils.validate_megatron_cfg(cfg)


def test_serialized_fp8_fp32_scales_reject_vllm_e8m0(monkeypatch):
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM_E8M0", "1")
    monkeypatch.setattr(train_utils, "peer_access_supported", lambda **_kwargs: True)
    cfg = example_dummy_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="VLLM_USE_DEEP_GEMM_E8M0=0"):
        prepare_runtime_environment(cfg)


def test_cli_overrides_plus_prefix_rejected():
    with pytest.raises(ValueError, match="The '\\+' prefix"):
        SkyRLTrainConfig.from_cli_overrides(["+new_field=value"])


def test_cli_overrides_invalid_field():
    with pytest.raises(ValueError, match="Invalid fields"):
        SkyRLTrainConfig.from_cli_overrides(["trainer.nonexistent_field=value"])


def test_remote_urls_override_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "`remote_urls` is no longer supported, external inference servers can be used with "
            "`external_proxy_url` and `external_server_urls` instead"
        ),
    ):
        SkyRLTrainConfig.from_cli_overrides(["generator.inference_engine.remote_urls=['http://127.0.0.1:8001']"])


def test_async_engine_true_override_is_ignored():
    cfg = SkyRLTrainConfig.from_cli_overrides(["generator.inference_engine.async_engine=true"])

    assert not hasattr(cfg.generator.inference_engine, "async_engine")


def test_async_engine_false_override_rejected():
    with pytest.raises(ValueError, match="`async_engine=False` is no longer supported"):
        SkyRLTrainConfig.from_cli_overrides(["generator.inference_engine.async_engine=false"])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            "generator.inference_engine.enable_http_endpoint=true",
            "`enable_http_endpoint` is no longer supported",
        ),
        (
            "generator.inference_engine.enable_http_endpoint=false",
            "`enable_http_endpoint` is no longer supported",
        ),
        (
            "generator.inference_engine.override_existing_update_group=enable",
            "`override_existing_update_group` is no longer supported",
        ),
        (
            "generator.inference_engine.override_existing_update_group=auto",
            "`override_existing_update_group` is no longer supported",
        ),
    ],
)
def test_removed_inference_engine_overrides_rejected(override: str, match: str):
    with pytest.raises(ValueError, match=match):
        SkyRLTrainConfig.from_cli_overrides([override])


@pytest.mark.parametrize(
    "override",
    [
        "trainer.rope_scaling={'type': 'linear'}",
        "trainer.rope_theta=10000",
        "trainer.rope_parameters={'rope_type': 'linear'}",
        "generator.rope_scaling={'type': 'linear'}",
        "generator.rope_theta=10000",
        "generator.rope_parameters={'rope_type': 'linear'}",
        "generator.inference_engine.rope_scaling={'type': 'linear'}",
        "generator.inference_engine.rope_theta=10000",
        "generator.inference_engine.rope_parameters={'rope_type': 'linear'}",
        "generator.inference_engine.engine_init_kwargs.rope_scaling={'type': 'linear'}",
        "generator.inference_engine.engine_init_kwargs.rope_theta=10000",
        "generator.inference_engine.engine_init_kwargs.rope_parameters={'rope_type': 'linear'}",
        "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_scaling={'type': 'linear'}",
        "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_theta=10000",
    ],
)
def test_native_rope_overrides_rejected(override):
    with pytest.raises(
        ValueError,
        match="engine_init_kwargs\\.hf_overrides\\.rope_parameters",
    ):
        SkyRLTrainConfig.from_cli_overrides([override])


def test_hf_overrides_rope_parameters_requires_trainer_side_override():
    with pytest.raises(
        ValueError,
        match="trainer\\.policy\\.model_config_kwargs\\.rope_parameters",
    ):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_type=linear",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.factor=2.0",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_theta=10000",
            ]
        )


def test_hf_overrides_rope_parameters_allowed_with_policy_model_config_kwargs():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_type=linear",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.factor=2.0",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_theta=10000",
            "trainer.policy.model_config_kwargs.rope_parameters.rope_type=linear",
            "trainer.policy.model_config_kwargs.rope_parameters.factor=2.0",
            "trainer.policy.model_config_kwargs.rope_parameters.rope_theta=10000",
        ]
    )

    assert cfg.generator.inference_engine.engine_init_kwargs["hf_overrides"]["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10000,
    }
    assert cfg.trainer.policy.model_config_kwargs["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10000,
    }


def test_hf_overrides_rope_parameters_allowed_with_megatron_transformer_config_kwargs():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=megatron",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_type=linear",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.factor=2.0",
            "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_theta=10000",
            "trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters.rope_type=linear",
            "trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters.factor=2.0",
            "trainer.policy.megatron_config.transformer_config_kwargs.rope_parameters.rope_theta=10000",
        ]
    )

    assert cfg.trainer.policy.megatron_config.transformer_config_kwargs["rope_parameters"] == {
        "rope_type": "linear",
        "factor": 2.0,
        "rope_theta": 10000,
    }


def test_hf_overrides_rope_parameters_must_match_fsdp_trainer_side_override():
    with pytest.raises(
        ValueError,
        match="trainer\\.policy\\.model_config_kwargs\\.rope_parameters",
    ):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_type=linear",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.factor=2.0",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_theta=10000",
                "trainer.policy.model_config_kwargs.rope_parameters.rope_type=linear",
                "trainer.policy.model_config_kwargs.rope_parameters.factor=4.0",
                "trainer.policy.model_config_kwargs.rope_parameters.rope_theta=10000",
            ]
        )


def test_hf_overrides_rope_parameters_must_match_megatron_trainer_side_override():
    with pytest.raises(
        ValueError,
        match="trainer\\.policy\\.megatron_config\\.transformer_config_kwargs\\.rope_parameters",
    ):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "trainer.strategy=megatron",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_type=linear",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.factor=2.0",
                "generator.inference_engine.engine_init_kwargs.hf_overrides.rope_parameters.rope_theta=10000",
                "trainer.policy.model_config_kwargs.rope_parameters.rope_type=linear",
                "trainer.policy.model_config_kwargs.rope_parameters.factor=2.0",
                "trainer.policy.model_config_kwargs.rope_parameters.rope_theta=10000",
            ]
        )


def test_run_engines_locally_false_requires_external_endpoint():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.run_engines_locally = False

    with pytest.raises(ValueError, match="run_engines_locally=false requires"):
        validate_inference_engine_cfg(cfg)


def _pd_cfg_with_role_kwargs(prefill_kwargs=None, decode_kwargs=None) -> SkyRLTrainConfig:
    """Build a minimally-valid P/D config, optionally with role-specific engine kwargs."""
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.enable_pd = True
    ie_cfg.num_engines = 2
    ie_cfg.num_prefill = 1
    if prefill_kwargs is not None:
        ie_cfg.prefill_init_kwargs = prefill_kwargs
    if decode_kwargs is not None:
        ie_cfg.decode_init_kwargs = decode_kwargs
    return cfg


def test_role_init_kwargs_conflict_with_engine_init_kwargs_rejected():
    cfg = _pd_cfg_with_role_kwargs(
        prefill_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector"}},
        decode_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector"}},
    )
    cfg.generator.inference_engine.engine_init_kwargs = {"gpu_memory_utilization": 0.8}

    with pytest.raises(ValueError, match="engine_init_kwargs cannot be combined with"):
        validate_inference_engine_cfg(cfg)


def test_role_init_kwargs_require_enable_pd():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.prefill_init_kwargs = {"kv_transfer_config": {"kv_connector": "NixlConnector"}}

    with pytest.raises(ValueError, match="only valid with enable_pd"):
        validate_inference_engine_cfg(cfg)


def test_partial_role_init_kwargs_rejected():
    # Only prefill_init_kwargs set -> decode_init_kwargs is missing kv_transfer_config.
    cfg = _pd_cfg_with_role_kwargs(
        prefill_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector"}},
    )

    with pytest.raises(ValueError, match="decode_init_kwargs must set kv_transfer_config"):
        validate_inference_engine_cfg(cfg)


def test_valid_pd_role_init_kwargs_passes():
    cfg = _pd_cfg_with_role_kwargs(
        prefill_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector"}},
        decode_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector"}},
    )

    # Should not raise.
    validate_inference_engine_cfg(cfg)


def test_offload_kv_for_weight_sync_rejects_colocated():
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = True
    cfg.generator.inference_engine.offload_kv_for_weight_sync = True
    with pytest.raises(AssertionError, match="non-colocated weight sync only"):
        validate_inference_engine_cfg(cfg)


def test_offload_kv_for_weight_sync_rejects_lora():
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.offload_kv_for_weight_sync = True
    cfg.trainer.policy.model.lora.rank = 8
    with pytest.raises(AssertionError, match="does not support LoRA"):
        validate_inference_engine_cfg(cfg)


def test_offload_kv_for_weight_sync_sync_trainer_ok():
    # Non-fully-async (synchronous trainer) is supported: plain sleep, no in-flight.
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.fully_async.enabled = False
    cfg.generator.inference_engine.offload_kv_for_weight_sync = True
    validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize("clear_kv_cache", [False, True])
def test_offload_kv_for_weight_sync_async_ok(clear_kv_cache):
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.offload_kv_for_weight_sync = True
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.clear_kv_cache_on_weight_sync = clear_kv_cache
    # Both clear_kv_cache settings are supported now.
    validate_inference_engine_cfg(cfg)


def test_temperature_propagation():
    """Test that temperature is copied from generator to algorithm config in __post_init__."""
    cfg = SkyRLTrainConfig.from_cli_overrides(["generator.sampling_params.temperature=0.7"])
    assert cfg.generator.sampling_params.temperature == 0.7
    assert cfg.trainer.algorithm.temperature == 0.7


def test_cross_field_defaults():
    """Test that cross-field defaults are applied correctly."""
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.max_prompt_length=1024",
            "trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct",
        ]
    )

    assert cfg.generator.max_input_length == 1024  # same as `trainer.max_prompt_length`
    assert cfg.trainer.ref.model.path == "Qwen/Qwen2.5-1.5B-Instruct"  # same as `trainer.policy.model.path`
    assert (
        cfg.generator.eval_sampling_params.max_generate_length == cfg.generator.sampling_params.max_generate_length
    )  # same as `generator.sampling_params.max_generate_length`


def test_fake_int4_qat_defaults():
    """Defaults must stay pinned to the llm-compressor RTN convention, disabled."""
    cfg = SkyRLTrainConfig.from_cli_overrides([])
    fq = cfg.trainer.policy.model.fake_int4_qat
    assert fq.enabled is False
    assert fq.group_size == 32
    assert fq.scale_divisor == 7.5
    assert fq.q_min == -8.0
    assert fq.bf16_base_path is None


def test_fake_int4_qat_cli_overrides():
    """All convention knobs must be settable from the CLI (the Kimi K2.x convention)."""
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=megatron",
            "trainer.policy.model.lora.rank=32",
            "trainer.policy.megatron_config.lora_config.merge_lora=false",
            "trainer.policy.model.fake_int4_qat.enabled=true",
            "trainer.policy.model.fake_int4_qat.group_size=32",
            "trainer.policy.model.fake_int4_qat.scale_divisor=7.0",
            "trainer.policy.model.fake_int4_qat.q_min=-7",
            "trainer.policy.model.fake_int4_qat.bf16_base_path=/data/bf16-dump",
        ]
    )
    fq = cfg.trainer.policy.model.fake_int4_qat
    assert fq.enabled is True
    assert fq.scale_divisor == 7.0
    assert fq.q_min == -7.0
    assert fq.bf16_base_path == "/data/bf16-dump"


def test_fake_int4_qat_requires_lora():
    with pytest.raises(AssertionError, match="currently requires LoRA"):
        SkyRLTrainConfig.from_cli_overrides(["trainer.policy.model.fake_int4_qat.enabled=true"])


def test_fake_int4_qat_requires_megatron():
    with pytest.raises(AssertionError, match="strategy=megatron"):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "trainer.policy.model.lora.rank=32",
                "trainer.policy.megatron_config.lora_config.merge_lora=false",
                "trainer.policy.model.fake_int4_qat.enabled=true",
            ]
        )


def test_fake_int4_qat_requires_unmerged_lora_sync():
    with pytest.raises(AssertionError, match="merge_lora=False"):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "trainer.strategy=megatron",
                "trainer.policy.model.lora.rank=32",
                "trainer.policy.model.fake_int4_qat.enabled=true",
            ]
        )


class TestOverridesDictToDotlist:
    """``overrides_dict_to_dotlist`` emits values that ``OmegaConf.from_cli`` parses
    back to the same Python object. See https://github.com/NovaSky-AI/SkyRL/issues/1567.
    """

    @pytest.mark.parametrize(
        ("value", "expected_arg"),
        [
            pytest.param(None, "k=null", id="none"),
            pytest.param(True, "k=true", id="bool-true"),
            pytest.param(False, "k=false", id="bool-false"),
            pytest.param(7, "k=7", id="int"),
            pytest.param(1.5, "k=1.5", id="float"),
            pytest.param("hello", 'k="hello"', id="str"),
            pytest.param("null", 'k="null"', id="str-null"),
            pytest.param("", 'k=""', id="str-empty"),
            pytest.param("a,b", 'k="a,b"', id="str-comma"),
            pytest.param("a: b", 'k="a: b"', id="str-colon"),
            pytest.param(["a", None], 'k=["a", null]', id="list"),
            pytest.param({"a": 1}, 'k={"a": 1}', id="dict"),
            # Non-ASCII stays literal: \\uXXXX escaping splits astral-plane
            # characters into surrogate halves that OmegaConf decodes separately.
            pytest.param("café", 'k="café"', id="non-ascii-bmp"),
            pytest.param("run-\U0001f600", 'k="run-\U0001f600"', id="non-ascii-astral"),
        ],
    )
    def test_serialization(self, value, expected_arg):
        assert overrides_dict_to_dotlist({"k": value}) == [expected_arg]

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            7,
            1.5,
            "hello",
            "null",
            "true",
            "",
            "1e5",
            "on",
            "a,b",
            "a: b",
            "[a]",
            "{a: b}",
            'say "hi"',
            "a\nb",
            "a\\b",
            "café",
            "run-\U0001f600",
            ["http://a:1", "http://b:2"],
            [],
            {"a": 1, "b": "x"},
            {},
        ],
    )
    def test_values_round_trip_through_omegaconf(self, value):
        (arg,) = overrides_dict_to_dotlist({"k": value})
        parsed = OmegaConf.to_container(OmegaConf.from_cli([arg]), resolve=False)["k"]
        assert parsed == value
        assert type(parsed) is type(value)
        if isinstance(parsed, str):
            # Lone surrogates only surface on encode.
            parsed.encode("utf-8")

    def test_multiple_keys_preserve_order(self):
        assert overrides_dict_to_dotlist({"a": 1, "b": None}) == ["a=1", "b=null"]

    def test_non_json_serializable_values_fall_back_to_str(self):
        assert overrides_dict_to_dotlist({"k": pathlib.Path("/tmp/x")}) == ["k=/tmp/x"]

    def test_circular_reference_falls_back_to_str(self):
        value = {}
        value["self"] = value
        (arg,) = overrides_dict_to_dotlist({"k": value})
        assert arg.startswith("k={")


class TestCliOverridesFromDict:
    """Dict overrides round-trip by type rather than through YAML re-parsing.

    The dict path serializes values into ``key=value`` strings for
    ``OmegaConf.from_cli``, which re-parses each value with YAML scalar rules.
    See https://github.com/NovaSky-AI/SkyRL/issues/1567.
    """

    def test_none_values_stay_none(self):
        """``None`` values arrive as ``None``, not the string ``"None"``."""
        cfg = SkyRLTrainConfig.from_cli_overrides(
            {
                "generator.inference_engine.external_server_urls": None,
                "generator.inference_engine.external_proxy_url": None,
                "generator.inference_engine.served_model_name": None,
            }
        )
        ie_cfg = cfg.generator.inference_engine
        assert ie_cfg.external_server_urls is None
        assert ie_cfg.external_proxy_url is None
        assert ie_cfg.served_model_name is None

    @pytest.mark.parametrize(
        "value",
        [
            "null",  # YAML null
            "true",  # YAML bool
            "false",
            "",  # empty scalar -> YAML null
            "[not-a-list]",  # YAML flow sequence
            "{not: a-dict}",  # YAML flow mapping
            "name: with colon",  # YAML block mapping
            "1e5",  # YAML float
            "on",  # YAML 1.1 bool
        ],
    )
    def test_string_values_stay_strings(self, value):
        """A ``str`` value stays a ``str``, even when it looks like YAML."""
        cfg = SkyRLTrainConfig.from_cli_overrides({"generator.inference_engine.served_model_name": value})
        assert cfg.generator.inference_engine.served_model_name == value

    def test_scalar_and_container_values_keep_their_types(self):
        cfg = SkyRLTrainConfig.from_cli_overrides(
            {
                "generator.inference_engine.max_num_seqs": 512,
                "generator.sampling_params.temperature": 0.7,
                "generator.inference_engine.enforce_eager": True,
                "generator.inference_engine.enable_prefix_caching": False,
                "generator.inference_engine.external_server_urls": ["http://a:1", "http://b:2"],
                "generator.inference_engine.engine_init_kwargs": {"a": 1, "b": "x"},
                "trainer.policy.model.path": "Qwen/Qwen2.5-1.5B-Instruct",
            }
        )
        ie_cfg = cfg.generator.inference_engine
        assert ie_cfg.max_num_seqs == 512
        assert cfg.generator.sampling_params.temperature == 0.7
        assert ie_cfg.enforce_eager is True
        assert ie_cfg.enable_prefix_caching is False
        assert ie_cfg.external_server_urls == ["http://a:1", "http://b:2"]
        assert ie_cfg.engine_init_kwargs == {"a": 1, "b": "x"}
        assert cfg.trainer.policy.model.path == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_non_json_serializable_values_fall_back_to_str(self):
        """Values ``json.dumps`` cannot handle serialize via ``str()``."""
        cfg = SkyRLTrainConfig.from_cli_overrides({"trainer.export_path": pathlib.Path("/tmp/export")})
        assert cfg.trainer.export_path == "/tmp/export"

    @pytest.mark.parametrize(
        ("key", "dict_value", "dotlist_arg", "expected"),
        [
            pytest.param(
                "generator.inference_engine.external_server_urls",
                None,
                "generator.inference_engine.external_server_urls=null",
                None,
                id="none",
            ),
            pytest.param(
                "generator.inference_engine.served_model_name",
                "null",
                "generator.inference_engine.served_model_name='null'",
                "null",
                id="str-null",
            ),
            pytest.param(
                "generator.inference_engine.external_server_urls",
                ["http://a:1"],
                "generator.inference_engine.external_server_urls=['http://a:1']",
                ["http://a:1"],
                id="list",
            ),
        ],
    )
    def test_dict_and_dotlist_paths_agree(self, key, dict_value, dotlist_arg, expected):
        """A dict override matches the dotlist spelling of the same value."""
        from_dict = SkyRLTrainConfig.from_cli_overrides({key: dict_value})
        from_list = SkyRLTrainConfig.from_cli_overrides([dotlist_arg])
        assert _get_nested_attr(from_dict, key) == expected
        assert _get_nested_attr(from_list, key) == expected

    def test_plus_prefix_rejected_from_dict(self):
        """``'+'``-prefixed keys are rejected on the dict path."""
        with pytest.raises(ValueError, match="The '\\+' prefix"):
            SkyRLTrainConfig.from_cli_overrides({"+new_field": "value"})


class TestTrainerUseSamplePackingAlias:
    """`trainer.use_sample_packing` is a deprecated alias for `trainer.remove_microbatch_padding`
    on the RL entrypoint config (mirrors the ``fsdp2``->``fsdp`` alias)."""

    def test_trainer_use_sample_packing_remapped_with_warning(self):
        with pytest.warns(DeprecationWarning, match="trainer.use_sample_packing.*has been renamed"):
            cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.use_sample_packing=false"])
        assert cfg.trainer.remove_microbatch_padding is False

    def test_trainer_use_sample_packing_remapped_from_dict(self):
        # The Tinker backend passes overrides as a dict of dotted keys.
        with pytest.warns(DeprecationWarning, match="trainer.use_sample_packing.*has been renamed"):
            cfg = SkyRLTrainConfig.from_cli_overrides({"trainer.use_sample_packing": True})
        assert cfg.trainer.remove_microbatch_padding is True

    def test_trainer_use_sample_packing_with_new_key_raises(self):
        with pytest.raises(ValueError, match="only one of trainer.use_sample_packing"):
            SkyRLTrainConfig.from_cli_overrides(
                ["trainer.use_sample_packing=true", "trainer.remove_microbatch_padding=false"]
            )


class TestSkyRLTrainConfig:
    @pytest.mark.parametrize(
        ("overrides", "expected_num_workers", "expected_persistent"),
        [
            pytest.param([], 8, False, id="default-no-http"),
            pytest.param(["data.dataloader.num_workers=0"], 0, False, id="explicit-zero"),
            pytest.param(["data.dataloader.persistent_workers=true"], 8, True, id="persistent-keeps-default-workers"),
        ],
    )
    def test_resolution(self, overrides: list[str], expected_num_workers: int, expected_persistent: bool) -> None:
        cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
        assert cfg.data.dataloader.num_workers == expected_num_workers
        assert cfg.data.dataloader.persistent_workers == expected_persistent

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            pytest.param(
                ["data.dataloader.num_workers=0", "data.dataloader.persistent_workers=true"],
                "persistent_workers requires num_workers > 0",
                id="persistent-without-workers",
            ),
            pytest.param(["data.dataloader.num_workers=-1"], "num_workers must be None or >= 0", id="negative-workers"),
        ],
    )
    def test_invalid_raises(self, overrides: list[str], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SkyRLTrainConfig.from_cli_overrides(overrides)


class TestMaxSeqLenValidation:
    """Tests for max_seq_len defaults and validation behavior."""

    def test_max_seq_len_defaults_to_none_when_not_set(self):
        cfg = SkyRLTrainConfig.from_cli_overrides([])
        assert cfg.trainer.algorithm.max_seq_len is None

    def test_max_seq_len_preserved_when_explicitly_set(self):
        cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.algorithm.max_seq_len=32768"])
        assert cfg.trainer.algorithm.max_seq_len == 32768

    def test_validate_cfg_requires_explicit_max_seq_len_for_seq_mean_token_sum_norm(self):
        cfg = _make_validated_test_config()
        cfg.trainer.algorithm.loss_reduction = "seq_mean_token_sum_norm"
        cfg.trainer.algorithm.max_seq_len = None

        with pytest.raises(ValueError, match=r"trainer\.algorithm\.max_seq_len"):
            validate_cfg(cfg)

    @pytest.mark.parametrize("loss_reduction", ["token_mean", "sequence_mean"])
    def test_validate_cfg_allows_missing_max_seq_len_for_other_reductions(self, loss_reduction):
        cfg = _make_validated_test_config()
        cfg.trainer.algorithm.loss_reduction = loss_reduction
        cfg.trainer.algorithm.max_seq_len = None

        validate_cfg(cfg)

    def test_validate_cfg_allows_explicit_max_seq_len_for_seq_mean_token_sum_norm(self):
        cfg = _make_validated_test_config()
        cfg.trainer.algorithm.loss_reduction = "seq_mean_token_sum_norm"
        cfg.trainer.algorithm.max_seq_len = 4096

        validate_cfg(cfg)


class TestTorchProfilerConfigValidation:
    """TorchProfilerConfig validation coverage."""

    @staticmethod
    def _cfg(**overrides):
        from skyrl.train.config.config import TorchProfilerConfig

        # Valid default for tests targeting other fields.
        overrides.setdefault("save_path", "/tmp/skyrl_prof_test")
        return TorchProfilerConfig(enable=True, **overrides)

    def test_disabled_skips_all_checks(self):
        from skyrl.train.config.config import TorchProfilerConfig

        TorchProfilerConfig(
            enable=False, export_type="bogus", activities=["gpu"], ranks=[], active=0, save_path=None
        ).validate()

    def test_defaults_are_valid_when_enabled(self):
        self._cfg().validate()  # must not raise

    def test_empty_ranks_rejected(self):
        with pytest.raises(ValueError, match=r"ranks.*non-empty"):
            self._cfg(ranks=[]).validate()

    def test_missing_save_path_rejected(self):
        with pytest.raises(ValueError, match=r"save_path.*must be set"):
            self._cfg(save_path=None).validate()
        with pytest.raises(ValueError, match=r"save_path.*must be set"):
            self._cfg(save_path="").validate()

    def test_cloud_save_path_rejected(self):
        for uri in ("s3://bucket/run/traces", "gs://bucket/run/traces", "gcs://bucket/run/traces"):
            with pytest.raises(ValueError, match=r"save_path.*local path"):
                self._cfg(save_path=uri).validate()

    def test_unknown_activity_rejected(self):
        with pytest.raises(ValueError, match=r"activities"):
            self._cfg(activities=["cpu", "gpu"]).validate()

    def test_empty_activities_rejected(self):
        with pytest.raises(ValueError, match=r"activities.*non-empty"):
            self._cfg(activities=[]).validate()

    def test_activities_case_insensitive(self):
        self._cfg(activities=["CPU", "CUDA"]).validate()  # must not raise

    def test_unknown_export_type_rejected(self):
        with pytest.raises(ValueError, match=r"export_type"):
            self._cfg(export_type="bogus").validate()

    def test_stacks_requires_with_stack(self):
        with pytest.raises(ValueError, match=r"with_stack"):
            self._cfg(export_type="stacks", with_stack=False).validate()
        self._cfg(export_type="stacks", with_stack=True).validate()

    def test_negative_schedule_field_rejected(self):
        with pytest.raises(ValueError, match=r"skip_first"):
            self._cfg(skip_first=-1).validate()

    def test_active_must_be_at_least_one(self):
        with pytest.raises(ValueError, match=r"active"):
            self._cfg(active=0).validate()

    def test_validate_cfg_invokes_profiler_validation(self):
        cfg = _make_validated_test_config()
        cfg.trainer.policy.torch_profiler_config.enable = True
        cfg.trainer.policy.torch_profiler_config.save_path = "/tmp/skyrl_prof_test"
        cfg.trainer.policy.torch_profiler_config.export_type = "bogus"
        with pytest.raises(ValueError, match=r"export_type"):
            validate_cfg(cfg)

    # FSDP manual-offload incompatibility.

    def test_fsdp_colocate_all_manual_offload_rejected(self):
        with pytest.raises(ValueError, match=r"Couldn't swap"):
            self._cfg().validate(strategy="fsdp", colocate_all=True, colocate_policy_ref=True, fsdp_cpu_offload=False)

    def test_fsdp_colocate_policy_ref_only_rejected(self):
        with pytest.raises(ValueError, match=r"Couldn't swap"):
            self._cfg().validate(strategy="fsdp", colocate_all=False, colocate_policy_ref=True, fsdp_cpu_offload=False)

    def test_fsdp_no_colocation_allowed(self):
        self._cfg().validate(strategy="fsdp", colocate_all=False, colocate_policy_ref=False, fsdp_cpu_offload=False)

    def test_fsdp_native_cpu_offload_allowed(self):
        self._cfg().validate(strategy="fsdp", colocate_all=True, colocate_policy_ref=True, fsdp_cpu_offload=True)

    def test_megatron_colocation_allowed(self):
        self._cfg().validate(strategy="megatron", colocate_all=True, colocate_policy_ref=True, fsdp_cpu_offload=False)

    def test_offload_check_skipped_without_context(self):
        self._cfg().validate()

    def test_validate_cfg_rejects_profiler_under_default_colocation(self):
        cfg = _make_validated_test_config()
        cfg.trainer.policy.torch_profiler_config.enable = True
        cfg.trainer.policy.torch_profiler_config.save_path = "/tmp/skyrl_prof_test"
        with pytest.raises(ValueError, match=r"Couldn't swap"):
            validate_cfg(cfg)

    def test_validate_cfg_allows_profiler_with_native_offload(self):
        cfg = _make_validated_test_config()
        cfg.trainer.policy.torch_profiler_config.enable = True
        cfg.trainer.policy.torch_profiler_config.save_path = "/tmp/skyrl_prof_test"
        cfg.trainer.policy.fsdp_config.cpu_offload = True
        validate_cfg(cfg)


class TestDeltaWeightSyncConfig:
    """Tests for `DeltaWeightSyncConfig`"""

    def test_delta_weight_sync_defaults(self):
        cfg = DeltaWeightSyncConfig(sync_dir="my_sync_dir", publish_staging_dir=None, local_checkpoint_dir=None)
        assert cfg.publish_staging_dir is not None
        assert cfg.local_checkpoint_dir is not None
        # `publish_staging_dir` and `local_checkpoint_dir` should be constructed based on `sync_dir`
        assert "my_sync_dir" in cfg.publish_staging_dir
        assert "my_sync_dir" in cfg.local_checkpoint_dir
