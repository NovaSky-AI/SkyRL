"""
uv run --isolated --extra dev pytest -s tests/train/test_config.py
"""

import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Optional

import pytest
from omegaconf import OmegaConf

from skyrl.train.config.config import (
    BaseConfig,
    SkyRLTrainConfig,
    _resolve_class_type,
    build_nested_dataclass,
)
from skyrl.train.utils.utils import validate_cfg, validate_inference_engine_cfg
from tests.train.util import example_dummy_config


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


def _fireworks_backend_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.backend = "fireworks"
    cfg.generator.inference_engine.run_engines_locally = False
    cfg.generator.inference_engine.served_model_name = "accounts/fireworks/models/gpt-oss-20b"
    cfg.generator.inference_engine.api_key = "fw-key"
    cfg.generator.inference_engine.hf_tokenizer_name = "openai/gpt-oss-20b"
    return cfg


def test_fireworks_backend_valid_config_passes():
    validate_inference_engine_cfg(_fireworks_backend_cfg())

    # external_proxy_url is optional; a server root without /v1 is accepted.
    cfg = _fireworks_backend_cfg()
    cfg.generator.inference_engine.external_proxy_url = "http://localhost:8000"
    validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("run_engines_locally", True, "run_engines_locally=false"),
        ("served_model_name", None, "served_model_name"),
        ("api_key", None, "api_key"),
        ("hf_tokenizer_name", None, "hf_tokenizer_name"),
        ("external_proxy_url", "https://api.fireworks.ai/inference/v1", "server root"),
    ],
)
def test_fireworks_backend_invalid_configs_raise(field, value, match):
    cfg = _fireworks_backend_cfg()
    setattr(cfg.generator.inference_engine, field, value)
    with pytest.raises(AssertionError, match=match):
        validate_inference_engine_cfg(cfg)


def test_hf_tokenizer_name_requires_fireworks_backend():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.hf_tokenizer_name = "openai/gpt-oss-20b"
    with pytest.raises(AssertionError, match="only settable"):
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
