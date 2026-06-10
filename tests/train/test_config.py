"""
uv run --isolated --extra dev pytest -s tests/train/test_config.py
"""

import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Optional
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from skyrl.train.config.config import (
    BaseConfig,
    SkyRLTrainConfig,
    _resolve_class_type,
    build_nested_dataclass,
)
from skyrl.train.utils.utils import validate_cfg
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
            "trainer.rope_scaling={'type': 'linear'}",
            "trainer.rope_theta=10000",
        ]
    )

    assert cfg.generator.max_input_length == 1024  # same as `trainer.max_prompt_length`
    assert cfg.trainer.ref.model.path == "Qwen/Qwen2.5-1.5B-Instruct"  # same as `trainer.policy.model.path`
    assert (
        cfg.generator.eval_sampling_params.max_generate_length == cfg.generator.sampling_params.max_generate_length
    )  # same as `generator.sampling_params.max_generate_length`
    assert cfg.generator.rope_scaling == cfg.trainer.rope_scaling
    assert cfg.generator.rope_theta == cfg.trainer.rope_theta


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
            pytest.param(["generator.inference_engine.enable_http_endpoint=true"], 0, False, id="http-endpoint"),
            pytest.param(
                ["generator.inference_engine.enable_http_endpoint=true", "data.dataloader.num_workers=4"],
                4,
                False,
                id="explicit-overrides-http",
            ),
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


class TestRouterReplayValidation:
    """Tests for rollout routing replay (R3) config validation."""

    @pytest.mark.parametrize(
        "strategy",
        [pytest.param("fsdp", id="fsdp"), pytest.param("megatron", id="megatron")],
    )
    @pytest.mark.parametrize(
        "ref_replay_enabled",
        [pytest.param(False, id="ref_off_warns"), pytest.param(True, id="ref_on_no_warning")],
    )
    def test_validate_cfg_warns_on_r3_ref_kl_mismatch(self, strategy: str, ref_replay_enabled: bool) -> None:
        cfg = _make_validated_test_config()
        cfg.trainer.strategy = strategy
        cfg.trainer.algorithm.use_kl_loss = True
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.distributed_executor_backend = "mp"
        backend_config_attr = f"{strategy}_config"
        getattr(cfg.trainer.policy, backend_config_attr).moe_enable_routing_replay = True
        getattr(cfg.trainer.ref, backend_config_attr).moe_enable_routing_replay = ref_replay_enabled

        with patch("skyrl.train.utils.utils.logger") as mock_logger:
            validate_cfg(cfg)

        mismatch_warnings = [
            call for call in mock_logger.warning.call_args_list if "moe_enable_routing_replay" in call.args[0]
        ]
        assert (
            bool(mismatch_warnings) != ref_replay_enabled
        ), "the ref routing-mismatch warning should fire exactly when the ref is not replaying"
