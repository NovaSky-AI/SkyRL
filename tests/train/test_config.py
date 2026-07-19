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
