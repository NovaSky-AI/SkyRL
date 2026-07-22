"""
CPU tests for build_skyrl_config_for_sft override wiring.

Verifies that SFTConfig fields (top-level, nested, and deeply nested) are
correctly bridged to the internal SkyRLTrainConfig by build_skyrl_config_for_sft.

uv run --isolated --extra dev pytest tests/train/test_sft_config.py -v
"""

import pytest

from skyrl.train.config import (
    SFTConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.config.sft_config import _normalize_dataset_cfg, validate_sft_cfg


def _sft_cfg_from_overrides(overrides: list[str]) -> SFTConfig:
    """Build an SFTConfig from CLI-style overrides."""
    return SFTConfig.from_cli_overrides(overrides)


class TestFSDP2StrategyAlias:
    """`strategy='fsdp2'` is accepted as a deprecated alias for `'fsdp'`."""

    def test_fsdp2_normalized_to_fsdp_with_warning(self):
        cfg = SFTConfig()
        cfg.strategy = "fsdp2"
        cfg.model.path = "test/my-model"
        with pytest.warns(DeprecationWarning, match="fsdp2.*has been renamed"):
            validate_sft_cfg(cfg)
        assert cfg.strategy == "fsdp"


class TestUseSamplePackingAlias:
    """`use_sample_packing` is accepted as a deprecated alias for `remove_microbatch_padding`."""

    def test_use_sample_packing_remapped_with_warning(self):
        with pytest.warns(DeprecationWarning, match="use_sample_packing.*has been renamed"):
            cfg = _sft_cfg_from_overrides(["use_sample_packing=true"])
        assert cfg.remove_microbatch_padding is True

    def test_use_sample_packing_remapped_from_dict(self):
        with pytest.warns(DeprecationWarning, match="use_sample_packing.*has been renamed"):
            cfg = SFTConfig.from_cli_overrides({"use_sample_packing": False})
        assert cfg.remove_microbatch_padding is False

    def test_use_sample_packing_with_new_key_raises(self):
        with pytest.raises(ValueError, match="only one of use_sample_packing"):
            _sft_cfg_from_overrides(["use_sample_packing=true", "remove_microbatch_padding=false"])


class TestTopLevelOverrides:
    """Top-level SFTConfig fields bridge to the correct SkyRLTrainConfig paths."""

    def test_model_path_bridges_to_policy_model(self):
        cfg = _sft_cfg_from_overrides(["model.path=test/my-model"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.model.path == "test/my-model"

    def test_remove_microbatch_padding_propagates(self):
        cfg = _sft_cfg_from_overrides(["remove_microbatch_padding=false"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.remove_microbatch_padding is False

        cfg_on = _sft_cfg_from_overrides(["remove_microbatch_padding=true"])
        skyrl_cfg_on = build_skyrl_config_for_sft(cfg_on)
        assert skyrl_cfg_on.trainer.remove_microbatch_padding is True


class TestMegatronConfigOverrides:
    """Megatron parallelism config overrides propagate correctly."""

    def test_tensor_model_parallel_size(self):
        cfg = _sft_cfg_from_overrides(
            [
                "megatron_config.tensor_model_parallel_size=4",
                "megatron_config.pipeline_model_parallel_size=1",
                "placement.num_gpus_per_node=4",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.megatron_config.tensor_model_parallel_size == 4

    def test_ddp_config_overlap_grad_reduce(self):
        """Deeply nested: megatron_config.ddp_config.overlap_grad_reduce."""
        cfg = _sft_cfg_from_overrides(
            [
                "megatron_config.ddp_config.overlap_grad_reduce=true",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.megatron_config.ddp_config.overlap_grad_reduce is True


class TestOptimizerConfigOverrides:
    """Optimizer config overrides propagate to policy.optimizer_config."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("lr", "1e-4", 1e-4),
            ("scheduler", "cosine", "cosine"),
            ("num_warmup_steps", "100", 100),
        ],
    )
    def test_optimizer_fields(self, field, value, expected):
        cfg = _sft_cfg_from_overrides([f"optimizer_config.{field}={value}"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        actual = getattr(skyrl_cfg.trainer.policy.optimizer_config, field)
        assert actual == expected


class TestLoraConfigOverrides:
    """LoRA config overrides propagate through model.lora → policy.model.lora."""

    def test_lora_rank_and_alpha_propagate(self):
        cfg = _sft_cfg_from_overrides(["model.path=test/my-model", "model.lora.rank=32", "model.lora.alpha=64"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.model.lora.rank == 32
        assert skyrl_cfg.trainer.policy.model.lora.alpha == 64

    def test_lora_target_modules_propagate(self):
        cfg = _sft_cfg_from_overrides(
            ["model.path=test/my-model", "model.lora.rank=16", "model.lora.target_modules=all-linear"]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.model.lora.target_modules == "all-linear"

    def test_lora_disabled_by_default(self):
        cfg = _sft_cfg_from_overrides([])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.model.lora.rank == 0


class TestTorchProfilerConfigOverrides:
    """SFT profiler config bridge coverage."""

    def test_disabled_by_default(self):
        cfg = _sft_cfg_from_overrides([])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.torch_profiler_config.enable is False

    def test_enable_and_schedule_propagate(self):
        cfg = _sft_cfg_from_overrides(
            [
                "torch_profiler_config.enable=true",
                "torch_profiler_config.skip_first=3",
                "torch_profiler_config.active=2",
                "torch_profiler_config.repeat=0",
                "torch_profiler_config.save_path=/tmp/sft_prof",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        prof = skyrl_cfg.trainer.policy.torch_profiler_config
        assert prof.enable is True
        assert prof.skip_first == 3
        assert prof.active == 2
        assert prof.repeat == 0
        assert prof.save_path == "/tmp/sft_prof"

    def test_capture_flags_propagate(self):
        cfg = _sft_cfg_from_overrides(
            [
                "torch_profiler_config.enable=true",
                "torch_profiler_config.profile_memory=true",
                "torch_profiler_config.with_stack=false",
                "torch_profiler_config.activities=[cuda]",
                "torch_profiler_config.save_path=/tmp/sft_prof",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        prof = skyrl_cfg.trainer.policy.torch_profiler_config
        assert prof.profile_memory is True
        assert prof.with_stack is False
        assert list(prof.activities) == ["cuda"]

    def test_invalid_profiler_config_rejected_by_sft_validation(self):
        cfg = _sft_cfg_from_overrides(
            [
                "model.path=test/my-model",
                "torch_profiler_config.enable=true",
                "torch_profiler_config.export_type=bogus",
                "torch_profiler_config.save_path=/tmp/sft_prof",
            ]
        )
        with pytest.raises(ValueError, match=r"export_type"):
            build_skyrl_config_for_sft(cfg)


class TestFSDPConfigOverrides:
    """FSDP config overrides propagate when strategy=fsdp."""

    def test_cpu_offload(self):
        cfg = _sft_cfg_from_overrides(
            [
                "strategy=fsdp",
                "fsdp_config.cpu_offload=true",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.fsdp_config.cpu_offload is True

    def test_reshard_after_forward(self):
        cfg = _sft_cfg_from_overrides(
            [
                "strategy=fsdp",
                "fsdp_config.reshard_after_forward=false",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.fsdp_config.reshard_after_forward is False


class TestMaxTokensPerMicrobatch:
    """``max_tokens_per_microbatch`` is the FFD bin capacity.

    It must be ``>= max_length`` (any single sequence fits in a bin) but need
    not be a multiple of ``max_length``.
    """

    def _packed_cfg(self, **overrides) -> SFTConfig:
        cfg = SFTConfig(
            strategy="megatron",
            max_length=128,
            remove_microbatch_padding=True,
            use_sequence_packing=True,
        )
        cfg.model.path = "test/my-model"
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_none_resolves_to_max_length(self):
        cfg = self._packed_cfg(max_tokens_per_microbatch=None)
        validate_sft_cfg(cfg)
        assert cfg.resolved_bin_capacity() == 128

    def test_equal_to_max_length_accepted(self):
        cfg = self._packed_cfg(max_tokens_per_microbatch=128)
        validate_sft_cfg(cfg)
        assert cfg.resolved_bin_capacity() == 128

    def test_non_multiple_above_max_length_accepted(self):
        # The old "must be a multiple of max_length" rule is gone: any budget
        # >= max_length is a valid bin capacity.
        cfg = self._packed_cfg(max_tokens_per_microbatch=200)
        validate_sft_cfg(cfg)
        assert cfg.resolved_bin_capacity() == 200

    def test_below_max_length_rejected(self):
        cfg = self._packed_cfg(max_tokens_per_microbatch=64)
        with pytest.raises(ValueError, match="must be >= max_length"):
            validate_sft_cfg(cfg)

    def test_bridge_sets_worker_micro_batch_size_to_one(self):
        # Each bin row is one worker micro-batch.
        cfg = self._packed_cfg(max_tokens_per_microbatch=256, micro_train_batch_size_per_gpu=4)
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.micro_train_batch_size_per_gpu == 1


class TestDatasetConfigNormalization:
    """Deprecated single-dataset fields translate to the list-based fields (RFC #1875)."""

    def test_zero_config_defaults(self):
        cfg = SFTConfig()
        _normalize_dataset_cfg(cfg)
        assert cfg.train_datasets == ["yahma/alpaca-cleaned"]
        assert cfg.train_dataset_splits == ["train[:100]"]
        assert cfg.train_dataset_weights == [1.0]
        assert cfg.dataset_name is None and cfg.dataset_split is None
        assert cfg.eval_datasets is None  # eval stays disabled

    def test_deprecated_train_fields_translate_with_warning(self):
        with pytest.warns(DeprecationWarning, match="dataset_name/dataset_split are deprecated"):
            cfg = _sft_cfg_from_overrides(["dataset_name=foo/bar", "dataset_split=train[:50]"])
            _normalize_dataset_cfg(cfg)
        assert cfg.train_datasets == ["foo/bar"]
        assert cfg.train_dataset_splits == ["train[:50]"]
        assert cfg.dataset_name is None and cfg.dataset_split is None

    def test_deprecated_translation_works_programmatically(self):
        cfg = SFTConfig()
        cfg.dataset_name = "foo/bar"
        with pytest.warns(DeprecationWarning, match="dataset_name/dataset_split are deprecated"):
            validate_sft_cfg(cfg)
        assert cfg.train_datasets == ["foo/bar"]
        assert cfg.train_dataset_splits == ["train[:100]"]  # split default preserved

    def test_deprecated_eval_fields_translate_with_warning(self):
        with pytest.warns(DeprecationWarning, match="eval_dataset_name/eval_dataset_split are deprecated"):
            cfg = _sft_cfg_from_overrides(["eval_dataset_name=baz/qux"])
            _normalize_dataset_cfg(cfg)
        assert cfg.eval_datasets == ["baz/qux"]
        assert cfg.eval_dataset_splits == ["validation"]  # old default preserved
        assert cfg.eval_dataset_names == ["baz_qux"]  # "/" sanitized for logging
        assert cfg.eval_dataset_name is None and cfg.eval_dataset_split is None

    def test_old_and_new_train_fields_conflict(self):
        cfg = _sft_cfg_from_overrides(["dataset_name=x", "train_datasets=[a]", "train_dataset_splits=[s]"])
        with pytest.raises(ValueError, match="only one of train_datasets"):
            _normalize_dataset_cfg(cfg)

    def test_old_and_new_eval_fields_conflict(self):
        cfg = _sft_cfg_from_overrides(["eval_dataset_name=x", "eval_datasets=[a]", "eval_dataset_splits=[s]"])
        with pytest.raises(ValueError, match="only one of eval_datasets"):
            _normalize_dataset_cfg(cfg)

    def test_idempotent(self):
        cfg = _sft_cfg_from_overrides(["train_datasets=[a,b]", "train_dataset_splits=[s1,s2]", "sampler=sequential"])
        _normalize_dataset_cfg(cfg)
        first = (cfg.train_datasets, cfg.train_dataset_splits, cfg.train_dataset_weights)
        _normalize_dataset_cfg(cfg)
        assert (cfg.train_datasets, cfg.train_dataset_splits, cfg.train_dataset_weights) == first


class TestMultiDatasetValidation:
    """List-shaped dataset fields are validated for lengths, weights and names."""

    def test_bracketed_splits_parse_with_inner_quotes(self):
        # HF slice syntax nests brackets inside the OmegaConf list, so each
        # element must be quoted on the CLI: "train_dataset_splits=['train[:2000]']".
        # (Unquoted elements with brackets are a YAML parse error.)
        cfg = _sft_cfg_from_overrides(
            [
                "train_datasets=['allenai/tulu-3-sft-mixture','yahma/alpaca-cleaned']",
                "train_dataset_splits=['train[:50000]','train[:10000]']",
            ]
        )
        _normalize_dataset_cfg(cfg)
        assert list(cfg.train_datasets) == ["allenai/tulu-3-sft-mixture", "yahma/alpaca-cleaned"]
        assert list(cfg.train_dataset_splits) == ["train[:50000]", "train[:10000]"]

    def test_default_weights_are_equal_mixing(self):
        cfg = _sft_cfg_from_overrides(["train_datasets=[a,b,c,d]", "train_dataset_splits=[s,s,s,s]"])
        _normalize_dataset_cfg(cfg)
        assert cfg.train_dataset_weights == pytest.approx([0.25] * 4)

    def test_splits_length_mismatch_rejected(self):
        cfg = _sft_cfg_from_overrides(["train_datasets=[a,b]", "train_dataset_splits=[s1]"])
        with pytest.raises(ValueError, match="one split per entry"):
            _normalize_dataset_cfg(cfg)

    def test_missing_splits_rejected(self):
        cfg = _sft_cfg_from_overrides(["train_datasets=[a,b]"])
        with pytest.raises(ValueError, match="one split per entry"):
            _normalize_dataset_cfg(cfg)

    def test_weights_length_mismatch_rejected(self):
        cfg = _sft_cfg_from_overrides(
            ["train_datasets=[a,b]", "train_dataset_splits=[s,s]", "train_dataset_weights=[1.0]"]
        )
        with pytest.raises(ValueError, match="one weight per entry"):
            _normalize_dataset_cfg(cfg)

    def test_nonpositive_weights_rejected(self):
        cfg = _sft_cfg_from_overrides(
            ["train_datasets=[a,b]", "train_dataset_splits=[s,s]", "train_dataset_weights=[1.0,0.0]"]
        )
        with pytest.raises(ValueError, match="must all be > 0"):
            _normalize_dataset_cfg(cfg)

    def test_weights_require_random_sampler(self):
        cfg = _sft_cfg_from_overrides(
            [
                "train_datasets=[a,b]",
                "train_dataset_splits=[s,s]",
                "train_dataset_weights=[0.5,0.5]",
                "sampler=sequential",
            ]
        )
        with pytest.raises(ValueError, match="only supported with sampler='random'"):
            _normalize_dataset_cfg(cfg)

    def test_eval_names_default_collision_rejected(self):
        cfg = _sft_cfg_from_overrides(["eval_datasets=[a/b,a/b]", "eval_dataset_splits=[s1,s2]"])
        with pytest.raises(ValueError, match="names collide"):
            _normalize_dataset_cfg(cfg)

    def test_eval_names_explicit_disambiguate(self):
        cfg = _sft_cfg_from_overrides(
            ["eval_datasets=[a/b,a/b]", "eval_dataset_splits=[s1,s2]", "eval_dataset_names=[early,late]"]
        )
        _normalize_dataset_cfg(cfg)
        assert cfg.eval_dataset_names == ["early", "late"]

    def test_eval_names_duplicates_rejected(self):
        cfg = _sft_cfg_from_overrides(["eval_datasets=[a,b]", "eval_dataset_splits=[s,s]", "eval_dataset_names=[x,x]"])
        with pytest.raises(ValueError, match="must be unique"):
            _normalize_dataset_cfg(cfg)

    def test_eval_names_length_mismatch_rejected(self):
        cfg = _sft_cfg_from_overrides(["eval_datasets=[a,b]", "eval_dataset_splits=[s,s]", "eval_dataset_names=[x]"])
        with pytest.raises(ValueError, match="one name per entry"):
            _normalize_dataset_cfg(cfg)

    def test_eval_lists_without_eval_datasets_rejected(self):
        cfg = _sft_cfg_from_overrides(["eval_dataset_splits=[s]"])
        with pytest.raises(ValueError, match="require eval_datasets"):
            _normalize_dataset_cfg(cfg)

    def test_eval_interval_requires_eval_datasets(self):
        cfg = _sft_cfg_from_overrides(["eval_interval=5"])
        with pytest.raises(ValueError, match="requires eval_datasets"):
            validate_sft_cfg(cfg)
