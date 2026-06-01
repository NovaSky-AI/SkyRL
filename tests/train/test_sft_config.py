"""
CPU tests for build_skyrl_config_for_sft override wiring.

Verifies that SFTConfig fields (top-level, nested, and deeply nested) are
correctly bridged to the internal SkyRLTrainConfig by build_skyrl_config_for_sft.

uv run --isolated --extra dev pytest tests/train/test_sft_config.py -v
"""

import pytest

from skyrl.train.config import (
    SFTConfig,
    SkyRLTrainConfig,
    build_skyrl_config_for_sft,
)
from skyrl.train.config.sft_config import validate_sft_cfg


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


class TestSequencePackingAutoEnablesRemoveMicrobatchPadding:
    """``use_sequence_packing=True`` implies the THD layout, so it auto-enables
    ``remove_microbatch_padding`` (with a warning) instead of erroring."""

    def _packed_cfg(self, remove_microbatch_padding: bool) -> SFTConfig:
        cfg = SFTConfig(
            strategy="megatron",
            max_length=128,
            remove_microbatch_padding=remove_microbatch_padding,
            use_sequence_packing=True,
        )
        cfg.model.path = "test/my-model"
        return cfg

    def test_auto_sets_when_left_default(self):
        cfg = self._packed_cfg(remove_microbatch_padding=True)
        validate_sft_cfg(cfg)
        assert cfg.remove_microbatch_padding is True

    def test_auto_corrects_explicit_false_with_warning(self, caplog):
        cfg = self._packed_cfg(remove_microbatch_padding=False)
        validate_sft_cfg(cfg)
        assert cfg.remove_microbatch_padding is True


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
