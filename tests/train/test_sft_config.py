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


def _sft_cfg_from_overrides(overrides: list[str]) -> SFTConfig:
    """Build an SFTConfig from CLI-style overrides."""
    return SFTConfig.from_cli_overrides(overrides)


class TestTopLevelOverrides:
    """Top-level SFTConfig fields bridge to the correct SkyRLTrainConfig paths."""

    def test_model_path_bridges_to_policy_model(self):
        cfg = _sft_cfg_from_overrides(["model.path=test/my-model"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.model.path == "test/my-model"

    def test_use_sample_packing_propagates(self):
        cfg = _sft_cfg_from_overrides(["use_sample_packing=false"])
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.use_sample_packing is False

        cfg_on = _sft_cfg_from_overrides(["use_sample_packing=true"])
        skyrl_cfg_on = build_skyrl_config_for_sft(cfg_on)
        assert skyrl_cfg_on.trainer.use_sample_packing is True


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
    """FSDP config overrides propagate when strategy=fsdp2."""

    def test_cpu_offload(self):
        cfg = _sft_cfg_from_overrides(
            [
                "strategy=fsdp2",
                "fsdp_config.cpu_offload=true",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.fsdp_config.cpu_offload is True

    def test_reshard_after_forward(self):
        cfg = _sft_cfg_from_overrides(
            [
                "strategy=fsdp2",
                "fsdp_config.reshard_after_forward=false",
            ]
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        assert skyrl_cfg.trainer.policy.fsdp_config.reshard_after_forward is False
