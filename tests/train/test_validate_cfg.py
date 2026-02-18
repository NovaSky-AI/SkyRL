"""
Tests for validate_cfg logic.

uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_validate_cfg.py
"""

import pytest
from omegaconf import OmegaConf

from skyrl.train.config import (
    SkyRLTrainConfig,
    AlgorithmConfig,
    GeneratorConfig,
    SamplingParams,
    TrainerConfig,
    InferenceEngineConfig,
)
from skyrl.train.utils.utils import validate_cfg


def _make_valid_cfg(**algorithm_overrides) -> SkyRLTrainConfig:
    """Create a minimal SkyRLTrainConfig that passes validate_cfg.

    All defaults are chosen so that the full validate_cfg pipeline passes
    without needing GPUs, real model paths, etc.
    """
    algo_kwargs = dict(
        advantage_estimator="grpo",
        loss_reduction="token_mean",
        use_kl_loss=False,
        use_kl_in_reward=False,
        policy_loss_type="regular",
    )
    algo_kwargs.update(algorithm_overrides)

    trainer_cfg = TrainerConfig(
        project_name="unit-test",
        run_name="test-run",
        logger="tensorboard",
        train_batch_size=2,
        policy_mini_batch_size=2,
        micro_train_batch_size_per_gpu=2,
        micro_forward_batch_size_per_gpu=2,
        eval_batch_size=2,
        max_prompt_length=100,
        algorithm=AlgorithmConfig(**algo_kwargs),
    )
    generator_cfg = GeneratorConfig(
        sampling_params=SamplingParams(max_generate_length=200),
        n_samples_per_prompt=1,
        max_turns=1,
        max_input_length=100,
        inference_engine=InferenceEngineConfig(
            enable_http_endpoint=False,
            http_endpoint_host="127.0.0.1",
            http_endpoint_port=8000,
            tensor_parallel_size=1,
        ),
    )
    cfg = SkyRLTrainConfig(trainer=trainer_cfg, generator=generator_cfg)
    cfg.trainer.placement.colocate_all = False
    return cfg


def _maybe_to_dictconfig(cfg, use_dictconfig):
    """Optionally convert a SkyRLTrainConfig to DictConfig for parametrized tests."""
    if use_dictconfig:
        return OmegaConf.structured(cfg)
    return cfg


class TestMaxSeqLenValidation:
    """Tests for the max_seq_len auto-calculation and explicit-override logic in validate_cfg."""

    @pytest.mark.parametrize("use_dictconfig", [False, True], ids=["dataclass", "dictconfig"])
    def test_max_seq_len_auto_calculated_when_none(self, use_dictconfig):
        """When max_seq_len is None (default), validate_cfg should compute it as
        max_input_length + max_generate_length."""
        cfg = _maybe_to_dictconfig(_make_valid_cfg(max_seq_len=None), use_dictconfig)
        assert cfg.trainer.algorithm.max_seq_len is None

        validate_cfg(cfg)

        expected = cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
        assert cfg.trainer.algorithm.max_seq_len == expected
        assert cfg.trainer.algorithm.max_seq_len == 300  # 100 + 200

    @pytest.mark.parametrize("use_dictconfig", [False, True], ids=["dataclass", "dictconfig"])
    def test_max_seq_len_preserved_when_explicitly_set(self, use_dictconfig):
        """When max_seq_len is explicitly set by the user, validate_cfg should NOT overwrite it."""
        cfg = _maybe_to_dictconfig(_make_valid_cfg(max_seq_len=32768), use_dictconfig)

        validate_cfg(cfg)

        assert cfg.trainer.algorithm.max_seq_len == 32768
