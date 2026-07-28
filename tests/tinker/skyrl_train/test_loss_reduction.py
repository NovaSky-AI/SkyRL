"""Unit tests for honoring ``trainer.algorithm.loss_reduction`` on the
Tinker-serving path (``SkyRLTrainBackend``).

The Tinker API sums per-token losses and expects pre-normalized advantages;
these tests cover the config default (``token_sum``, preserving that contract)
and the application of an explicitly configured reduction when the batch is
built. No Ray runtime or GPUs are needed — the functions under test are pure.
Requires the SkyRL-Train backend deps (ray/vllm) to be importable. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_loss_reduction.py
"""

from __future__ import annotations

import pytest
import torch

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

_apply_tinker_loss_reduction = skyrl_train_backend._apply_tinker_loss_reduction
_build_skyrl_train_config = skyrl_train_backend._build_skyrl_train_config
MegatronBackendOverrides = skyrl_train_backend.MegatronBackendOverrides

ADVANTAGES = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
LOSS_MASK = torch.tensor([[1.0, 1.0], [1.0, 0.0]])


class TestApplyTinkerLossReduction:
    def test_token_sum_is_identity(self):
        out = _apply_tinker_loss_reduction(ADVANTAGES, LOSS_MASK, "token_sum", micro_batch_size=1, max_seq_len=None)
        assert out is ADVANTAGES

    def test_token_mean_scales_by_masked_token_count(self):
        out = _apply_tinker_loss_reduction(ADVANTAGES, LOSS_MASK, "token_mean", micro_batch_size=1, max_seq_len=None)
        assert torch.allclose(out, ADVANTAGES / 3.0)

    def test_seq_mean_token_sum_norm_scales_by_constant(self):
        out = _apply_tinker_loss_reduction(
            ADVANTAGES, LOSS_MASK, "seq_mean_token_sum_norm", micro_batch_size=1, max_seq_len=32768
        )
        assert torch.allclose(out, ADVANTAGES / (2 * 32768))

    def test_prompt_mean_raises(self):
        with pytest.raises(ValueError, match="prompt_mean"):
            _apply_tinker_loss_reduction(ADVANTAGES, LOSS_MASK, "prompt_mean", micro_batch_size=1, max_seq_len=None)


class TestBuildSkyRLTrainConfigLossReduction:
    def test_loss_reduction_defaults_to_token_sum(self):
        cfg = _build_skyrl_train_config("Qwen/Qwen2.5-0.5B-Instruct", MegatronBackendOverrides())
        assert cfg.trainer.algorithm.loss_reduction == "token_sum"

    def test_explicit_loss_reduction_is_honored(self):
        overrides = MegatronBackendOverrides.model_validate(
            {
                "trainer.algorithm.loss_reduction": "seq_mean_token_sum_norm",
                "trainer.algorithm.max_seq_len": 32768,
            }
        )
        cfg = _build_skyrl_train_config("Qwen/Qwen2.5-0.5B-Instruct", overrides)
        assert cfg.trainer.algorithm.loss_reduction == "seq_mean_token_sum_norm"
        assert cfg.trainer.algorithm.max_seq_len == 32768

    def test_seq_mean_token_sum_norm_requires_max_seq_len(self):
        overrides = MegatronBackendOverrides.model_validate(
            {"trainer.algorithm.loss_reduction": "seq_mean_token_sum_norm"}
        )
        with pytest.raises(ValueError, match=r"trainer\.algorithm\.max_seq_len"):
            _build_skyrl_train_config("Qwen/Qwen2.5-0.5B-Instruct", overrides)
