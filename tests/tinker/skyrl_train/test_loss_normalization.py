"""Unit tests for SkyRLTrainBackend._normalize_policy_loss_request.

No Ray runtime or GPUs are needed — the method is pure. Requires the
SkyRL-Train backend deps (ray/vllm) to be importable. Run:
  uv run --extra dev --extra fsdp pytest tests/tinker/skyrl_train/test_loss_normalization.py
"""

from __future__ import annotations

import pytest
import torch

# Skip if skyrl_train_backend.py cannot be imported
skyrl_train_backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")

_normalize = skyrl_train_backend.SkyRLTrainBackend._normalize_policy_loss_request


def test_ppo_thresholds_map_to_eps_clip():
    loss_fn, config = _normalize(None, "policy", "ppo", {"clip_low_threshold": 0.8, "clip_high_threshold": 1.28})
    assert loss_fn == "regular"
    assert config == pytest.approx({"eps_clip_low": 0.2, "eps_clip_high": 0.28})


def test_gspo_thresholds_map_to_native_sequence_loss():
    loss_fn, config = _normalize(
        None,
        "policy",
        "gspo",
        {"clip_low_threshold": 0.98, "clip_high_threshold": 1.03},
    )
    assert loss_fn == "gspo"
    assert config == {
        "eps_clip_low": pytest.approx(0.02),
        "eps_clip_high": pytest.approx(0.03),
        "loss_reduction": "sequence_mean",
        "use_entropy_loss": False,
        "use_kl_loss": False,
    }


def test_gspo_batch_normalization_gives_each_trainable_sequence_equal_weight():
    batch = skyrl_train_backend.TrainingInputBatch(
        {
            "advantages": torch.tensor([[2.0, 2.0, 0.0, 0.0], [-1.0, -1.0, -1.0, 0.0], [0.0] * 4]),
            "loss_mask": torch.ones(3, 4),
        }
    )

    skyrl_train_backend.SkyRLTrainBackend._normalize_gspo_batch(batch)

    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0] * 4]
    assert (batch["advantages"] * batch["loss_mask"]).sum(dim=-1).tolist() == pytest.approx([1.0, -0.5, 0.0])


def test_gspo_selects_only_rows_with_gradient_contributions():
    batch = skyrl_train_backend.TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2], [3, 0], [4, 5]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0], [1, 1]]),
            "loss_mask": torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        }
    )
    batch.metadata = {"response_length": 2}

    selected, indices = skyrl_train_backend.SkyRLTrainBackend._select_trainable_gspo_rows(batch)

    assert indices == [0, 2]
    assert selected["sequences"].tolist() == [[1, 2], [4, 5]]
    assert selected.metadata == batch.metadata


def test_gspo_keeps_entirely_inactive_batch_for_zero_gradient_backward():
    batch = skyrl_train_backend.TrainingInputBatch(
        {
            "sequences": torch.tensor([[1, 2], [3, 4]]),
            "loss_mask": torch.zeros(2, 2),
        }
    )

    selected, indices = skyrl_train_backend.SkyRLTrainBackend._select_trainable_gspo_rows(batch)

    assert selected is batch
    assert indices == [0, 1]


def test_gspo_pruning_metrics_sum_across_sdk_chunks_without_duplication():
    first = skyrl_train_backend.SkyRLTrainBackend._get_gspo_pruning_metrics([3, 2, 4, 1], {0, 2}, 0, 2)
    second = skyrl_train_backend.SkyRLTrainBackend._get_gspo_pruning_metrics([3, 2, 4, 1], {0, 2}, 2, 4)

    assert first == {
        "skyrl.ai/submitted_datums:sum": 1.0,
        "skyrl.ai/skipped_datums:sum": 1.0,
        "skyrl.ai/submitted_tokens:sum": 3.0,
        "skyrl.ai/skipped_tokens:sum": 2.0,
    }
    assert second == {
        "skyrl.ai/submitted_datums:sum": 1.0,
        "skyrl.ai/skipped_datums:sum": 1.0,
        "skyrl.ai/submitted_tokens:sum": 4.0,
        "skyrl.ai/skipped_tokens:sum": 1.0,
    }


def test_dppo_deltas_are_nested_under_dppo():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_low": 0.2, "delta_high": 0.3})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_low": 0.2, "delta_high": 0.3}}


def test_dppo_partial_deltas():
    loss_fn, config = _normalize(None, "policy", "dppo", {"delta_high": 0.05})
    assert loss_fn == "dppo"
    assert config == {"dppo": {"delta_high": 0.05}}


def test_dppo_without_config_passes_through():
    loss_fn, config = _normalize(None, "policy", "dppo", None)
    assert loss_fn == "dppo"
    assert config is None


def test_critic_config_passes_through_unchanged():
    loss_fn, config = _normalize(None, "critic", "ppo", {"value_clip": 0.2})
    assert loss_fn == "ppo"
    assert config == {"value_clip": 0.2}
