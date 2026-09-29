"""CPU tests for the VLM correctness fixes.

Run: uv run --extra dev pytest tests/train/test_vlm_correctness_guards.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    concat_nonempty_tensors,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    scope_megatron_vlm_lora_targets,
)
from skyrl.train.config import SFTConfig, SkyRLTrainConfig
from skyrl.train.sft_trainer import (
    _check_modality_homogeneity,
    _normalize_chat_messages,
    _resolve_num_training_steps,
)

# ---------------------------------------------------------------------------
# #13: a VLM critic is rejected at config time
# ---------------------------------------------------------------------------


def test_vlm_generator_rejects_critic():
    with pytest.raises(ValueError, match="critic"):
        SkyRLTrainConfig.from_cli_overrides(
            [
                "generator.vision_language_generator=true",
                "trainer.critic.model.path=Qwen/Qwen3-VL-2B-Instruct",
            ]
        )


def test_vlm_generator_without_critic_is_accepted():
    cfg = SkyRLTrainConfig.from_cli_overrides(["generator.vision_language_generator=true"])
    assert cfg.generator.vision_language_generator


# ---------------------------------------------------------------------------
# #4: mixed image / text-only batches carry empty tensors for text rows
# ---------------------------------------------------------------------------


def test_concat_nonempty_skips_empty_rows():
    a = torch.ones(4, 8)
    empty = torch.zeros(0, 8)
    b = torch.full((2, 8), 2.0)
    out = concat_nonempty_tensors(TensorList([a, empty, b]))
    assert out.shape == (6, 8)
    assert torch.equal(out[:4], a) and torch.equal(out[4:], b)


def test_concat_nonempty_returns_none_when_no_row_has_images():
    assert concat_nonempty_tensors(TensorList([torch.zeros(0, 8), torch.zeros(0, 8)])) is None
    assert concat_nonempty_tensors(None) is None


# ---------------------------------------------------------------------------
# #33: OpenAI image_url parts become processor-style image parts
# ---------------------------------------------------------------------------


def test_normalize_image_url_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image_url", "image_url": "https://example.com/b.png"},
                {"type": "text", "text": "What is this?"},
            ],
        },
        {"role": "assistant", "content": "A cat."},
    ]
    out = _normalize_chat_messages(messages)
    parts = out[0]["content"]
    assert parts[0] == {"type": "image", "image": "data:image/png;base64,AAAA"}
    assert parts[1] == {"type": "image", "image": "https://example.com/b.png"}
    assert parts[2] == {"type": "text", "text": "What is this?"}
    assert out[1]["content"] == "A cat."


def test_normalize_keeps_native_image_parts_and_strings():
    messages = [
        {"role": "user", "content": [{"type": "image", "image": "x"}, {"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": "yo"},
    ]
    assert _normalize_chat_messages(messages)[0]["content"] == messages[0]["content"]


def test_normalize_rejects_video_parts():
    messages = [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "v.mp4"}}]}]
    with pytest.raises(NotImplementedError, match="Video"):
        _normalize_chat_messages(messages)


# ---------------------------------------------------------------------------
# #34: mixed text/image training data fails at load, not mid-epoch
# ---------------------------------------------------------------------------


def _row(with_image: bool):
    row = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "num_actions": 1, "loss_mask": [1]}
    if with_image:
        row["pixel_values"] = [[0.0]]
        row["image_grid_thw"] = [[1, 1, 1]]
    return row


def test_modality_homogeneity_accepts_uniform_sources():
    _check_modality_homogeneity([[_row(True), _row(True)]], ["a"])
    _check_modality_homogeneity([[_row(False)], [_row(False)]], ["a", "b"])


def test_modality_homogeneity_rejects_mixed_rows():
    with pytest.raises(ValueError, match="mixes 1 image rows with 1 text-only rows"):
        _check_modality_homogeneity([[_row(True), _row(False)]], ["a"])


def test_modality_homogeneity_rejects_mixed_across_sources():
    with pytest.raises(ValueError, match="Mixed text\\+image"):
        _check_modality_homogeneity([[_row(True)], [_row(False)]], ["img", "txt"])


def test_modality_homogeneity_uses_modality_counts_when_available():
    class FakeStore:
        def modality_counts(self):
            return 3, 5

    with pytest.raises(ValueError, match="mixes 3 image rows with 2 text-only rows"):
        _check_modality_homogeneity([FakeStore()], ["store"])


# ---------------------------------------------------------------------------
# #52: Megatron SFT with num_epochs needs an explicit step count, not a silent default
# ---------------------------------------------------------------------------


def _sft_cfg(**overrides):
    return SFTConfig.from_cli_overrides({"model.path": "test/my-model", **overrides})


def test_megatron_epoch_based_run_requires_a_step_count():
    with pytest.raises(ValueError, match="max_training_steps"):
        _resolve_num_training_steps(_sft_cfg(strategy="megatron", num_epochs=2))


def test_megatron_epoch_based_run_accepts_max_training_steps():
    assert _resolve_num_training_steps(_sft_cfg(strategy="megatron", num_epochs=2, max_training_steps=100)) == 100


def test_megatron_num_steps_is_capped_by_max_training_steps():
    assert _resolve_num_training_steps(_sft_cfg(strategy="megatron", num_steps=50, max_training_steps=20)) == 20


def test_fsdp_epoch_based_run_resolves_to_none():
    assert _resolve_num_training_steps(_sft_cfg(strategy="fsdp", num_epochs=2)) is None


# ---------------------------------------------------------------------------
# Megatron LoRA targets on a VLM never match the vision tower
# ---------------------------------------------------------------------------


def _scope(targets, *, is_vlm=True, from_all_linear=False, exclude=None):
    return scope_megatron_vlm_lora_targets(
        targets, is_vlm=is_vlm, from_all_linear=from_all_linear, exclude_modules=exclude
    )


def test_megatron_all_linear_is_scoped_to_the_language_model():
    assert _scope(["linear_qkv", "linear_fc1"], from_all_linear=True) == [
        "*language_model*linear_qkv",
        "*language_model*linear_fc1",
    ]


@pytest.mark.parametrize("targets", [["linear_qkv", "linear_fc1"], ["*.linear_qkv"], ["*decoder*linear_proj"]])
def test_megatron_explicit_unscoped_targets_are_rejected_for_vlm(targets):
    with pytest.raises(ValueError, match="vision tower"):
        _scope(targets)


def test_megatron_explicit_scoped_targets_are_kept():
    targets = ["*language_model*linear_qkv", "*language_model*.layers.0.*linear_fc1"]
    assert _scope(targets) == targets


def test_megatron_exclude_modules_is_rejected_for_vlm():
    with pytest.raises(ValueError, match="exclude_modules"):
        _scope(["linear_qkv"], from_all_linear=True, exclude=["linear_fc1"])


def test_megatron_text_model_targets_are_untouched():
    assert _scope(["linear_qkv"], is_vlm=False) == ["linear_qkv"]
    assert _scope(["linear_qkv"], is_vlm=False, exclude=["linear_fc1"]) == ["linear_qkv"]


def test_megatron_scoped_patterns_skip_bridge_vision_modules():
    import re

    from skyrl.backends.skyrl_train.workers.worker_utils import (
        MEGATRON_VLM_LANGUAGE_MODEL_PREFIX,  # noqa: F401
    )

    def wildcard_match(pattern, key):  # Megatron-Bridge peft.utils.wildcard_match
        return re.compile("^" + pattern.replace("*", "(.*)") + "$").match(key) is not None

    patterns = _scope(["linear_qkv", "linear_fc1"], from_all_linear=True)
    lm = "module.language_model.decoder.layers.3.self_attention.linear_qkv"
    vit = "module.vision_model.decoder.layers.3.self_attention.linear_qkv"
    moe = "module.language_model.decoder.layers.3.mlp.experts.linear_fc1"
    assert any(wildcard_match(p, lm) for p in patterns)
    assert any(wildcard_match(p, moe) for p in patterns)
    assert not any(wildcard_match(p, vit) for p in patterns)
