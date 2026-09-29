"""CPU tests for the VLM correctness fixes (VLM_GAPS.md #4, #13, #14, #33, #34).

Run: uv run --extra dev pytest tests/train/test_vlm_correctness_guards.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    concat_nonempty_tensors,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.sft_trainer import (
    _check_modality_homogeneity,
    _normalize_chat_messages,
)
from skyrl.utils.tok import VISION_TOWER_MODULE_REGEX, lora_exclude_modules_for_model

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
# #14: LoRA excludes the vision tower by default for VLMs
# ---------------------------------------------------------------------------


def test_lora_exclude_defaults_to_vision_tower_for_vlm():
    assert lora_exclude_modules_for_model(is_vlm=True, configured=None) == VISION_TOWER_MODULE_REGEX
    assert lora_exclude_modules_for_model(is_vlm=False, configured=None) is None
    assert lora_exclude_modules_for_model(is_vlm=True, configured="custom") == "custom"


@pytest.mark.parametrize(
    "name",
    [
        "model.visual.blocks.3.attn.qkv",
        "model.visual.merger.linear_fc1",
        "model.vision_tower.encoder.layers.0.mlp.fc1",
        "model.multi_modal_projector.linear_1",
    ],
)
def test_vision_tower_regex_matches_vision_modules(name):
    import re

    assert re.fullmatch(VISION_TOWER_MODULE_REGEX, name)


@pytest.mark.parametrize("name", ["model.language_model.layers.0.self_attn.q_proj", "lm_head"])
def test_vision_tower_regex_leaves_language_model_alone(name):
    import re

    assert re.fullmatch(VISION_TOWER_MODULE_REGEX, name) is None


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
