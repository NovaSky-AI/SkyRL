"""Tests for the vLLM-free multimodal kwargs codec and its dispatch in ``decode_mm_kwargs``."""

import torch

from skyrl.backends.mm_codec import (
    PREFIX,
    decode_mm_kwargs_item,
    encode_mm_kwargs_item,
    is_native_blob,
)
from skyrl.backends.renderer import decode_mm_kwargs


def _sample_data():
    return {
        "pixel_values": torch.randn(4, 3, 8, 8, dtype=torch.bfloat16),
        "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.long),
    }


def test_codec_roundtrips_vision_tensors():
    data = _sample_data()
    blob = encode_mm_kwargs_item(data)
    assert is_native_blob(blob)
    assert blob.startswith(PREFIX)
    out = decode_mm_kwargs_item(blob)
    assert set(out) == {"pixel_values", "image_grid_thw"}
    assert torch.equal(out["pixel_values"], data["pixel_values"])
    assert torch.equal(out["image_grid_thw"], data["image_grid_thw"])


def test_is_native_blob_rejects_untagged():
    assert not is_native_blob("aGVsbG8=")
    assert not is_native_blob("")
    assert not is_native_blob(None)


def test_decode_mm_kwargs_accepts_native_blob_without_vllm():
    # No vLLM stub here: a native-engine blob must decode via the safetensors codec
    # alone, which is what a vLLM-free Trainium node relies on.
    data = _sample_data()
    blob = encode_mm_kwargs_item(data)
    result = decode_mm_kwargs({"image": [blob]})
    assert torch.equal(result["pixel_values"], data["pixel_values"])
    assert torch.equal(result["image_grid_thw"], data["image_grid_thw"])


def test_decode_mm_kwargs_empty():
    result = decode_mm_kwargs(None)
    assert result["pixel_values"] is None
    assert result["image_grid_thw"] is None
