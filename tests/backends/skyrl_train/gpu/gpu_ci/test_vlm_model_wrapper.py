"""
GPU tests for VLM (vision-language model) loading and forward pass via HFModelWrapper.

Requires a GPU and the Qwen3-VL-2B-Instruct model weights.

source .venv/bin/activate && python -m pytest tests/backends/skyrl_train/gpu/gpu_ci/test_vlm_model_wrapper.py -v
"""

import pytest
import torch
from transformers import AutoProcessor

from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def vlm_model():
    """Load the VLM model once for all tests in this module."""
    model = HFModelWrapper(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=True,
        bf16=True,
        sequence_parallel_size=1,
        use_sample_packing=False,
    )
    model.model.eval()
    model.model.to("cuda")
    return model


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)


def test_vlm_model_loading(vlm_model):
    """VLM model should load with is_vlm=True and correct model class."""
    assert vlm_model.is_vlm is True
    # Qwen3-VL uses Qwen2_5_VLForConditionalGeneration or similar
    class_name = type(vlm_model.model).__name__
    assert "ConditionalGeneration" in class_name or "VL" in class_name, f"Expected a VL model class, got {class_name}"


def test_vlm_forward_with_vision_data(vlm_model, processor):
    """Forward pass with dummy vision data should produce correct output shapes."""
    # Build a simple prompt with an image placeholder
    batch_size = 1
    num_actions = 4

    # Create a dummy image (3x28x28 — small for testing)
    import numpy as np
    from PIL import Image

    dummy_image = Image.fromarray(np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image, "resized_height": 28, "resized_width": 28},
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "A test response here"}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].to("cuda")  # (1, seq_len)
    attention_mask = inputs["attention_mask"].to("cuda")  # (1, seq_len)
    pixel_values_raw = inputs["pixel_values"].to("cuda")  # (num_patches, dim)
    image_grid_thw_raw = inputs["image_grid_thw"].to("cuda")  # (num_images, 3)

    # Wrap in TensorList (one entry per batch element)
    pixel_values = TensorList([pixel_values_raw])
    image_grid_thw = TensorList([image_grid_thw_raw])

    with torch.no_grad():
        action_log_probs = vlm_model(
            input_ids,
            num_actions,
            attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    assert action_log_probs.shape == (
        batch_size,
        num_actions,
    ), f"Expected shape ({batch_size}, {num_actions}), got {action_log_probs.shape}"
    # Log probs should be finite and negative
    assert torch.isfinite(action_log_probs).all(), "action_log_probs contains non-finite values"
    assert (action_log_probs <= 0).all(), "Log probabilities should be <= 0"


def test_vlm_forward_text_only(vlm_model, processor):
    """Forward pass without vision kwargs should still produce valid output (text-only fallback)."""
    batch_size = 1
    num_actions = 3

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], return_tensors="pt", padding=True)

    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        action_log_probs = vlm_model(
            input_ids,
            num_actions,
            attention_mask,
        )

    assert action_log_probs.shape == (batch_size, num_actions)
    assert torch.isfinite(action_log_probs).all()
    assert (action_log_probs <= 0).all()
