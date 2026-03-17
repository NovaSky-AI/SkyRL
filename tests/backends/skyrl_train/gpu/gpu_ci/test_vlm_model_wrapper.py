"""
GPU tests for VLM (vision-language model) loading and forward pass via HFModelWrapper.
"""

import numpy as np
import pytest
import torch
from PIL import Image
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


def make_solid_color_image(color_rgb, size=28):
    """Return a solid-color PIL Image of the given (R, G, B) tuple."""
    arr = np.full((size, size, 3), color_rgb, dtype=np.uint8)
    return Image.fromarray(arr)


def build_vlm_inputs(processor, prompt_text, response_text, image=None, device="cuda"):
    """Build tokenized VLM inputs and compute num_actions dynamically.

    Returns a dict with keys:
        input_ids, attention_mask (on device)
        num_actions (int)
        pixel_values, image_grid_thw (TensorList, on device) — only when image is provided
    """
    # Build user content
    user_content = []
    if image is not None:
        user_content.append({"type": "image", "image": image, "resized_height": 28, "resized_width": 28})
    user_content.append({"type": "text", "text": prompt_text})

    # Full messages (prompt + response)
    full_messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]},
    ]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    # Prompt-only messages (to compute num_actions)
    prompt_messages = [{"role": "user", "content": user_content}]
    prompt_text_only = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    images_list = [image] if image is not None else None
    full_inputs = processor(text=[full_text], images=images_list, return_tensors="pt", padding=True)
    prompt_inputs = processor(text=[prompt_text_only], images=images_list, return_tensors="pt", padding=True)

    num_actions = full_inputs["input_ids"].shape[1] - prompt_inputs["input_ids"].shape[1]

    result = {
        "input_ids": full_inputs["input_ids"].to(device),
        "attention_mask": full_inputs["attention_mask"].to(device),
        "num_actions": num_actions,
    }

    if image is not None:
        result["pixel_values"] = TensorList([full_inputs["pixel_values"].to(device)])
        result["image_grid_thw"] = TensorList([full_inputs["image_grid_thw"].to(device)])

    return result


def test_vlm_model_loading(vlm_model):
    """VLM model should load with is_vlm=True and correct model class."""
    assert vlm_model.is_vlm is True
    # Qwen3-VL uses Qwen2_5_VLForConditionalGeneration or similar
    class_name = type(vlm_model.model).__name__
    assert "ConditionalGeneration" in class_name or "VL" in class_name, f"Expected a VL model class, got {class_name}"


def test_vlm_forward_with_vision_data(vlm_model, processor):
    """Forward pass with dummy vision data should produce correct output shapes."""
    image = make_solid_color_image((128, 64, 200))
    inputs = build_vlm_inputs(processor, "Describe this image.", "A test response here", image=image)

    batch_size = 1
    num_actions = inputs["num_actions"]

    with torch.no_grad():
        action_log_probs = vlm_model(
            inputs["input_ids"],
            num_actions,
            inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )

    assert action_log_probs.shape == (
        batch_size,
        num_actions,
    ), f"Expected shape ({batch_size}, {num_actions}), got {action_log_probs.shape}"
    assert torch.isfinite(action_log_probs).all(), "action_log_probs contains non-finite values"
    assert (action_log_probs <= 0).all(), "Log probabilities should be <= 0"


def test_vlm_forward_text_only(vlm_model, processor):
    """Forward pass without vision kwargs should still produce valid output (text-only fallback)."""
    inputs = build_vlm_inputs(processor, "Hello, world!", "Hi there!")

    batch_size = 1
    num_actions = inputs["num_actions"]

    with torch.no_grad():
        action_log_probs = vlm_model(
            inputs["input_ids"],
            num_actions,
            inputs["attention_mask"],
        )

    assert action_log_probs.shape == (batch_size, num_actions)
    assert torch.isfinite(action_log_probs).all()
    assert (action_log_probs <= 0).all()


def test_vlm_vision_affects_output(vlm_model, processor):
    """Providing vision data should change the log probs compared to text-only."""
    image = make_solid_color_image((255, 0, 0))
    inputs = build_vlm_inputs(processor, "What color is this image?", "The image is red.", image=image)
    num_actions = inputs["num_actions"]

    with torch.no_grad():
        log_probs_with = vlm_model(
            inputs["input_ids"],
            num_actions,
            inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )

        log_probs_without = vlm_model(
            inputs["input_ids"],
            num_actions,
            inputs["attention_mask"],
        )

    assert not torch.allclose(
        log_probs_with, log_probs_without, atol=1e-3
    ), "Log probs with and without vision data should differ"


def test_vlm_log_probs_match_manual(vlm_model, processor):
    """Wrapper log probs should match a manual log_softmax + gather computation."""
    image = make_solid_color_image((0, 255, 0))
    inputs = build_vlm_inputs(processor, "Describe this image.", "A green square.", image=image)
    num_actions = inputs["num_actions"]

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pv = inputs["pixel_values"]
    igt = inputs["image_grid_thw"]

    # Wrapper path
    with torch.no_grad():
        wrapper_log_probs = vlm_model(
            input_ids,
            num_actions,
            attention_mask,
            pixel_values=pv,
            image_grid_thw=igt,
        )

    # Manual path: run the raw model
    pv_cat = torch.cat(pv.tensors, dim=0)
    igt_cat = torch.cat(igt.tensors, dim=0)

    with torch.no_grad():
        output = vlm_model.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            pixel_values=pv_cat,
            image_grid_thw=igt_cat,
        )

    logits = output["logits"].float()
    log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
    shifted_labels = torch.roll(input_ids, -1, dims=1)
    gathered = log_probs_full.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    manual_log_probs = gathered[:, -num_actions - 1 : -1]

    torch.testing.assert_close(wrapper_log_probs.float(), manual_log_probs, atol=5e-2, rtol=1e-2)


def test_vlm_different_images_diverge(vlm_model, processor):
    """Different images with the same text should produce different log probs."""
    red_image = make_solid_color_image((255, 0, 0))
    blue_image = make_solid_color_image((0, 0, 255))

    prompt = "What color is this image?"
    response = "It is a solid color."

    inputs_red = build_vlm_inputs(processor, prompt, response, image=red_image)
    inputs_blue = build_vlm_inputs(processor, prompt, response, image=blue_image)

    num_actions = inputs_red["num_actions"]
    assert num_actions == inputs_blue["num_actions"], "Tokenization should match for identical text"

    with torch.no_grad():
        lp_red = vlm_model(
            inputs_red["input_ids"],
            num_actions,
            inputs_red["attention_mask"],
            pixel_values=inputs_red["pixel_values"],
            image_grid_thw=inputs_red["image_grid_thw"],
        )
        lp_blue = vlm_model(
            inputs_blue["input_ids"],
            num_actions,
            inputs_blue["attention_mask"],
            pixel_values=inputs_blue["pixel_values"],
            image_grid_thw=inputs_blue["image_grid_thw"],
        )

    assert not torch.allclose(lp_red, lp_blue, atol=1e-3), "Red and blue images should produce different log probs"


def test_vlm_semantic_color_recognition(vlm_model, processor):
    """Model should assign highest log P(response | prompt, image) to the correct color name."""
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }
    responses = {
        "red": "The color is red",
        "green": "The color is green",
        "blue": "The color is blue",
    }
    prompt = "What color do you see in this image?"

    for true_color, rgb in colors.items():
        image = make_solid_color_image(rgb)
        log_p = {}

        for resp_color, resp_text in responses.items():
            inputs = build_vlm_inputs(processor, prompt, resp_text, image=image)
            num_actions = inputs["num_actions"]

            with torch.no_grad():
                action_lp = vlm_model(
                    inputs["input_ids"],
                    num_actions,
                    inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"],
                )

            log_p[resp_color] = action_lp.sum().item()

        # The correct color should have the highest log probability
        best = max(log_p, key=log_p.get)
        assert best == true_color, (
            f"For {true_color} image, expected highest log P for '{true_color}' " f"but got '{best}'. Scores: {log_p}"
        )


def test_vlm_forward_batched_vision(vlm_model, processor):
    """Batched forward with different images should match per-sample results."""
    images = [
        make_solid_color_image((255, 0, 0)),  # red
        make_solid_color_image((0, 0, 255)),  # blue
    ]
    prompt = "Describe this image."
    response = "A solid color square."

    # 1. Run each sample individually
    per_sample_lps = []
    per_sample_inputs = []
    for img in images:
        inp = build_vlm_inputs(processor, prompt, response, image=img)
        per_sample_inputs.append(inp)
        with torch.no_grad():
            lp = vlm_model(
                inp["input_ids"],
                inp["num_actions"],
                inp["attention_mask"],
                pixel_values=inp["pixel_values"],
                image_grid_thw=inp["image_grid_thw"],
            )
        per_sample_lps.append(lp)

    # 2. Build batched input
    num_actions = per_sample_inputs[0]["num_actions"]
    # Same text → same seq length → simple cat along batch dim
    input_ids = torch.cat([inp["input_ids"] for inp in per_sample_inputs], dim=0)
    attention_mask = torch.cat([inp["attention_mask"] for inp in per_sample_inputs], dim=0)
    # TensorList with one tensor per sample
    pixel_values = TensorList([inp["pixel_values"].tensors[0] for inp in per_sample_inputs])
    image_grid_thw = TensorList([inp["image_grid_thw"].tensors[0] for inp in per_sample_inputs])

    # 3. Run batched forward
    with torch.no_grad():
        batched_lps = vlm_model(
            input_ids,
            num_actions,
            attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    # 4. Validate
    batch_size = len(images)
    assert batched_lps.shape == (batch_size, num_actions)
    for i, single_lp in enumerate(per_sample_lps):
        torch.testing.assert_close(batched_lps[i : i + 1], single_lp, atol=5e-2, rtol=1e-2)
