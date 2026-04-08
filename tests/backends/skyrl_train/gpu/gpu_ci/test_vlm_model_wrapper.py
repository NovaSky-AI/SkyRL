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

    if "mm_token_type_ids" in full_inputs:
        result["mm_token_type_ids"] = full_inputs["mm_token_type_ids"].to(device)

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


def test_vlm_log_probs_match_manual(vlm_model, processor):
    """Wrapper log probs should match a manual log_softmax + gather computation."""
    image = make_solid_color_image((0, 255, 0))
    inputs = build_vlm_inputs(processor, "Describe this image.", "A green square.", image=image)
    num_actions = inputs["num_actions"]

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pv = inputs["pixel_values"]
    igt = inputs["image_grid_thw"]

    mm_token_type_ids = inputs.get("mm_token_type_ids")

    # Wrapper path
    with torch.no_grad():
        wrapper_log_probs = vlm_model(
            input_ids,
            num_actions,
            attention_mask,
            pixel_values=pv,
            image_grid_thw=igt,
            mm_token_type_ids=mm_token_type_ids,
        )

    # Manual path: run the raw model
    pv_cat = torch.cat(pv.tensors, dim=0)
    igt_cat = torch.cat(igt.tensors, dim=0)

    manual_kwargs = dict(pixel_values=pv_cat, image_grid_thw=igt_cat)
    if mm_token_type_ids is not None:
        manual_kwargs["mm_token_type_ids"] = mm_token_type_ids

    with torch.no_grad():
        output = vlm_model.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            **manual_kwargs,
        )

    logits = output["logits"].float()
    log_probs_full = torch.nn.functional.log_softmax(logits, dim=-1)
    shifted_labels = torch.roll(input_ids, -1, dims=1)
    gathered = log_probs_full.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    manual_log_probs = gathered[:, -num_actions - 1 : -1]

    torch.testing.assert_close(wrapper_log_probs.float(), manual_log_probs, atol=5e-2, rtol=1e-2)


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
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                )

            log_p[resp_color] = action_lp.sum().item()

        # The correct color should have the highest log probability
        best = max(log_p, key=log_p.get)
        assert best == true_color, (
            f"For {true_color} image, expected highest log P for '{true_color}' " f"but got '{best}'. Scores: {log_p}"
        )


def _build_batched_vlm_inputs(processor, prompt, response, images, device="cuda"):
    """Build batched model inputs from a list of images with shared prompt/response text."""
    per_sample = [build_vlm_inputs(processor, prompt, response, image=img, device=device) for img in images]
    num_actions = per_sample[0]["num_actions"]
    result = {
        "input_ids": torch.cat([inp["input_ids"] for inp in per_sample], dim=0),
        "attention_mask": torch.cat([inp["attention_mask"] for inp in per_sample], dim=0),
        "num_actions": num_actions,
        "pixel_values": TensorList([inp["pixel_values"].tensors[0] for inp in per_sample]),
        "image_grid_thw": TensorList([inp["image_grid_thw"].tensors[0] for inp in per_sample]),
    }
    if "mm_token_type_ids" in per_sample[0]:
        result["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"] for inp in per_sample], dim=0)
    return result


def test_vlm_forward_batched_vision(vlm_model, processor):
    """Batched forward with permuted image order should produce correspondingly permuted outputs.

    Compares batched-vs-batched (not batched-vs-unbatched) to avoid numerical
    differences from the vision encoder's varlen flash attention when the number
    of packed sequences differs.
    """
    red = make_solid_color_image((255, 0, 0))
    blue = make_solid_color_image((0, 0, 255))
    prompt = "Describe this image."
    response = "A solid color square."

    # 1. Run batch in original order [red, blue]
    fwd = _build_batched_vlm_inputs(processor, prompt, response, [red, blue])
    num_actions = fwd["num_actions"]

    with torch.no_grad():
        lps_orig = vlm_model(
            fwd["input_ids"],
            num_actions,
            fwd["attention_mask"],
            pixel_values=fwd["pixel_values"],
            image_grid_thw=fwd["image_grid_thw"],
            mm_token_type_ids=fwd.get("mm_token_type_ids"),
        )

    # 2. Run batch in reversed order [blue, red]
    rev = _build_batched_vlm_inputs(processor, prompt, response, [blue, red])

    with torch.no_grad():
        lps_rev = vlm_model(
            rev["input_ids"],
            num_actions,
            rev["attention_mask"],
            pixel_values=rev["pixel_values"],
            image_grid_thw=rev["image_grid_thw"],
            mm_token_type_ids=rev.get("mm_token_type_ids"),
        )

    # 3. Basic shape / sanity checks
    assert lps_orig.shape == (2, num_actions)
    assert lps_rev.shape == (2, num_actions)
    assert torch.isfinite(lps_orig).all()
    assert (lps_orig <= 0).all()

    # 4. Different images should produce different log probs within each batch
    assert not torch.allclose(lps_orig[0], lps_orig[1], atol=1e-3)

    # 5. Permuted inputs → permuted outputs (validates TensorList piping)
    torch.testing.assert_close(lps_orig[0], lps_rev[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(lps_orig[1], lps_rev[0], atol=1e-4, rtol=1e-4)
