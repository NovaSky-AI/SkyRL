#!/usr/bin/env python3
"""Numerical parity checker for HF Qwen3-VL vs tx JAX Qwen3-VL.

Compares prefill and one-step decode logits/hidden states.

Examples:
  python3 scripts/compare_qwen3_vl_hf_jax.py --model-id Qwen/Qwen3-VL-4B-Instruct --prompt "Describe this image." --image /path/img.jpg
  python3 scripts/compare_qwen3_vl_hf_jax.py --model-id Qwen/Qwen3-VL-4B-Instruct --prompt "Hello"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from tx.models.configs import Qwen3VLModelConfig
from tx.models.qwen3_vl import Qwen3VLForCausalLM
from tx.utils.models import load_safetensors, resolve_model_path


@dataclass
class PreparedInputs:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray | None = None
    image_grid_thw: np.ndarray | None = None
    pixel_values_videos: np.ndarray | None = None
    video_grid_thw: np.ndarray | None = None


def _to_numpy(t: torch.Tensor | None) -> np.ndarray | None:
    if t is None:
        return None
    return t.detach().cpu().numpy()


def _build_single_example_inputs(
    model_id: str,
    prompt: str,
    image: str | None,
    video: str | None,
) -> PreparedInputs:
    if image is None and video is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        encoded = tokenizer([prompt], return_tensors="pt")
        return PreparedInputs(
            input_ids=_to_numpy(encoded["input_ids"]).astype(np.int32),
            attention_mask=_to_numpy(encoded["attention_mask"]).astype(np.int32),
        )

    processor = AutoProcessor.from_pretrained(model_id)
    content: list[dict[str, Any]] = []
    images: list[str] = []
    videos: list[str] = []

    if image is not None:
        content.append({"type": "image", "image": image})
        images.append(image)
    if video is not None:
        content.append({"type": "video", "video": video})
        videos.append(video)
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    encoded = processor(**kwargs)

    return PreparedInputs(
        input_ids=_to_numpy(encoded["input_ids"]).astype(np.int32),
        attention_mask=_to_numpy(encoded["attention_mask"]).astype(np.int32),
        pixel_values=_to_numpy(encoded.get("pixel_values")),
        image_grid_thw=_to_numpy(encoded.get("image_grid_thw")),
        pixel_values_videos=_to_numpy(encoded.get("pixel_values_videos")),
        video_grid_thw=_to_numpy(encoded.get("video_grid_thw")),
    )


def _make_jax_model(model_id: str) -> Qwen3VLForCausalLM:
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(model_id)
    config = Qwen3VLModelConfig(
        base_config,
        max_lora_adapters=0,
        max_lora_rank=0,
        shard_attention_heads=True,
        gradient_checkpointing=False,
    )
    mesh = jax.make_mesh(
        (1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2
    )
    with jax.set_mesh(mesh):
        model = Qwen3VLForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))

    weights_dir = resolve_model_path(model_id)
    load_safetensors(weights_dir, config, model)
    return model


def _compare(name: str, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"{name}: max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}")
    return max_abs, mean_abs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF and JAX Qwen3-VL numerically."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--video", default=None)
    parser.add_argument("--decode-token-id", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    if args.image is not None and not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if args.video is not None and not Path(args.video).exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    prepared = _build_single_example_inputs(
        args.model_id, args.prompt, args.image, args.video
    )

    print("Loading HF model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, attn_implementation="eager", use_safetensors=True
    )
    hf_model.eval()

    print("Loading JAX model...")
    jax_model = _make_jax_model(args.model_id)

    hf_kwargs: dict[str, Any] = {
        "input_ids": torch.tensor(prepared.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(prepared.attention_mask, dtype=torch.long),
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if prepared.pixel_values is not None:
        hf_kwargs["pixel_values"] = torch.tensor(prepared.pixel_values)
    if prepared.image_grid_thw is not None:
        hf_kwargs["image_grid_thw"] = torch.tensor(
            prepared.image_grid_thw, dtype=torch.long
        )
    if prepared.pixel_values_videos is not None:
        hf_kwargs["pixel_values_videos"] = torch.tensor(prepared.pixel_values_videos)
    if prepared.video_grid_thw is not None:
        hf_kwargs["video_grid_thw"] = torch.tensor(
            prepared.video_grid_thw, dtype=torch.long
        )

    with torch.no_grad():
        hf_prefill = hf_model(**hf_kwargs)
    hf_prefill_logits = hf_prefill.logits.detach().cpu().numpy()

    jax_prefill = jax_model(
        jnp.asarray(prepared.input_ids, dtype=jnp.int32),
        attention_mask=jnp.asarray(prepared.attention_mask, dtype=jnp.int32),
        pixel_values=jnp.asarray(prepared.pixel_values)
        if prepared.pixel_values is not None
        else None,
        image_grid_thw=(
            jnp.asarray(prepared.image_grid_thw, dtype=jnp.int32)
            if prepared.image_grid_thw is not None
            else None
        ),
        pixel_values_videos=(
            jnp.asarray(prepared.pixel_values_videos)
            if prepared.pixel_values_videos is not None
            else None
        ),
        video_grid_thw=(
            jnp.asarray(prepared.video_grid_thw, dtype=jnp.int32)
            if prepared.video_grid_thw is not None
            else None
        ),
        output_hidden_states=True,
    )
    jax_prefill_hidden = np.asarray(jax_prefill.last_hidden_state)
    jax_prefill_logits = np.asarray(
        jax_model.compute_logits(jax_prefill.last_hidden_state)
    )

    print("== Prefill ==")
    _compare(
        "prefill_hidden",
        jax_prefill_hidden,
        hf_prefill.hidden_states[-1].detach().cpu().numpy(),
    )
    prefill_max, _ = _compare("prefill_logits", jax_prefill_logits, hf_prefill_logits)

    next_token_id = args.decode_token_id
    if next_token_id is None:
        next_token_id = int(prepared.input_ids[0, -1])
    next_token = np.array([[next_token_id]], dtype=np.int32)

    hf_decode_kwargs = {
        "input_ids": torch.tensor(next_token, dtype=torch.long),
        "attention_mask": torch.tensor(
            np.concatenate(
                [prepared.attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
            )
        ),
        "past_key_values": hf_prefill.past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    with torch.no_grad():
        hf_decode = hf_model(**hf_decode_kwargs)
    hf_decode_logits = hf_decode.logits.detach().cpu().numpy()

    decode_positions = jnp.asarray(
        prepared.attention_mask.sum(axis=1, keepdims=True), dtype=jnp.int32
    )
    decode_attention_mask = jnp.asarray(
        np.concatenate(
            [prepared.attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
        ),
        dtype=jnp.int32,
    )
    jax_decode = jax_model(
        jnp.asarray(next_token, dtype=jnp.int32),
        attention_mask=decode_attention_mask,
        positions=decode_positions,
        kv_cache=jax_prefill.kv_cache,
    )
    jax_decode_logits = np.asarray(
        jax_model.compute_logits(jax_decode.last_hidden_state)
    )

    print("== Decode (1 step) ==")
    decode_max, _ = _compare("decode_logits", jax_decode_logits, hf_decode_logits)

    passed = np.allclose(
        jax_prefill_logits, hf_prefill_logits, rtol=args.rtol, atol=args.atol
    ) and np.allclose(
        jax_decode_logits, hf_decode_logits, rtol=args.rtol, atol=args.atol
    )
    print(f"PASS={passed} (rtol={args.rtol}, atol={args.atol})")
    if not passed:
        raise SystemExit(
            f"Parity check failed: prefill_max={prefill_max:.6e}, decode_max={decode_max:.6e}"
        )


if __name__ == "__main__":
    main()
