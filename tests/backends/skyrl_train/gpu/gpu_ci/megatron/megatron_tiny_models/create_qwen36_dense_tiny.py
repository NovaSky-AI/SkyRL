"""Build a tiny random dense Qwen3.6 (``qwen3_5`` hybrid GDN + gated attention) checkpoint.

Shrinks ``Qwen/Qwen3.6-27B``'s config to a few layers and small widths while keeping the
architecture that matters for LoRA import tests: GatedDeltaNet layers with separate
``in_proj_qkv/z/b/a`` linears, full-attention layers with the attention output gate
(``q_proj`` holds q and gate), MRoPE with ``partial_rotary_factor``, an untied ``lm_head``
and the VL ``model.language_model.*`` namespace.

Usage (standalone):
    uv run --isolated --extra megatron python \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/megatron_tiny_models/create_qwen36_dense_tiny.py /tmp/qwen36-dense-tiny-random
"""

import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

SOURCE_MODEL_ID = "Qwen/Qwen3.6-27B"
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def create_qwen36_dense_tiny(save_folder: str, seed: int = 0) -> str:
    """Write the tiny checkpoint to ``save_folder`` and return it (no-op when it exists)."""
    if os.path.exists(os.path.join(save_folder, "model.safetensors")):
        return save_folder
    os.makedirs(save_folder, exist_ok=True)

    processor = AutoProcessor.from_pretrained(SOURCE_MODEL_ID, trust_remote_code=True)
    processor.save_pretrained(save_folder)

    with open(hf_hub_download(SOURCE_MODEL_ID, filename="config.json", repo_type="model"), "r", encoding="utf-8") as f:
        config_json = json.load(f)
    text = config_json["text_config"]
    text.update(
        {
            "num_hidden_layers": len(LAYER_TYPES),
            "layer_types": LAYER_TYPES,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "mtp_num_hidden_layers": 0,
            "max_position_embeddings": 4096,
        }
    )
    # rotary dim = head_dim * partial_rotary_factor = 8 -> 4 frequency pairs split across t/h/w.
    text["rope_parameters"]["mrope_section"] = [2, 1, 1]
    config_json["vision_config"]["depth"] = 2
    with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    config = AutoConfig.from_pretrained(save_folder, trust_remote_code=True)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3_5ForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            torch.nn.init.normal_(p, 0, 0.1)
        for i, layer_type in enumerate(LAYER_TYPES):
            if layer_type == "linear_attention":
                attn = model.model.language_model.layers[i].linear_attn
                attn.A_log = torch.nn.Parameter(attn.A_log.float())
                attn.norm.float()
    model.generation_config.do_sample = True
    model.cpu().save_pretrained(save_folder)
    return save_folder


if __name__ == "__main__":
    print(create_qwen36_dense_tiny(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen36-dense-tiny-random"))
