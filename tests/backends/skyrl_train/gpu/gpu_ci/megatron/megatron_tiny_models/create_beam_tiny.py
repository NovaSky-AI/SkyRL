"""Build a tiny random Beam (``BeamForCausalLM``) checkpoint.

Shrinks ``reflection/Beam-1.0-501B-A23B``'s config to 4 layers and small widths while keeping every
layer kind: sliding-window RoPE attention (layers 0-2) and NoPE + SSMax full attention (layer 3),
a dense MLP (layer 0) and grouped-sigmoid ``noaux_tc`` MoE layers with an fp32 router and
``e_score_correction_bias`` (layers 1-3), sandwich norms, the per-head attention output-gate bias
and the final logit softcap. The release's tokenizer and chat template are copied unchanged.

transformers has no Beam model class at SkyRL's pin, so the safetensors are written directly in
the checkpoint layout (per-expert ``mlp.experts.<n>.*`` tensors, bf16 except the fp32 router).

Shapes are chosen for the 4 x L4 CI mesh: 2 KV heads (vLLM TP2 packs ``qkv_gate_proj``), 8 experts
(Megatron EP2), and ``num_experts_per_tok // topk_group == 2`` (Megatron-Bridge's Beam router
requirement).

Usage (standalone):
    uv run --isolated --extra megatron python \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/megatron_tiny_models/create_beam_tiny.py /tmp/beam-tiny-random
"""

import json
import os
import shutil
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

SOURCE_MODEL_ID = "reflection/Beam-1.0-501B-A23B"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
LAYER_TYPES = ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
MLP_LAYER_TYPES = ["dense", "sparse", "sparse", "sparse"]
TINY_CONFIG = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "moe_intermediate_size": 128,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 2,
    "topk_group": 1,
    # Shorter than the test sequences, so the sliding-window mask is exercised.
    "sliding_window": 64,
    "max_position_embeddings": 4096,
    "layer_types": LAYER_TYPES,
    "mlp_layer_types": MLP_LAYER_TYPES,
    # The release's cap (30) never binds on a model this small; with the wider lm_head init below the
    # top logits reach ~7, so a cap of 4 bends them and a path that drops the softcap fails parity.
    "final_logit_softcapping": 4.0,
}


def _beam_tiny_state_dict(config: dict, generator: torch.Generator) -> dict[str, torch.Tensor]:
    hidden, head_dim = config["hidden_size"], config["head_dim"]
    q_rows = config["num_attention_heads"] * head_dim
    kv_rows = config["num_key_value_heads"] * head_dim

    def normal(*shape, dtype=torch.bfloat16, std=0.02):
        return (torch.randn(*shape, generator=generator) * std).to(dtype)

    def norm(size):
        # Near-one RMSNorm weights, so a swapped norm changes the logprobs.
        return (1.0 + torch.randn(size, generator=generator) * 0.1).to(torch.bfloat16)

    tensors = {
        "model.embed_tokens.weight": normal(config["vocab_size"], hidden),
        "lm_head.weight": normal(config["vocab_size"], hidden, std=0.1),
        "model.norm.weight": norm(hidden),
    }
    for i, mlp_type in enumerate(config["mlp_layer_types"]):
        p = f"model.layers.{i}"
        tensors.update(
            {
                f"{p}.input_layernorm.weight": norm(hidden),
                f"{p}.post_attention_layernorm.weight": norm(hidden),
                f"{p}.pre_feedforward_layernorm.weight": norm(hidden),
                f"{p}.post_feedforward_layernorm.weight": norm(hidden),
                f"{p}.self_attn.q_proj.weight": normal(q_rows, hidden),
                f"{p}.self_attn.k_proj.weight": normal(kv_rows, hidden),
                f"{p}.self_attn.v_proj.weight": normal(kv_rows, hidden),
                f"{p}.self_attn.gate_proj.weight": normal(q_rows, hidden),
                f"{p}.self_attn.gate_bias": normal(config["num_attention_heads"], std=0.5),
                f"{p}.self_attn.o_proj.weight": normal(hidden, q_rows),
                f"{p}.self_attn.q_norm.weight": norm(head_dim),
                f"{p}.self_attn.k_norm.weight": norm(head_dim),
            }
        )
        if mlp_type == "dense":
            inter = config["intermediate_size"]
            tensors.update(
                {
                    f"{p}.mlp.gate_proj.weight": normal(inter, hidden),
                    f"{p}.mlp.up_proj.weight": normal(inter, hidden),
                    f"{p}.mlp.down_proj.weight": normal(hidden, inter),
                }
            )
            continue
        inter = config["moe_intermediate_size"]
        # The router and its correction bias are fp32 in the release; a std well above bf16
        # resolution keeps the top-k choice stable across the bf16 trainer and vLLM.
        tensors[f"{p}.mlp.gate.weight"] = normal(config["num_experts"], hidden, dtype=torch.float32, std=0.1)
        tensors[f"{p}.mlp.gate.e_score_correction_bias"] = normal(config["num_experts"], dtype=torch.float32, std=0.1)
        for e in range(config["num_experts"]):
            tensors.update(
                {
                    f"{p}.mlp.experts.{e}.gate_proj.weight": normal(inter, hidden),
                    f"{p}.mlp.experts.{e}.up_proj.weight": normal(inter, hidden),
                    f"{p}.mlp.experts.{e}.down_proj.weight": normal(hidden, inter),
                }
            )
    return tensors


def create_beam_tiny(save_folder: str, seed: int = 0) -> str:
    """Write the tiny checkpoint to ``save_folder`` and return it (no-op when it exists)."""
    if os.path.exists(os.path.join(save_folder, "model.safetensors")):
        return save_folder
    os.makedirs(save_folder, exist_ok=True)

    for filename in TOKENIZER_FILES:
        shutil.copy(hf_hub_download(SOURCE_MODEL_ID, filename=filename), os.path.join(save_folder, filename))
    with open(hf_hub_download(SOURCE_MODEL_ID, filename="config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(TINY_CONFIG)
    config["dtype"] = "bfloat16"
    with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    generator = torch.Generator().manual_seed(seed)
    tensors = _beam_tiny_state_dict(config, generator)
    tmp_path = os.path.join(save_folder, "model.safetensors.tmp")
    save_file(tensors, tmp_path, metadata={"format": "pt"})
    # Written last and atomically: the safetensors file marks a complete checkpoint.
    os.replace(tmp_path, os.path.join(save_folder, "model.safetensors"))
    return save_folder


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/beam-tiny-random"
    print(create_beam_tiny(out))
