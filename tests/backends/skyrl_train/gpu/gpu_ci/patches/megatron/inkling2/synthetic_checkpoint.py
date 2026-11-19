"""A synthetic Inkling2 checkpoint in the release layout, for tests that cannot use the real weights.

``model.llm.*`` names, packed routed experts with gate / up rows interleaved, and one layer group
per expert storage format of the release: BF16, NVFP4 (``_scale`` / ``_amax`` /
``_outer_scale_denominator``) and MXFP8 (UE8M0 scales). Every expert is selected per token
(``num_experts_per_tok == n_routed_experts``): with fewer, BF16 rounding flips near-tied expert
choices and comparisons against the reference become noisy without anything being wrong.
"""

import json
import os
from typing import Optional

import torch

L, H, HEADS, KV, HEAD_DIM, E, FFN, V, UNPADDED_V, D_REL, WINDOW = 7, 256, 4, 2, 64, 8, 128, 512, 500, 8, 16
LOCAL = [i for i in range(L) if i % 3 != 2]
EXPERT_FORMAT = {0: "bf16", 1: "nvfp4", 2: "nvfp4", 3: "nvfp4", 4: "mxfp8", 5: "mxfp8", 6: "mxfp8"}


def _expert_tensors(name: str, out: int, inp: int, fmt: str, g: torch.Generator) -> dict:
    if fmt == "bf16":
        return {name: (0.02 * torch.randn(E, out, inp, generator=g)).bfloat16()}
    if fmt == "nvfp4":
        return {
            name: torch.randint(0, 256, (E, out, inp // 2), generator=g, dtype=torch.uint8),
            name + "_scale": (0.5 + 1.5 * torch.rand(E, out, inp // 16, generator=g)).to(torch.float8_e4m3fn),
            # Dequantized magnitudes ~ |code| * scale * amax / 2688 ~ 0.02.
            name + "_amax": 20 + 15 * torch.rand(E, generator=g),
            name + "_outer_scale_denominator": torch.tensor([2688.0]),
        }
    return {
        name: (4 * torch.randn(E, out, inp, generator=g)).to(torch.float8_e4m3fn),
        name + "_scale": torch.full((E, out, inp // 32), 119, dtype=torch.uint8),  # 2^-8
    }


def write_checkpoint(path: str, tokenizer: Optional[str] = None) -> None:
    """Write the checkpoint to ``path``; ``tokenizer`` (a Hugging Face name) adds its files, which
    SkyRL's workers load. The tiny vocabulary means token ids from it must not be fed to the model."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)

    def w(*shape, std=0.02, mean=0.0, dtype=torch.bfloat16):
        return (mean + std * torch.randn(*shape, generator=g)).to(dtype)

    p = "model.llm."
    t = {
        p + "embed.weight": w(V, H, std=1.0),
        p + "embed_norm.weight": w(H, std=0.2, mean=1.0),
        p + "norm.weight": w(H, std=0.2, mean=1.0),
        p + "unembed.weight": w(V, H),
        p + "attn_res.output_query": w(H, std=0.5, dtype=torch.float32),
    }
    for i in range(L):
        local, lp = i in LOCAL, f"{p}layers.{i}."
        if i > 0:
            t[f"{p}attn_res.attn_queries.{i - 1}"] = (
                w(H, std=0.5, dtype=torch.float32) if local else w(3, H, std=0.5, dtype=torch.float32)
            )
        t[f"{p}attn_res.mlp_queries.{i}"] = w(H, std=0.5, dtype=torch.float32)
        for norm in ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm"):
            t[lp + norm + ".weight"] = w(H, std=0.2, mean=1.0)
        t[lp + "mlp_norm.weight"] = w(H, std=0.2, mean=1.0)
        a = lp + "attn."
        t[a + "wq_du.weight"] = w(HEADS * HEAD_DIM, H)
        t[a + "wk_dv.weight"] = w(KV * HEAD_DIM, H)
        t[a + "wv_dv.weight"] = w(KV * HEAD_DIM, H)
        t[a + "wr_du.weight"] = w(HEADS * D_REL, H)
        t[a + "gate_attn.weight"] = w(HEADS * HEAD_DIM, H)
        t[a + "attn_sink_proj.weight"] = w(HEADS, H)
        t[a + "wo_ud.weight"] = w(H, HEADS * HEAD_DIM)
        t[a + "q_l2_clip.scale"] = w(HEADS, HEAD_DIM, std=0.3, mean=1.0)
        t[a + "rel_l2_clip.scale"] = w(HEADS, D_REL, std=0.3, mean=1.0)
        t[a + "rel_logits_proj.proj"] = w(D_REL, WINDOW, std=0.3)
        for conv in ("k_sconv", "v_sconv"):
            if local:
                t[a + conv + ".weight"] = w(KV * HEAD_DIM, 1, 4, std=0.3)
            else:
                t[a + conv + ".static_w"] = w(4, KV * HEAD_DIM, std=0.3)
                t[a + conv + ".weight_proj"] = w(KV * 4, H, std=0.3)
        t[lp + "attn_sconv.weight"] = w(H, 1, 4, std=0.3)
        t[lp + "mlp_sconv.weight"] = w(H, 1, 4, std=0.3)
        t[lp + "mlp.gate.weight"] = w(E, H)
        t[lp + "mlp.gate.bias"] = w(E, std=0.05, dtype=torch.float32)
        t.update(_expert_tensors(lp + "mlp.experts.w13_weight", 2 * FFN, H, EXPERT_FORMAT[i], g))
        t.update(_expert_tensors(lp + "mlp.experts.w2_weight", H, FFN, EXPERT_FORMAT[i], g))

    save_file(t, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {name: "model.safetensors" for name in t}}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    text_config = dict(
        vocab_size=V,
        unpadded_vocab_size=UNPADDED_V,
        hidden_size=H,
        num_hidden_layers=L,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV,
        swa_head_dim=HEAD_DIM,
        local_layer_ids=LOCAL,
        sliding_window_size=WINDOW,
        rel_extent=WINDOW,
        d_rel=D_REL,
        log_scaling_n_floor=20,
        n_layers_per_attn_res_block=3,
        n_routed_experts=E,
        # Every expert selected: BF16 rounding otherwise flips near-tied expert choices.
        num_experts_per_tok=E,
        intermediate_size=FFN,
        model_max_length=4096,
    )
    config = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2_mm_model",
        "text_config": text_config,
        "vision_config": None,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    if tokenizer is not None:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer).save_pretrained(path)
