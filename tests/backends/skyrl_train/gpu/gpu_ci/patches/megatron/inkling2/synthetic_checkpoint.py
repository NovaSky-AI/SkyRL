"""A synthetic Inkling2 checkpoint in the release layout, for tests that cannot use the real weights.

``model.llm.*`` names, packed routed experts with gate / up rows interleaved, and one layer group
per expert storage format of the release: BF16, NVFP4 (``_scale`` / ``_amax`` /
``_outer_scale_denominator``) and MXFP8 (UE8M0 scales). Every expert is selected per token
(``num_experts_per_tok == n_routed_experts``): with fewer, BF16 rounding flips near-tied expert
choices and comparisons against the reference become noisy without anything being wrong.

``hidden`` overrides the hidden size (vLLM's fused loop kernels support only the release sizes).

``looped=True`` writes a looped checkpoint (``model_type: inkling2_loop``, ``loop_config``): 12 layers, core =
layers 3-8 (two attn-res blocks of 3) run 3 times, rank-8 controllers and the cross-pass message queries, scaled so the messages move the output.
"""

import json
import math
import os
from typing import Optional

import torch

L, H, HEADS, KV, HEAD_DIM, E, FFN, V, UNPADDED_V, D_REL, WINDOW = 7, 256, 4, 2, 64, 8, 128, 512, 500, 8, 16
LOCAL = [i for i in range(L) if i % 3 != 2]
EXPERT_FORMAT = {0: "bf16", 1: "nvfp4", 2: "nvfp4", 3: "nvfp4", 4: "mxfp8", 5: "mxfp8", 6: "mxfp8"}
LOOPED_L = 12
LOOP = dict(core_start=3, core_end=9, num_passes=3, gate_rank=8, is_varloop=False)
LOOPED_EXPERT_FORMAT = {i: ("bf16", "nvfp4", "mxfp8")[i % 3] for i in range(LOOPED_L)}


def shape(looped: bool):
    """``(num_layers, local_layer_ids, expert formats, loop_config)`` of the checkpoint."""
    if looped:
        return LOOPED_L, [i for i in range(LOOPED_L) if i % 3 != 2], LOOPED_EXPERT_FORMAT, LOOP
    return L, LOCAL, EXPERT_FORMAT, None


def _expert_tensors(name: str, out: int, inp: int, fmt: str, g: torch.Generator, scale: float = 1.0) -> dict:
    """``scale`` multiplies the magnitudes (MXFP8 rounds it to a power of two)."""
    if fmt == "bf16":
        return {name: (0.02 * scale * torch.randn(E, out, inp, generator=g)).bfloat16()}
    if fmt == "nvfp4":
        return {
            name: torch.randint(0, 256, (E, out, inp // 2), generator=g, dtype=torch.uint8),
            name + "_scale": (0.5 + 1.5 * torch.rand(E, out, inp // 16, generator=g)).to(torch.float8_e4m3fn),
            # Dequantized magnitudes ~ |code| * scale * amax / 2688 ~ 0.02.
            name + "_amax": scale * (20 + 15 * torch.rand(E, generator=g)),
            name + "_outer_scale_denominator": torch.tensor([2688.0]),
        }
    return {
        name: (4 * torch.randn(E, out, inp, generator=g)).to(torch.float8_e4m3fn),
        name
        + "_scale": torch.full((E, out, inp // 32), 119 + round(math.log2(scale)), dtype=torch.uint8),  # 2^-8 * scale
    }


def write_checkpoint(
    path: str, tokenizer: Optional[str] = None, looped: bool = False, hidden: Optional[int] = None
) -> None:
    """Write the checkpoint to ``path``; ``tokenizer`` (a Hugging Face name) adds its files, which
    SkyRL's workers load. The tiny vocabulary means token ids from it must not be fed to the model."""
    hid = H if hidden is None else hidden
    # Inputs of width ``hid`` scale like 1 / sqrt(hid), so a wider model is conditioned like the default.
    s = (H / hid) ** 0.5
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    num_layers, local_ids, expert_format, loop = shape(looped)

    def w(*shape, std=0.02, mean=0.0, dtype=torch.bfloat16):
        return (mean + std * torch.randn(*shape, generator=g)).to(dtype)

    p = "model.llm."
    t = {
        p + "embed.weight": w(V, hid, std=1.0),
        p + "embed_norm.weight": w(hid, std=0.2, mean=1.0),
        p + "norm.weight": w(hid, std=0.2, mean=1.0),
        p + "unembed.weight": w(V, hid, std=0.02 * s),
        p + "attn_res.output_query": w(hid, std=0.5 * s, dtype=torch.float32),
    }
    if loop:
        n_core = loop["core_end"] - loop["core_start"]
        # Wider models need sharper message queries for a swapped query to move the output measurably.
        msg_std = 0.5 if hid == H else 2.0
        for name in ("middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"):
            t[p + name] = w(2 * n_core, hid, std=msg_std * s, dtype=torch.float32)
    for i in range(num_layers):
        local, lp = i in local_ids, f"{p}layers.{i}."
        if loop and loop["core_start"] <= i < loop["core_end"]:
            r = loop["gate_rank"]
            for module in ("attn", "k", "v", "mlp"):
                c = f"{lp}controllers.{module}."
                # Unit-scale gates: tanh(gain), tanh(forget) ~ O(1).
                t[c + "down_weight"] = w(r, 2 * hid, std=(2 * hid) ** -0.5)
                t[c + "gain_weight"] = w(hid, r, std=r**-0.5)
                t[c + "forget_weight"] = w(hid, r, std=r**-0.5)
        if i > 0:
            t[f"{p}attn_res.attn_queries.{i - 1}"] = (
                w(hid, std=0.5 * s, dtype=torch.float32) if local else w(3, hid, std=0.5 * s, dtype=torch.float32)
            )
        t[f"{p}attn_res.mlp_queries.{i}"] = w(hid, std=0.5 * s, dtype=torch.float32)
        for norm in ("attn_norm",) if local else ("q_input_norm", "k_input_norm", "v_input_norm"):
            t[lp + norm + ".weight"] = w(hid, std=0.2, mean=1.0)
        t[lp + "mlp_norm.weight"] = w(hid, std=0.2, mean=1.0)
        a = lp + "attn."
        t[a + "wq_du.weight"] = w(HEADS * HEAD_DIM, hid, std=0.02 * s)
        t[a + "wk_dv.weight"] = w(KV * HEAD_DIM, hid, std=0.02 * s)
        t[a + "wv_dv.weight"] = w(KV * HEAD_DIM, hid, std=0.02 * s)
        t[a + "wr_du.weight"] = w(HEADS * D_REL, hid, std=0.02 * s)
        t[a + "gate_attn.weight"] = w(HEADS * HEAD_DIM, hid, std=0.02 * s)
        t[a + "attn_sink_proj.weight"] = w(HEADS, hid, std=0.02 * s)
        t[a + "wo_ud.weight"] = w(hid, HEADS * HEAD_DIM)
        t[a + "q_l2_clip.scale"] = w(HEADS, HEAD_DIM, std=0.3, mean=1.0)
        t[a + "rel_l2_clip.scale"] = w(HEADS, D_REL, std=0.3, mean=1.0)
        t[a + "rel_logits_proj.proj"] = w(D_REL, WINDOW, std=0.3)
        for conv in ("k_sconv", "v_sconv"):
            if local:
                t[a + conv + ".weight"] = w(KV * HEAD_DIM, 1, 4, std=0.3)
            else:
                t[a + conv + ".static_w"] = w(4, KV * HEAD_DIM, std=0.3)
                t[a + conv + ".weight_proj"] = w(KV * 4, hid, std=0.3 * s)
        t[lp + "attn_sconv.weight"] = w(hid, 1, 4, std=0.3)
        t[lp + "mlp_sconv.weight"] = w(hid, 1, 4, std=0.3)
        t[lp + "mlp.gate.weight"] = w(E, hid, std=0.02 * s)
        t[lp + "mlp.gate.bias"] = w(E, std=0.05, dtype=torch.float32)
        t.update(_expert_tensors(lp + "mlp.experts.w13_weight", 2 * FFN, hid, expert_format[i], g, s))
        t.update(_expert_tensors(lp + "mlp.experts.w2_weight", hid, FFN, expert_format[i], g))

    save_file(t, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {name: "model.safetensors" for name in t}}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    text_config = dict(
        vocab_size=V,
        unpadded_vocab_size=UNPADDED_V,
        hidden_size=hid,
        num_hidden_layers=num_layers,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV,
        swa_head_dim=HEAD_DIM,
        local_layer_ids=local_ids,
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
    if loop:
        text_config["loop_config"] = loop
    config = {
        "architectures": ["Inkling2LoopForConditionalGeneration" if loop else "Inkling2ForConditionalGeneration"],
        "model_type": "inkling2_loop" if loop else "inkling2_mm_model",
        "text_config": text_config,
        "vision_config": None,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)
    if tokenizer is not None:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(tokenizer).save_pretrained(path)
