"""Build an N-layer BF16 CI slice of moonshotai/Kimi-K3.

    uv run --isolated --with torch --with safetensors python examples/train/kimi_k3/make_slice.py \
        <src_dir> <dst_dir> [--num-layers 4] [--num-experts 128] [--attn-res-block-size 2]

``src_dir`` only needs the configs / remote code / tokenizer plus the shards holding the kept
layers, the embedding, LM head, final norms and the vision tower (shards 1-4 and 94-96 for the
default 4 layers).

Kimi K3's ``linear_attn_config`` layer lists are 1-based (``is_kda_layer(i)`` checks ``i + 1``),
so the first four layers are KDA (dense MLP), KDA (MoE), KDA (MoE), NoPE-MLA (MoE): 4 layers cover
every layer type. The slice:

* keeps the first ``--num-experts`` routed experts per MoE layer (router ``gate.weight`` and
  ``e_score_correction_bias`` rows truncated to match, top-k unchanged). All 896 experts in bf16
  are ~59 GB per MoE layer, which does not fit next to a colocated vLLM on 4xH100;
* dequantizes the MXFP4 routed experts to bf16 (exact: an E2M1 value times a power-of-two E8M0
  scale is representable in bf16) and drops ``quantization_config``, so vLLM and the Megatron
  bridge both run plain bf16 and weight sync needs no requantization;
* sets ``attn_res_block_size`` (12 in the release) to ``--attn-res-block-size`` so the attention
  residual bank spans more than one block within 4 layers. HF, vLLM and the bridge all read it from
  the config; ``0`` keeps the released value.

All other tensors (attention, shared experts, dense MLP, norms, embedding, LM head, vision tower)
are copied unchanged. MTP is not present in the release (``num_nextn_predict_layers=0``).

Published as ``eatang/Kimi-K3-4layer``; republish with ``hf upload eatang/Kimi-K3-4layer <dst_dir>``.
A local override must keep "kimi-k3" in its path: the GPU tests key the model's settings on it.
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "language_model.model."
LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"(.*\.block_sparse_moe\.experts\.)(\d+)\.(w[123])\.weight_(packed|scale)$")
ROUTER_KEYS = (".block_sparse_moe.gate.weight", ".block_sparse_moe.gate.e_score_correction_bias")
# E2M1 code points, low nibble first (same table as megatron.bridge quantization_utils).
FP4_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    w = packed.view(torch.uint8)
    table = torch.tensor(FP4_E2M1, dtype=torch.float32)
    vals = torch.stack([table[(w & 0xF).long()], table[(w >> 4).long()]], dim=-1).reshape(w.shape[0], -1)
    s = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
    out = vals * s.repeat_interleave(vals.shape[1] // s.shape[1], dim=1)
    bf = out.to(torch.bfloat16)
    assert torch.equal(bf.float(), out), "MXFP4 -> bf16 dequant is not exact"
    return bf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--attn-res-block-size", type=int, default=2, help="0 keeps the released value")
    args = p.parse_args()
    src, dst, n = args.src, args.dst, args.num_layers
    os.makedirs(dst, exist_ok=True)

    cfg = json.load(open(os.path.join(src, "config.json")))
    tc = cfg["text_config"]
    num_experts = min(args.num_experts, tc["num_experts"])
    assert num_experts >= tc["num_experts_per_token"]

    # Group the tensors to keep by source shard.
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for k, f in weight_map.items():
        m = LAYER_RE.match(k)
        if m and int(m.group(1)) >= n:
            continue
        e = EXPERT_RE.match(k)
        if e and int(e.group(2)) >= num_experts:
            continue
        by_shard.setdefault(f, []).append(k)

    # Output: one shard per decoder layer, plus one for everything else.
    out: dict[str, dict[str, torch.Tensor]] = {}

    def put(name, t):
        m = LAYER_RE.match(name)
        out.setdefault(f"layer{int(m.group(1))}" if m else "rest", {})[name] = t.contiguous()

    pending: dict[str, dict[str, torch.Tensor]] = {}
    for f in sorted(by_shard):
        print(f"reading {f} ({len(by_shard[f])} tensors)", flush=True)
        with safe_open(os.path.join(src, f), "pt") as sf:
            for k in by_shard[f]:
                t = sf.get_tensor(k)
                e = EXPERT_RE.match(k)
                if e:
                    base = f"{e.group(1)}{e.group(2)}.{e.group(3)}.weight"
                    slot = pending.setdefault(base, {})
                    slot[e.group(4)] = t
                    if len(slot) == 2:
                        put(base, dequant_mxfp4(slot["packed"], slot["scale"]))
                        del pending[base]
                elif k.endswith(ROUTER_KEYS):
                    put(k, t[:num_experts].clone())
                else:
                    put(k, t)
    assert not pending, f"unpaired MXFP4 tensors: {list(pending)[:4]}"

    names = [f"layer{i}" for i in range(n)] + ["rest"]
    total = len(names)
    index = {"metadata": {}, "weight_map": {}}
    size = 0
    for i, key in enumerate(names, 1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        tensors = out[key]
        save_file(tensors, os.path.join(dst, fname), metadata={"format": "pt"})
        for k, t in tensors.items():
            index["weight_map"][k] = fname
            size += t.numel() * t.element_size()
        print(f"wrote {fname}: {len(tensors)} tensors", flush=True)
    index["metadata"]["total_size"] = size
    json.dump(index, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    # Config: truncate layers / layer-type lists (1-based), experts, drop quantization.
    tc["num_hidden_layers"] = n
    lac = tc["linear_attn_config"]
    lac["kda_layers"] = [i for i in lac["kda_layers"] if i <= n]
    lac["full_attn_layers"] = [i for i in lac["full_attn_layers"] if i <= n]
    assert sorted(lac["kda_layers"] + lac["full_attn_layers"]) == list(range(1, n + 1))
    tc["num_experts"] = num_experts
    if args.attn_res_block_size:
        tc["attn_res_block_size"] = args.attn_res_block_size
    tc.pop("quantization_config", None)
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for f in os.listdir(src):
        if f.endswith((".py", ".model", ".jinja")) or (
            f.endswith(".json") and f not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    print(f"done: {n} layers, {num_experts} experts, {size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
