"""Build a BF16 N-layer slice of deepseek-ai/DeepSeek-V4-Flash for the Stage 0 roundtrip tests.

The default keeps layers 0, 2, 3 and 4, which covers every layer type in the model:

    layer 0: sliding-window attention only (compress ratio 0) + hash-routed MoE
    layer 2: CSA (compress ratio 4, with the lightning indexer) + hash-routed MoE
    layer 3: HCA (compress ratio 128) + score-routed MoE
    layer 4: CSA + score-routed MoE

They are renumbered 0..N-1; ``compress_ratios`` and ``num_hash_layers`` are rewritten to match (the
hash-routed layers are a contiguous prefix in the full model, so they stay one in the slice).
Embeddings, final norm, lm_head and the global hyper-connection head are kept; the MTP layer is
dropped.

Everything is dequantized to BF16 so that Megatron, vLLM and HF all run the same unquantized
weights, and ``quantization_config`` is removed:

- FP8 (e4m3) weights with 128x128 blocks of ue8m0 ``scale``: attention, shared experts, indexer
  ``wq_b``.
- MXFP4 experts: I8 ``weight`` with two e2m1 values per byte (low nibble first) and ue8m0
  ``scale`` per 32 values along K.

The release ships no Jinja chat template, so ``../chat_template.jinja`` is embedded in the slice's
``tokenizer_config.json``.

Reads the checkpoint from ``DSV4_LOCAL_DIR`` (default ~/models/DeepSeek-V4-Flash).

Usage: python make_slice.py <out_dir> [comma-separated layer ids, default 0,2,3,4]
"""

import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOCAL = os.environ.get("DSV4_LOCAL_DIR", os.path.expanduser("~/models/DeepSeek-V4-Flash"))
SHARD_BYTES = 4 << 30
FP8_BLOCK = 128
FP4_BLOCK = 32

FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def e8m0_exponent(scale):
    return scale.view(torch.uint8).to(torch.int32) - 127


def dequant_fp8_block(w, scale):
    rows, cols = w.shape
    exp = e8m0_exponent(scale).repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)[:rows, :cols]
    return torch.ldexp(w.float(), exp)


def dequant_mxfp4(packed, scale):
    # packed: [out, in/2] int8, scale: [out, in/32] e8m0.
    packed = packed.view(torch.uint8)
    lo = FP4_VALUES[(packed & 0x0F).long()]
    hi = FP4_VALUES[(packed >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1)
    exp = e8m0_exponent(scale)
    return torch.ldexp(vals.view(packed.shape[0], -1, FP4_BLOCK), exp.unsqueeze(-1)).reshape(vals.shape)


def main():
    out = sys.argv[1]
    layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 2, 3, 4]
    remap = {old: new for new, old in enumerate(layers)}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(LOCAL, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(LOCAL, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]

    def layer_of(key):
        m = re.match(r"layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    wanted = sorted(k for k in index if not k.startswith("mtp.") and (layer_of(k) is None or layer_of(k) in remap))

    def rename(key):
        m = re.match(r"layers\.(\d+)\.(.*)", key)
        return f"layers.{remap[int(m.group(1))]}.{m.group(2)}" if m else key

    handles = {}

    def fetch(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(LOCAL, shard), framework="pt")
        return handles[shard].get_tensor(key)

    by_layer = {}
    for key in wanted:
        by_layer.setdefault(layer_of(key) if layer_of(key) is not None else -1, []).append(key)

    shards, cur, cur_bytes = [], {}, 0

    def flush():
        nonlocal cur, cur_bytes
        fname = f"part-{len(shards) + 1:05d}.safetensors"
        save_file(cur, os.path.join(out, fname))
        shards.append({k: fname for k in cur})
        print(f"wrote {fname}", flush=True)
        cur, cur_bytes = {}, 0

    def emit(name, t):
        nonlocal cur_bytes
        cur[name] = t.contiguous()
        cur_bytes += t.numel() * t.element_size()
        if cur_bytes >= SHARD_BYTES:
            flush()

    for layer in sorted(by_layer):
        keys = set(by_layer[layer])
        for key in sorted(keys):
            if key.endswith(".scale") and key[: -len(".scale")] + ".weight" in keys:
                continue
            t = fetch(key)
            name = rename(key)
            scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
            if scale_key in keys and t.dtype == torch.float8_e4m3fn:
                emit(name, dequant_fp8_block(t, fetch(scale_key)).to(torch.bfloat16))
            elif scale_key in keys and t.dtype == torch.int8:
                emit(name, dequant_mxfp4(t, fetch(scale_key)).to(torch.bfloat16))
            else:
                assert scale_key not in keys, (key, t.dtype)
                emit(name, t)
        print(f"layer {layer} done", flush=True)
    if cur:
        flush()

    weight_map = {}
    total = len(shards)
    for i, names in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(out, f"part-{i:05d}.safetensors"), os.path.join(out, final))
        weight_map.update({k: final for k in names})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    num_hash = config["num_hash_layers"]
    config["num_hidden_layers"] = len(layers)
    config["compress_ratios"] = [config["compress_ratios"][i] for i in layers]
    config["num_hash_layers"] = sum(1 for i in layers if i < num_hash)
    assert all(i < num_hash for i in layers[: config["num_hash_layers"]]), "hash layers must stay a prefix"
    config["num_nextn_predict_layers"] = 0
    config.pop("quantization_config")
    config.pop("expert_dtype", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for name in ("generation_config.json", "tokenizer.json"):
        shutil.copy(os.path.join(LOCAL, name), os.path.join(out, name))
    # The release ships no Jinja chat template (only encoding/encoding_dsv4.py); embed ours.
    with open(os.path.join(LOCAL, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), "..", "chat_template.jinja")) as f:
        tok_config["chat_template"] = f.read()
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)
    print("done", out)


if __name__ == "__main__":
    main()
