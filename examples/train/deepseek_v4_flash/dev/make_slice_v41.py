"""Build a text-only BF16 N-layer slice of deepseek-ai/DeepSeek-V4.1-Flash-1017 for the Stage 0 roundtrip tests.

The default keeps layers 0, 1, 2, 3, 20 and 24, the fewest that cover every V4.1 layer type:

    layer 0:  sliding-window attention only (compress ratio 0)
    layer 1:  sliding-window attention + Engram n-gram lookup
    layer 2:  compress ratio 2; KV source and index source (owns its compressor and index keys)
    layer 3:  compress ratio 2; reads layer 2's compressed KV and top-k indices
    layer 20: compress ratio 1; KV source, index source, and the candidate-block source
    layer 24: compress ratio 1; index source on layer 20's KV / index keys, filtered by its candidate blocks

They are renumbered 0..N-1 and every layer-id list in the config (``kv_source_layer_ids``,
``index_source_layer_ids``, ``candidate_source_layer_id``, ``engram_layer_ids``) is rewritten to match.
A consumer layer needs its source in the slice; the script checks that. Engram hash multipliers are
seeded by the engram layer id, so an engram layer must keep its id (layer 1 does with the default).

Text-only: the vision tower, aligner, image-span embeddings and the gate's ``bias_vl`` are dropped, as
are the DSpark MTP layers. Everything trainable is dequantized to BF16 and ``quantization_config`` is
removed:

- FP8 (e4m3) weights with 32x32 blocks of ue8m0 ``scale``.
- MXFP4 experts: I8 ``weight`` with two e2m1 values per byte (low nibble first) and ue8m0 ``scale`` per
  32 values along K.

The Engram tables (``layers.N.engram.embed.{weight,scale}``, ~98 GB each in FP8) stay FP8: they are
frozen in training and dequantized row by row on lookup.

Reads the checkpoint from ``DSV41_LOCAL_DIR`` (default ~/models/DeepSeek-V4.1-Flash-1017)
and the chat template from ``DSV41_DOCS_DIR`` (the deepseek-ai/DeepSeek-V4.1-Flash repo's chat_template.jinja).

Usage: python make_slice_v41.py <out_dir> [comma-separated layer ids, default 0,1,2,3,20,24]
"""

import glob
import json
import os
import re
import shutil
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LOCAL = os.environ.get("DSV41_LOCAL_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-1017"))
DOCS = os.environ.get("DSV41_DOCS_DIR", os.path.expanduser("~/models/DeepSeek-V4.1-Flash-docs"))
SHARD_BYTES = 4 << 30
FP8_BLOCK = 32
FP4_BLOCK = 32
DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_start", "image_end", "image_newline")
KEEP_FP8 = re.compile(r"layers\.\d+\.engram\.embed\.(weight|scale)$")

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


def build_index():
    """The release ships no model.safetensors.index.json, so read every shard's header."""
    index = {}
    for path in sorted(glob.glob(os.path.join(LOCAL, "*.safetensors"))):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        index.update({k: os.path.basename(path) for k in header if k != "__metadata__"})
    return index


def slice_text_config(tc, layers):
    remap = {old: new for new, old in enumerate(layers)}

    def ids(key):
        kept = [remap[i] for i in tc[key] if i in remap]
        return kept

    for consumer in layers:
        if tc["compress_ratios"][consumer]:
            sources = [i for i in tc["kv_source_layer_ids"] if i <= consumer]
            assert sources and sources[-1] in remap, f"layer {consumer} reads KV from {sources[-1:]}, not in the slice"
            idx_sources = [i for i in tc["index_source_layer_ids"] if i <= consumer]
            assert idx_sources and idx_sources[-1] in remap, f"layer {consumer} reads top-k from {idx_sources[-1:]}"
        if 0 <= tc["candidate_source_layer_id"] < consumer and consumer in tc["index_source_layer_ids"]:
            assert tc["candidate_source_layer_id"] in remap, f"layer {consumer} needs the candidate source"
    for layer_id in tc["engram_layer_ids"]:
        if layer_id in remap:
            assert remap[layer_id] == layer_id, f"engram layer {layer_id} must keep its id (hash seeds use it)"

    engram = [(remap[i], n) for i, n in zip(tc["engram_layer_ids"], tc["engram_num_embeddings"]) if i in remap]
    cand = tc["candidate_source_layer_id"]
    tc = dict(tc)
    tc["num_hidden_layers"] = len(layers)
    tc["compress_ratios"] = [tc["compress_ratios"][i] for i in layers]
    tc["kv_source_layer_ids"] = ids("kv_source_layer_ids")
    tc["index_source_layer_ids"] = ids("index_source_layer_ids")
    tc["candidate_source_layer_id"] = remap.get(cand, -1)
    tc["engram_layer_ids"] = [i for i, _ in engram]
    tc["engram_num_embeddings"] = [n for _, n in engram]
    # DSpark MTP is dropped.
    tc["num_nextn_predict_layers"] = 0
    tc["dspark_target_layer_ids"] = []
    return tc


def main():
    out = sys.argv[1]
    layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 20, 24]
    assert layers == sorted(layers), "layers must be ascending (sources run before their consumers)"
    remap = {old: new for new, old in enumerate(layers)}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(LOCAL, "config.json")) as f:
        config = json.load(f)
    config["text_config"] = slice_text_config(config["text_config"], layers)
    index = build_index()

    def layer_of(key):
        m = re.match(r"layers\.(\d+)\.", key)
        return int(m.group(1)) if m else None

    def wanted_key(k):
        if k.startswith(DROP_PREFIXES) or k.endswith(".gate.bias_vl"):
            return False
        return layer_of(k) is None or layer_of(k) in remap

    wanted = sorted(k for k in index if wanted_key(k))

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
            if KEEP_FP8.match(key):
                if key.endswith(".weight"):
                    # ~98 GB: give the table and its scale shards of their own.
                    if cur:
                        flush()
                    emit(rename(key), fetch(key))
                    flush()
                    emit(rename(key[: -len(".weight")] + ".scale"), fetch(key[: -len(".weight")] + ".scale"))
                continue
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

    config.pop("quantization_config")
    config.pop("vision_config", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    shutil.copy(os.path.join(LOCAL, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    with open(os.path.join(LOCAL, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    with open(os.path.join(DOCS, "chat_template.jinja")) as f:
        tok_config["chat_template"] = f.read()
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)
    print("done", out)


if __name__ == "__main__":
    main()
