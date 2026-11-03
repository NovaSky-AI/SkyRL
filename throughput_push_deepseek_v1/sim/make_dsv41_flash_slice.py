"""Write a compressed DeepSeek-V4.1-Flash config for the one-node RL simulation (no weights).

Keeps the real architecture and its layer roles, at a size that colocates a full-fine-tune trainer and
a vLLM engine on one 8xB200 node (~25B params):

- 8 layers in the real order: 2 dense / sliding-window layers, 3 ratio-2 compressed (CSA) layers, 3
  ratio-1 layers; compressed-KV sources at 2 and 5, indexer sources at 2 / 5 / 7, the candidate
  pre-filter at 5 (the real model: 40 layers, sources 2/8/14/20, indexers 2..36, candidate 20);
- 64 routed experts, top-6 (real: 384);
- no Engram layers (their tables are read from checkpoint files, and they are frozen lookups) and no
  MTP / DSpark layers; the checkpoint's FP8 / MXFP4 quantization_config is kept, since vLLM's SM100
  default attention (FlashMLA mega-attention) needs the FP8 weight scales, and that is how the real
  model is served (the BF16 trainer's weights are quantized on each sync);
- a minimal chat template in DeepSeek's turn format (the checkpoint ships ``encoding/encoding.py``
  instead of a Jinja template, which SkyRL's generator needs); simulator-only.

Usage: python make_dsv41_flash_slice.py SRC_CHECKPOINT_DIR OUT_DIR
"""

import json
import os
import re
import shutil
import sys

# Top-level checkpoint keys of components the slice leaves out.
DROPPED_PREFIXES = ("aligner.", "image_", "vision.", "mtp.")

# DeepSeek's turn format (encoding/encoding.py): BOS, <｜User｜> / <｜Assistant｜> markers, EOS after each
# assistant turn.
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ m['content'] }}"
    "{% elif m['role'] == 'user' %}<｜User｜>{{ m['content'] }}"
    "{% elif m['role'] == 'assistant' %}<｜Assistant｜>{{ m['content'] }}<｜end▁of▁sentence｜>"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<｜Assistant｜>{% endif %}"
)


def _role(text_config: dict, layer: int) -> tuple:
    return (
        text_config["compress_ratios"][layer],
        layer in text_config["kv_source_layer_ids"],
        layer in text_config["index_source_layer_ids"],
        layer == text_config["candidate_source_layer_id"],
    )


def write_key_index(src: str, out: str, src_text: dict, out_text: dict) -> None:
    """Write a weight-free ``model.safetensors.index.json`` naming the slice's HF parameters.

    Megatron-Bridge's export (the trainer -> vLLM weight sync) lists the HF keys from the checkpoint's
    index and reads the quantization scales, so the slice needs both even though the trainer random-initializes and vLLM loads dummy
    weights. Each slice layer takes the key set of the first real layer with the same role (compress
    ratio, compressed-KV source, indexer source, candidate source) and no Engram, with experts cut to
    ``n_routed_experts``.
    """
    weight_map = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    per_layer: dict[int, list[str]] = {}
    keys = {"weight_map": {}}
    scale_sources: dict[str, str] = {}  # slice scale key -> source checkpoint key
    for key in weight_map:
        m = re.match(r"layers\.(\d+)\.(.*)", key)
        if m:
            per_layer.setdefault(int(m.group(1)), []).append(m.group(2))
        elif not key.startswith(DROPPED_PREFIXES):
            keys["weight_map"][key] = "model-dummy.safetensors"
    engram = set(src_text.get("engram_layer_ids") or [])
    n_experts = out_text["n_routed_experts"]
    for layer in range(out_text["num_hidden_layers"]):
        template = next(
            real
            for real in range(src_text["num_hidden_layers"])
            if real not in engram and _role(src_text, real) == _role(out_text, layer)
        )
        for name in per_layer[template]:
            e = re.match(r"ffn\.experts\.(\d+)\.", name)
            if e and int(e.group(1)) >= n_experts:
                continue
            keys["weight_map"][f"layers.{layer}.{name}"] = "model-dummy.safetensors"
            if name.endswith(".scale"):
                scale_sources[f"layers.{layer}.{name}"] = f"layers.{template}.{name}"
    for key in keys["weight_map"]:
        if key.endswith(".scale") and key not in scale_sources:
            scale_sources[key] = key
    # The export re-quantizes each FP8 / MXFP4 weight in its checkpoint scale's layout and reads
    # that scale from disk, so the real scales are copied in (weights stay absent).
    from safetensors import safe_open
    from safetensors.torch import save_file

    by_file: dict[str, list[tuple[str, str]]] = {}
    for dst, src_key in scale_sources.items():
        by_file.setdefault(weight_map[src_key], []).append((dst, src_key))
    scales = {}
    for filename, pairs in by_file.items():
        with safe_open(os.path.join(src, filename), framework="pt") as f:
            for dst, src_key in pairs:
                scales[dst] = f.get_tensor(src_key).clone()  # layers sharing a template need their own copy
    save_file(scales, os.path.join(out, "model-scales.safetensors"))
    for dst in scales:
        keys["weight_map"][dst] = "model-scales.safetensors"
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(keys, f, indent=1)


def main(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    config = json.load(open(os.path.join(src, "config.json")))
    src_text = dict(config["text_config"])
    config["text_config"].update(
        num_hidden_layers=8,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[2, 5],
        index_source_layer_ids=[2, 5, 7],
        candidate_source_layer_id=5,
        engram_layer_ids=[],
        engram_num_embeddings=[],
        n_routed_experts=64,
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[5, 6, 7],
    )
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    write_key_index(src, out, src_text, config["text_config"])
    shutil.copy(os.path.join(src, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    tokenizer_config = json.load(open(os.path.join(src, "tokenizer_config.json")))
    tokenizer_config["chat_template"] = CHAT_TEMPLATE
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
    if os.path.isdir(os.path.join(src, "encoding")):
        shutil.copytree(os.path.join(src, "encoding"), os.path.join(out, "encoding"), dirs_exist_ok=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
