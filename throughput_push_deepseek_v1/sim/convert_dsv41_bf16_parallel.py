"""Parallel text-only BF16 conversion of DeepSeek-V4.1-Flash (or an N-layer slice of it).

Same output as make_slice_v41.py (dequantized FP8 / MXFP4 -> BF16, Engram tables kept FP8, MTP / vision
dropped), but each layer is converted by its own worker process, so the full 40-layer model takes minutes
instead of hours. The tokenizer gets the chat-mode template (thinking off), as the 2-node LoRA recipe expects.

    DSV41_LOCAL_DIR=/shared/models/DeepSeek-V4.1-Flash-1017 \
      python convert_dsv41_bf16_parallel.py <out_dir> [layer ids, default all] [workers, default 24]
"""

import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import make_slice_v41 as ms  # noqa: E402

SHARD_BYTES = 4 << 30


def convert_group(args):
    out, group, keys_by_layer, index, remap = args
    torch.set_num_threads(4)
    handles = {}

    def fetch(key):
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(ms.LOCAL, shard), framework="pt")
        return handles[shard].get_tensor(key)

    def rename(key):
        parts = key.split(".", 2)
        if parts[0] == "layers":
            return f"layers.{remap[int(parts[1])]}.{parts[2]}"
        return key

    written, cur, cur_bytes = [], {}, 0

    def flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        fname = f"part-g{group:03d}-{len(written):04d}.safetensors"
        save_file(cur, os.path.join(out, fname))
        written.append((fname, list(cur)))
        cur, cur_bytes = {}, 0

    def emit(name, t):
        nonlocal cur_bytes
        cur[name] = t.contiguous()
        cur_bytes += t.numel() * t.element_size()
        if cur_bytes >= SHARD_BYTES:
            flush()

    keys = set(keys_by_layer)
    for key in sorted(keys):
        if ms.KEEP_FP8.match(key):
            if key.endswith(".weight"):
                flush()
                emit(rename(key), fetch(key))
                flush()
                skey = key[: -len(".weight")] + ".scale"
                emit(rename(skey), fetch(skey))
            continue
        if key.endswith(".scale") and key[: -len(".scale")] + ".weight" in keys:
            continue
        t = fetch(key)
        scale_key = key[: -len(".weight")] + ".scale" if key.endswith(".weight") else None
        if scale_key in keys and t.dtype == torch.float8_e4m3fn:
            emit(rename(key), ms.dequant_fp8_block(t, fetch(scale_key)).to(torch.bfloat16))
        elif scale_key in keys and t.dtype == torch.int8:
            emit(rename(key), ms.dequant_mxfp4(t, fetch(scale_key)).to(torch.bfloat16))
        else:
            assert scale_key not in keys, (key, t.dtype)
            emit(rename(key), t)
    flush()
    print(f"group {group} done ({len(written)} files)", flush=True)
    return written


def main():
    out = sys.argv[1]
    with open(os.path.join(ms.LOCAL, "config.json")) as f:
        config = json.load(f)
    n_layers = config["text_config"]["num_hidden_layers"]
    layers = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else list(range(n_layers))
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    remap = {old: new for new, old in enumerate(layers)}
    os.makedirs(out, exist_ok=True)
    config["text_config"] = ms.slice_text_config(config["text_config"], layers)
    index = ms.build_index()

    def layer_of(k):
        p = k.split(".", 2)
        return int(p[1]) if p[0] == "layers" else None

    groups = {}
    for k in index:
        if k.startswith(ms.DROP_PREFIXES) or k.endswith(".gate.bias_vl"):
            continue
        layer = layer_of(k)
        if layer is not None and layer not in remap:
            continue
        groups.setdefault(-1 if layer is None else layer, []).append(k)

    jobs = [(out, i, keys, index, remap) for i, (_, keys) in enumerate(sorted(groups.items()))]
    weight_map = {}
    files = []
    with ProcessPoolExecutor(workers) as pool:
        for written in pool.map(convert_group, jobs):
            files.extend(written)
    total = len(files)
    for i, (fname, names) in enumerate(sorted(files), 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(out, fname), os.path.join(out, final))
        weight_map.update({k: final for k in names})
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    config.pop("quantization_config")
    config.pop("vision_config", None)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.join(ms.LOCAL, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    with open(os.path.join(ms.LOCAL, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    with open(os.path.join(HERE, "chat_template_v41_chat.jinja")) as f:
        tok_config["chat_template"] = f.read()
    with open(os.path.join(out, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2, ensure_ascii=False)
    print("done", out, total, "files", flush=True)


if __name__ == "__main__":
    main()
