"""Cut a smaller slice out of a DeepSeek-V4.1 slice built by make_slice_v41.py (no dequantization, just a copy).

Used to bisect Megatron-vs-vLLM mismatches by layer type. Layer ids are the source slice's; config lists are
rewritten with the same rules (and checks) as make_slice_v41.py.

Usage: python sub_slice_v41.py <src_slice_dir> <out_dir> <comma-separated layer ids>
"""

import json
import os
import re
import shutil
import sys

from make_slice_v41 import slice_text_config
from safetensors import safe_open
from safetensors.torch import save_file

SHARD_BYTES = 4 << 30


def main():
    src, out = sys.argv[1], sys.argv[2]
    layers = [int(x) for x in sys.argv[3].split(",")]
    remap = {old: new for new, old in enumerate(layers)}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    config["text_config"] = slice_text_config(config["text_config"], layers)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]

    def target(key):
        m = re.match(r"layers\.(\d+)\.(.*)", key)
        if not m:
            return key
        layer = int(m.group(1))
        return f"layers.{remap[layer]}.{m.group(2)}" if layer in remap else None

    shards, cur, cur_bytes = [], {}, 0

    def flush():
        nonlocal cur, cur_bytes
        if not cur:
            return
        fname = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(cur, os.path.join(out, fname))
        shards.append({k: fname for k in cur})
        cur, cur_bytes = {}, 0

    by_file = {}
    for key, fname in index.items():
        if target(key) is not None:
            by_file.setdefault(fname, []).append(key)
    for fname in sorted(by_file):
        with safe_open(os.path.join(src, fname), framework="pt") as f:
            for key in sorted(by_file[fname]):
                t = f.get_tensor(key)
                cur[target(key)] = t
                cur_bytes += t.numel() * t.element_size()
                if cur_bytes >= SHARD_BYTES:
                    flush()
    flush()
    weight_map = {k: v for names in shards for k, v in names.items()}
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(src, name), os.path.join(out, name))
    print("done", out, len(weight_map), "tensors")


if __name__ == "__main__":
    main()
