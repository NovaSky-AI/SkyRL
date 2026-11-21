"""Text-only DeepSeek-V4.1-Flash checkpoint (the released FP8 / MXFP4 weights) for the RL recipes.

Drops the vision tower (``vision.*``, ``aligner.*``, ``image_*``), the vision-language gate biases (``*.bias_vl``) and
the DSpark speculative-decoding stages (``mtp.*``), and writes the matching config (no ``vision_config``,
``num_nextn_predict_layers=0``). Shards without dropped tensors are linked (``--link symlink|hardlink``) or copied;
the others are rewritten without them. Weights are not converted: FP8 / MXFP4 stay as released.

    hf download deepseek-ai/DeepSeek-V4.1-Flash --local-dir /models/DeepSeek-V4.1-Flash
    python examples/train/deepseek_v4_flash/make_text_only_checkpoint.py \
        /models/DeepSeek-V4.1-Flash /models/DeepSeek-V4.1-Flash-fp8 --link symlink
"""

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def dropped(name: str) -> bool:
    return (
        name.startswith(("mtp.", "vision.", "aligner.", "image_"))
        or name.endswith(".bias_vl")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--link", choices=["symlink", "hardlink", "copy"], default="copy")
    a = p.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    in_place = os.path.realpath(a.src) == os.path.realpath(a.dst)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg.pop("vision_config", None)
    tc = cfg.get("text_config", cfg)
    tc["num_nextn_predict_layers"] = 0
    tc["dspark_target_layer_ids"] = []
    tc["compress_ratios"] = tc["compress_ratios"][: tc["num_hidden_layers"]]
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=2)

    for f in [] if in_place else os.listdir(a.src):
        if f.endswith((".json", ".jinja", ".txt", ".model", ".py")) and f not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    by_shard = {}
    for k, s in index["weight_map"].items():
        by_shard.setdefault(s, []).append(k)
    weight_map, n_dropped = {}, 0
    for shard, keys in sorted(by_shard.items()):
        keep = [k for k in keys if not dropped(k)]
        n_dropped += len(keys) - len(keep)
        src, dst = os.path.join(a.src, shard), os.path.join(a.dst, shard)
        if not keep:
            if in_place:
                os.remove(src)
            continue
        if in_place and len(keep) == len(keys):
            pass
        elif len(keep) == len(keys):
            if os.path.lexists(dst):
                os.remove(dst)
            if a.link == "symlink":
                os.symlink(os.path.realpath(src), dst)
            elif a.link == "hardlink":
                os.link(src, dst)
            else:
                shutil.copy(src, dst)
        else:
            with safe_open(src, "pt") as r:
                tensors = {k: r.get_tensor(k).clone() for k in keep}
                meta = r.metadata()
            save_file(tensors, dst + ".tmp", metadata=meta)
            os.replace(dst + ".tmp", dst)
        for k in keep:
            weight_map[k] = shard
    index["weight_map"] = weight_map
    json.dump(index, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    print(f"{a.dst}: {len(weight_map)} tensors kept, {n_dropped} dropped")


if __name__ == "__main__":
    main()
