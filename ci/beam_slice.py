"""Build the 6-layer Beam slice used by the beam-6layer test rows (SKYRL_BEAM_SLICE_PATH).

Keeps layers 0-5 (dense 0-2, MoE 3-5; sliding-window 0-4, full/NoPE+SSMax 5) plus the embeddings, final norm and
LM head, and truncates the per-layer config lists to match. Only the shards that hold those tensors are downloaded,
one at a time, and each is deleted once its tensors are written, so peak disk use is the slice plus one shard.

Usage: HF_TOKEN=... python ci/beam_slice.py <out_dir>
"""

import argparse
import json
import os
import re
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "reflection/Beam-1.0-501B-A23B"
NUM_LAYERS = 6
PER_LAYER_CONFIG_KEYS = ("layer_types", "mlp_layer_types")
SKIP_FILES = ("config.json", "model.safetensors.index.json")


def keep(key: str) -> bool:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return match is None or int(match.group(1)) < NUM_LAYERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    os.makedirs(args.out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.out_dir) as download_dir:

        def fetch(filename: str) -> str:
            return hf_hub_download(REPO, filename, token=token, local_dir=download_dir)

        weight_map = json.load(open(fetch("model.safetensors.index.json")))["weight_map"]
        wanted = {k for k in weight_map if keep(k)}
        shards = sorted({weight_map[k] for k in wanted})
        print(f"{len(wanted)} tensors across {len(shards)} shards", flush=True)

        new_map, total_size = {}, 0
        for i, shard in enumerate(shards, start=1):
            shard_path = fetch(shard)
            tensors = {}
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    if key in wanted:
                        tensors[key] = f.get_tensor(key)
            name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(tensors, os.path.join(args.out_dir, name), metadata={"format": "pt"})
            for key, tensor in tensors.items():
                new_map[key] = name
                total_size += tensor.numel() * tensor.element_size()
            del tensors
            os.remove(shard_path)
            print(f"[{i}/{len(shards)}] {shard} -> {name}", flush=True)

        missing = wanted - new_map.keys()
        assert not missing, f"tensors missing from the slice: {sorted(missing)[:5]}"
        with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": new_map}, f, indent=2)

        config = json.load(open(fetch("config.json")))
        config["num_hidden_layers"] = NUM_LAYERS
        for key in PER_LAYER_CONFIG_KEYS:
            config[key] = config[key][:NUM_LAYERS]
        with open(os.path.join(args.out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Tokenizer, generation config, chat template.
        for filename in HfApi(token=token).list_repo_files(REPO):
            if (
                filename.endswith(".safetensors")
                or filename in SKIP_FILES
                or filename.startswith(".")
                or "/" in filename
            ):
                continue
            shutil.copy(fetch(filename), os.path.join(args.out_dir, filename))

    print(f"slice written to {args.out_dir}: {len(new_map)} tensors, {total_size / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
