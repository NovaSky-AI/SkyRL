#!/usr/bin/env python3
"""Convert sharded FSDP2 checkpoints to a HuggingFace-compatible checkpoint.

Each rank file contains DTensors sharded along dim 0. This script loads all
rank shards, concatenates the local tensors back into full parameters, and
saves the result as safetensors alongside the tokenizer / config files.

Usage:
    python scripts/convert_fsdp_to_hf.py \
        --ckpt-dir exports/checkpoints/final \
        --output-dir exports/hf-checkpoints \
        --dtype bfloat16
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def discover_shards(ckpt_dir: Path) -> tuple[int, list[Path]]:
    """Find all model shard files and return (world_size, sorted paths)."""
    config_path = ckpt_dir / "fsdp_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)
    world_size = cfg["world_size"]

    shard_paths = []
    for rank in range(world_size):
        p = ckpt_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
        shard_paths.append(p)

    return world_size, shard_paths


def merge_shards(shard_paths: list[Path], target_dtype: torch.dtype | None) -> dict[str, torch.Tensor]:
    """Load all rank shards and concatenate DTensor local tensors."""
    print(f"Loading {len(shard_paths)} shards...")

    all_shards = []
    for path in shard_paths:
        print(f"  Loading {path.name}...")
        sd = torch.load(path, map_location="cpu", weights_only=False)
        all_shards.append(sd)

    keys = list(all_shards[0].keys())
    print(f"  Merging {len(keys)} parameters...")

    merged = {}
    for key in keys:
        tensors = []
        for sd in all_shards:
            t = sd[key]
            local = t._local_tensor if hasattr(t, "_local_tensor") else t
            tensors.append(local)

        first = all_shards[0][key]
        if hasattr(first, "placements") and len(first.placements) > 0:
            placement = first.placements[0]
            if hasattr(placement, "dim"):
                shard_dim = placement.dim
            else:
                shard_dim = 0
        else:
            shard_dim = 0

        full_tensor = torch.cat(tensors, dim=shard_dim)

        if target_dtype is not None and full_tensor.is_floating_point():
            full_tensor = full_tensor.to(target_dtype)

        merged[key] = full_tensor

    del all_shards
    return merged


def save_hf_checkpoint(merged: dict[str, torch.Tensor], output_dir: Path, ckpt_dir: Path):
    """Save merged weights as safetensors and copy tokenizer/config files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_source = ckpt_dir / "huggingface"

    if hf_source.exists():
        for f in hf_source.iterdir():
            shutil.copy2(f, output_dir / f.name)
            print(f"  Copied {f.name}")

    has_lm_head = any("lm_head" in k for k in merged)
    config_path = output_dir / "config.json"
    if config_path.exists() and has_lm_head:
        with open(config_path) as f:
            config = json.load(f)
        if "architectures" not in config:
            model_type = config.get("model_type", "")
            arch_map = {
                "qwen3_5": "Qwen3_5ForConditionalGeneration",
                "qwen3_5_text": "Qwen3_5ForCausalLM",
                "qwen3": "Qwen3ForCausalLM",
                "qwen2": "Qwen2ForCausalLM",
                "llama": "LlamaForCausalLM",
            }
            arch = arch_map.get(model_type)
            if arch:
                config["architectures"] = [arch]
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"  Added architectures: [{arch}] to config.json")

    total_bytes = sum(t.nelement() * t.element_size() for t in merged.values())
    total_gb = total_bytes / (1024 ** 3)
    print(f"  Total model size: {total_gb:.2f} GB")

    MAX_SHARD_GB = 5
    if total_gb > MAX_SHARD_GB:
        shard_max_bytes = MAX_SHARD_GB * (1024 ** 3)
        shard_idx = 0
        current_shard = {}
        current_bytes = 0
        weight_map = {}
        shard_files = []

        keys = list(merged.keys())
        for key in keys:
            tensor = merged.pop(key)
            t_bytes = tensor.nelement() * tensor.element_size()

            if current_bytes > 0 and current_bytes + t_bytes > shard_max_bytes:
                shard_name = f"model-{shard_idx + 1:05d}-of-TOTAL.safetensors"
                save_file(current_shard, output_dir / shard_name)
                shard_files.append(shard_name)
                print(f"  Saved {shard_name} ({current_bytes / (1024**3):.2f} GB)")
                shard_idx += 1
                current_shard = {}
                current_bytes = 0

            current_shard[key] = tensor
            current_bytes += t_bytes

        if current_shard:
            shard_name = f"model-{shard_idx + 1:05d}-of-TOTAL.safetensors"
            save_file(current_shard, output_dir / shard_name)
            shard_files.append(shard_name)
            print(f"  Saved {shard_name} ({current_bytes / (1024**3):.2f} GB)")

        total_shards = len(shard_files)
        final_names = []
        for old_name in shard_files:
            new_name = old_name.replace("TOTAL", f"{total_shards:05d}")
            (output_dir / old_name).rename(output_dir / new_name)
            final_names.append(new_name)

        all_keys = list(torch.load(ckpt_dir / f"model_world_size_{4}_rank_0.pt", map_location="cpu", weights_only=False).keys())
        key_idx = 0
        for shard_i, shard_name in enumerate(final_names):
            from safetensors import safe_open
            with safe_open(output_dir / shard_name, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = shard_name

        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        index_path = output_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Saved {index_path.name}")
    else:
        save_file(merged, output_dir / "model.safetensors")
        print(f"  Saved model.safetensors ({total_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP2 sharded checkpoints to HuggingFace format")
    parser.add_argument("--ckpt-dir", type=str, default="exports/checkpoints/final",
                        help="Path to the FSDP checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="exports/hf-checkpoints",
                        help="Path to write the HuggingFace checkpoint")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=list(DTYPE_MAP.keys()),
                        help="Target dtype for saved weights (default: bfloat16)")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    target_dtype = DTYPE_MAP[args.dtype]

    print(f"Converting FSDP checkpoint: {ckpt_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target dtype: {args.dtype}")

    world_size, shard_paths = discover_shards(ckpt_dir)
    print(f"Found {world_size} shards (FSDP2)")

    merged = merge_shards(shard_paths, target_dtype)

    sample_key = next(iter(merged))
    print(f"  Sample: {sample_key} -> {merged[sample_key].shape} ({merged[sample_key].dtype})")

    print("Saving HuggingFace checkpoint...")
    save_hf_checkpoint(merged, output_dir, ckpt_dir)

    print(f"\nDone! HuggingFace checkpoint saved to: {output_dir}")
    print(f"You can now upload with:")
    print(f"  hf upload <your-username>/<model-name> {output_dir}")


if __name__ == "__main__":
    main()
