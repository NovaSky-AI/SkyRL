#!/usr/bin/env python3
"""
Script to convert DeepSpeed checkpoint -> HuggingFace safetensors.

Assumes ZeRO-3 is used.
"""

import argparse
import gc
import os
import shutil
import sys
from typing import Dict, Optional, Tuple

import torch
from pathlib import Path
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoModelForCausalLM, AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        required=True,
        help="Path to DeepSpeed checkpoint directory (containing global_step_N subdirs, or a specific global_step_N dir).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for the HuggingFace model.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to HF config.json if not present in <ckpt_dir>/huggingface/.",
    )
    parser.add_argument(
        "--max-safetensor-shard-size",
        dest="max_shard_size",
        default="2GB",
        help="Optional maximum shard size for safetensors. Defaults to 2GB.",
    )
    return parser.parse_args()


def find_hf_dir(ckpt_dir: str) -> Optional[str]:
    """
    Returns path to huggingface directory, or None if not found.
    """
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    if os.path.isdir(hf_dir):
        return hf_dir

    print("Could not find HF latest directory")
    return None


def get_hf_config(ckpt_dir: str, user_config_path: Optional[str]) -> str:
    if user_config_path:
        if os.path.isfile(user_config_path):
            return user_config_path
        raise FileNotFoundError(f"Provided --config does not exist: {user_config_path}")

    hf_dir = find_hf_dir(ckpt_dir)
    if hf_dir:
        cfg = os.path.join(hf_dir, "config.json")
        if os.path.isfile(cfg):
            return cfg
    raise FileNotFoundError(f"Could not find config.json in checkpoint ({ckpt_dir}) and no --config was provided.")


def gather_full_state_dict_zero3(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    raw = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
    if raw is None:
        raise RuntimeError(f"Failed to load ZeRO-3 checkpoint from {ckpt_dir}")

    keys_to_process = [k for k, v in raw.items() if isinstance(v, torch.Tensor)]
    normalized = {}
    for k in keys_to_process:
        k_norm = k[7:] if k.startswith("module.") else k
        normalized[k_norm] = raw[k].detach().cpu()
        del raw[k]

    del raw
    gc.collect()
    return normalized


def apply_tied_embeddings_drop(state_dict: Dict[str, torch.Tensor], config_path: str) -> None:
    """
    If config.tie_word_embeddings is True and 'lm_head.weight' is present, remove it.
    HF will re-tie on load. This mirrors SkyRL's DS saver.
    """
    cfg = AutoConfig.from_pretrained(config_path)
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" in state_dict:
        state_dict.pop("lm_head.weight", None)


def save_safetensors_and_config(out_dir: str, state_dict: Dict[str, torch.Tensor], config_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    safetensors_save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
    cfg = AutoConfig.from_pretrained(config_path)
    cfg.save_pretrained(out_dir)


def copy_auxiliary_files(hf_src_dir: Optional[str], out_dir: str):
    """Copy all auxiliary files, auto-detecting what's needed."""
    if not hf_src_dir or not os.path.isdir(hf_src_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    for item in os.listdir(hf_src_dir):
        src_path = os.path.join(hf_src_dir, item)

        # Skip model weight files
        if item.startswith(("model.safetensors")):
            continue
        if os.path.isfile(src_path):
            try:
                shutil.copy2(src_path, os.path.join(out_dir, item))
            except Exception as e:
                print(f"Warning: Failed to copy {item}: {e}")


def hf_builtin_conversion(ckpt_dir: str, out_dir: str, config_path: str, max_shard_size: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(config_path)
        print(f"Detected model type: {config.model_type}")

        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            config=config,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # In case of custom models
        )

        # Save as safetensors
        model.save_pretrained(out_dir, safe_serialization=True, max_shard_size=max_shard_size)
        print(f"Used HuggingFace built-in conversion successfully")

        del model
        gc.collect()
        return True
    except Exception as e:
        print(f"HF built-in conversion failed: {e}")
        if "model" in locals():
            del model
            gc.collect()
        return False


def main():
    args = parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = os.path.abspath(args.out_dir)

    config_path = get_hf_config(ckpt_dir, args.config)

    if not hf_builtin_conversion(ckpt_dir, out_dir, config_path, args.max_shard_size):
        try:
            state_dict = gather_full_state_dict_zero3(ckpt_dir)
            apply_tied_embeddings_drop(state_dict, config_path)
            save_safetensors_and_config(out_dir, state_dict, config_path)
        except Exception as e:
            print(f"[FAILURE] ZeRO-3 gather failed: {e}", file=sys.stderr)
            return

    hf_src_dir = find_hf_dir(ckpt_dir)
    copy_auxiliary_files(hf_src_dir, out_dir)
    print(f"[SUCCESS]: safetensors conversion completed")


if __name__ == "__main__":
    main()
