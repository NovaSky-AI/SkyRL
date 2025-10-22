#!/usr/bin/env python3
"""
Standalone converter: DeepSpeed checkpoint -> HuggingFace safetensors.
"""

import argparse
import os
import shutil
import sys
from typing import Dict, Optional, Tuple

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_file as safetensors_save_file
from transformers import AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="Path to DeepSpeed checkpoint directory (containing global_step_N subdirs, or a specific global_step_N dir).")
    parser.add_argument("--out-dir", required=True, help="Output directory for the HuggingFace model.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to HF config.json if not present in <ckpt_dir>/huggingface/.",
    )
    return parser.parse_args()


def find_hf_metadata_dir(ckpt_dir: str) -> Optional[str]:
    """
    Find huggingface metadata directory using DeepSpeed 'latest' file.
    
    Returns path to huggingface directory, or None if not found.
    """
    # Check for DeepSpeed 'latest' file
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.isfile(latest_file):
        try:
            with open(latest_file, "r") as f:
                latest_step_name = f.read().strip()
            latest_step_dir = os.path.join(ckpt_dir, latest_step_name)
            hf_dir = os.path.join(latest_step_dir, "huggingface")
            if os.path.isdir(hf_dir):
                return hf_dir
            print("Could not find HF latest directory")
        except OSError as e:
            print(f"Warning: Could not read 'latest' file in {ckpt_dir}: {e}")
    
    return None


def resolve_config_path(ckpt_dir: str, user_config_path: Optional[str]) -> str:
    """
    Prefer ckpt_dir/huggingface/config.json. Fall back to user-provided --config.
    Error if neither exists.
    """
    if user_config_path and os.path.isfile(user_config_path):
        return user_config_path
    hf_dir = find_hf_metadata_dir(ckpt_dir)
    if hf_dir:
        cfg = os.path.join(hf_dir, "config.json")
        if os.path.isfile(cfg):
            return cfg
    raise FileNotFoundError(
        f"Could not find config.json in checkpoint ({ckpt_dir}/huggingface/config.json) "
        "and no --config was provided."
    )


def config_dir_from_path(config_path: str) -> str:
    """Return the directory to feed into AutoConfig.from_pretrained."""
    if os.path.isdir(config_path):
        return config_path
    # If a file path like .../config.json is provided, use its directory.
    return os.path.dirname(config_path)


def _normalize_keys_to_hf(state_dict) -> Dict[str, torch.Tensor]:
    """Strip leading 'module.' if present; leave other names as-is."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        k_norm = k[7:] if k.startswith("module.") else k
        out[k_norm] = v
    return out


def gather_full_state_dict_zero3(ckpt_dir: str) -> Dict[str, torch.Tensor]:
    """
    Use DeepSpeed zero-to-fp32 to reconstruct the full FP32 parameter state dict from a ZeRO-3 checkpoint.
    Returns a flat dict[name] -> tensor on CPU in FP32.
    """
    raw = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
    if raw is None:
        raise RuntimeError(f"Failed to load ZeRO-3 checkpoint from {ckpt_dir}")
        
    # Ensure tensors are CPU float32 and normalize keys
    out: Dict[str, torch.Tensor] = {}
    raw = _normalize_keys_to_hf(raw)
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().to(torch.float32).cpu()
    return out


def apply_tied_embeddings_drop(state_dict: Dict[str, torch.Tensor], config_path: str) -> None:
    """
    If config.tie_word_embeddings is True and 'lm_head.weight' is present, remove it.
    HF will re-tie on load; this mirrors SkyRL's DS saver.
    """
    cfg = AutoConfig.from_pretrained(config_dir_from_path(config_path))
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" in state_dict:
        state_dict.pop("lm_head.weight", None)


def save_safetensors_and_config(out_dir: str, state_dict: Dict[str, torch.Tensor], config_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    safetensors_save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
    cfg = AutoConfig.from_pretrained(config_dir_from_path(config_path))
    cfg.save_pretrained(out_dir)


def copy_auxiliary_files(hf_src_dir: Optional[str], out_dir: str) -> Tuple[int, int]:
    """Copy all auxiliary files, auto-detecting what's needed."""
    if not hf_src_dir or not os.path.isdir(hf_src_dir):
        return (0, 0)
    
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    found = 0
    for item in os.listdir(hf_src_dir):
        src_path = os.path.join(hf_src_dir, item)
        
        # Skip model weight files (we're replacing these)
        if item.startswith(('pytorch_model', 'model.safetensors')):
            continue
        if item.startswith('.'):
            continue
        if os.path.isfile(src_path):
            found += 1
            try:
                shutil.copy2(src_path, os.path.join(out_dir, item))
                copied += 1
            except OSError as e:
                print(f"Warning: Failed to copy {item}: {e}")
    
    return (copied, found)


def try_hf_builtin_conversion(ckpt_dir: str, out_dir: str, config_path: str) -> bool:
    """
    Try using HuggingFace's built-in conversion first.
    Returns True if successful, False if fallback needed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # HF can auto-detect model type from config.json
        config = AutoConfig.from_pretrained(config_dir_from_path(config_path))
        print(f"Detected model type: {config.model_type}")
        
        # Try to load directly from the checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            config=config,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True  # In case of custom models
        )
        
        # Save as safetensors
        model.save_pretrained(out_dir, safe_serialization=True)
        print(f"Used HuggingFace built-in conversion successfully")
        return True
        
    except Exception as e:
        print(f"HF built-in conversion failed: {e}")
        print(f"Falling back to manual DeepSpeed conversion")
        return False


def main() -> None:
    args = parse_args()
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    out_dir = os.path.abspath(args.out_dir)

    config_path = resolve_config_path(ckpt_dir, args.config)

    if try_hf_builtin_conversion(ckpt_dir, out_dir, config_path):
        hf_src_dir = find_hf_metadata_dir(ckpt_dir)
        copied, total = copy_auxiliary_files(hf_src_dir, out_dir)
        if total != copied:
            print(f"WARNING: Copied {copied}/{total} auxiliary files")
        print(f"[SUCCESS]: safetensors conversion completed")
        return

    try:
        state_dict = gather_full_state_dict_zero3(ckpt_dir)
    except Exception as e:
        print(f"[FAILURE] ZeRO-3 gather failed: {e}", file=sys.stderr)
        sys.exit(1)

    apply_tied_embeddings_drop(state_dict, config_path)
    save_safetensors_and_config(out_dir, state_dict, config_path)

    # Copy tokenizer and generation files if present
    hf_src_dir = find_hf_metadata_dir(ckpt_dir)
    copied, total = copy_auxiliary_files(hf_src_dir, out_dir)
    if total != copied:
        print(f"WARNING: Copied {copied}/{total} auxiliary files")
    print(f"[SUCCESS]: safetensors conversion completed")


if __name__ == "__main__":
    main()