#!/usr/bin/env python3
"""
Script to convert DeepSpeed checkpoint -> HuggingFace safetensors.
Assumes ZeRO-3 is used.
"""

import argparse
import gc
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from safetensors.torch import save_model
from transformers import AutoModelForCausalLM, AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--validate-load", action="store_true", help="Validate conversion by loading model")
    return parser.parse_args()


def find_hf_dir(ckpt_dir: Path) -> Optional[Path]:
    """Returns path to huggingface directory, or None if not found."""
    hf_dir = ckpt_dir / "huggingface"
    if hf_dir.is_dir():
        return hf_dir
    print("Could not find HF directory")
    return None


def get_hf_config(ckpt_dir: Path, user_config_path: Optional[str]) -> Path:
    if user_config_path:
        config_path = Path(user_config_path)
        if config_path.is_file():
            return config_path
        raise FileNotFoundError(f"Provided --config does not exist: {user_config_path}")

    hf_dir = find_hf_dir(ckpt_dir)
    if hf_dir:
        cfg = hf_dir / "config.json"
        if cfg.is_file():
            return cfg
    raise FileNotFoundError(f"Could not find config.json in checkpoint ({ckpt_dir})")


def gather_full_state_dict_zero3(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    raw = get_fp32_state_dict_from_zero_checkpoint(str(ckpt_dir))
    if raw is None:
        raise RuntimeError(f"Failed to load ZeRO-3 checkpoint from {ckpt_dir}")

    if isinstance(raw, dict):
        for key in ["module", "model_state_dict", "state_dict"]:
            if key in raw:
                raw = raw[key]
                break

    keys_to_process = [k for k, v in raw.items() if isinstance(v, torch.Tensor)]
    normalized = {}
    for k in keys_to_process:
        k_norm = k.removeprefix("module.")
        normalized[k_norm] = raw[k].detach().cpu()
        del raw[k]

    del raw
    gc.collect()
    return normalized


def copy_auxiliary_files(hf_src_dir: Optional[Path], out_dir: Path):
    """Copy configuration and tokenizer files (not model weights)."""
    if not hf_src_dir or not hf_src_dir.is_dir():
        return

    out_dir.mkdir(exist_ok=True, parents=True)

    for item in hf_src_dir.iterdir():
        if item.suffix in {".json", ".txt", ".jinja", ".model"}:
            try:
                shutil.copy2(item, out_dir / item.name)
                print(f"    Copied: {item.name}")
            except Exception as e:
                print(f"Warning: Failed to copy {item.name}: {e}")


def convert_and_save_safetensor(ckpt_dir: Path, out_dir: Path, config_path: Path) -> bool:
    try:
        config = AutoConfig.from_pretrained(str(config_path))
        print(f"Detected model type: {config.model_type}")

        print("[1/4] Gathering state dict from ZeRO-3 checkpoint...")
        state = gather_full_state_dict_zero3(ckpt_dir)

        print("[2/4] Initializing HuggingFace model...")
        hf_dir = find_hf_dir(ckpt_dir)
        if not hf_dir:
            raise RuntimeError("HuggingFace directory not found")

        model = AutoModelForCausalLM.from_pretrained(str(hf_dir), dtype="auto", trust_remote_code=True)

        print("[3/4] Loading state dict...")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        del state
        gc.collect()

        print("[4/4] Saving model.safetensors...")
        out_dir.mkdir(exist_ok=True, parents=True)
        save_model(model, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        print(f"Model saved to {out_dir}")

        del model
        gc.collect()
        return True
    except Exception as e:
        print(f"HF built-in conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def validate_load(out_dir: Path):
    """Validate that the saved model can be loaded."""
    try:
        _ = AutoModelForCausalLM.from_pretrained(str(out_dir), local_files_only=True, trust_remote_code=True)
        print("[validate] HF Load OK")
    except Exception as e:
        print(f"[validate] HF Load failed: {e}")
        raise


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    config_path = get_hf_config(ckpt_dir, args.config)

    if not convert_and_save_safetensor(ckpt_dir, out_dir, config_path):
        print("\n[FAILURE] safetensors conversion failed")
        return

    hf_src_dir = find_hf_dir(ckpt_dir)
    copy_auxiliary_files(hf_src_dir, out_dir)

    print("\n[SUCCESS] safetensors conversion completed!")

    if args.validate_load:
        validate_load(out_dir)


if __name__ == "__main__":
    main()
