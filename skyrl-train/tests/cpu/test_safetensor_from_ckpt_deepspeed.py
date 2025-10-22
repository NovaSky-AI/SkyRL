#!/usr/bin/env python3
"""
Test script to validate checkpoint to safetensors conversion
"""
import argparse
import subprocess
import os
import torch
from safetensors import safe_open
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

CONVERSION_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "safetensor_from_ckpt_deepspeed.py"))
ALIASES = {
    "lm_head.weight": "transformer.wte.weight",  # common tied embeddings
}

def find_checkpoint_subdir(ckpt_dir: str) -> str:
    latest_file = os.path.join(ckpt_dir, "latest")
    if os.path.isfile(latest_file):
        try:
            with open(latest_file, "r") as f:
                latest_step_name = f.read().strip()
            latest_step_dir = os.path.join(ckpt_dir, latest_step_name)
            if os.path.isdir(latest_step_dir):
                return latest_step_dir
            print("Did not find latest step dir")
        except (IOError, OSError):
            pass
    print("Did not find HF latest file")
    return ckpt_dir

def _get_tensor(d, k):
    if k in d:
        return d[k]
    if k in ALIASES and ALIASES[k] in d:
        return d[ALIASES[k]]
    return None

def validate_conversion(ckpt_dir, safetensors_dir):
    """Compare weights between original and converted checkpoints."""
    print("\nStarting validation...")
    
    safetensor_path = os.path.join(safetensors_dir, "model.safetensors")
    config_path = os.path.join(safetensors_dir, "config.json")
    
    if not os.path.exists(safetensor_path):
        print(f"Safetensor file not found: {safetensor_path}")
        return False
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return False
    
    print(f"Found safetensor file: {safetensor_path}")
    print(f"Found config file: {config_path}")
    
    try:
        print("Reconstructing original DeepSpeed checkpoint...")
        original_state = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
        print(f"Original checkpoint: {len(original_state)} parameters")
        
        print("Loading converted safetensors...")
        safetensor_state = {}
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                safetensor_state[key] = f.get_tensor(key)
        print(f"Safetensors: {len(safetensor_state)} parameters")
        
        # Don’t hard-fail on raw counts—account for aliases (e.g., tied lm_head)
        keys_pt = set(original_state.keys())
        keys_st = set(safetensor_state.keys())
        effective_st = keys_st | {k for k in ALIASES if ALIASES[k] in keys_st}

        only_in_pt = sorted(k for k in keys_pt - effective_st)
        only_in_st = sorted(k for k in keys_st - keys_pt)

        if only_in_pt:
            print(f"Keys only in original ({len(only_in_pt)} shown up to 20):")
            for k in only_in_pt[:20]:
                print("   ", k)
            if len(only_in_pt) > 20:
                print("   ...")

        if only_in_st:
            print(f"Keys only in safetensors ({len(only_in_st)} shown up to 20):")
            for k in only_in_st[:20]:
                print("   ", k)
            if len(only_in_st) > 20:
                print("   ...")
        
        mismatches = 0
        compared = 0

        # Compare tensors on the intersection + aliased keys
        for k in sorted(keys_pt):
            b = _get_tensor(safetensor_state, k)
            if b is None:
                # real miss (not covered by alias)
                continue

            a = original_state[k]

            if a.shape != b.shape:
                print(f"Shape mismatch for {k}: {tuple(a.shape)} vs {tuple(b.shape)}")
                mismatches += 1
                continue

            a32, b32 = a.float(), b.float()
            if not torch.allclose(a32, b32):
                diff = (a32 - b32).abs()
                print(f"Value mismatch for {k}: max|Δ|={diff.max().item():.3e}, mean|Δ|={diff.mean().item():.3e}")
                mismatches += 1

            compared += 1
        
        print(f"\nSummary: compared {compared} tensors "
            f"(original {len(keys_pt)}, safetensors {len(keys_st)}; alias-aware).")
        if mismatches == 0:
            if only_in_pt or only_in_st:
                print("All compared tensors match (there are asymmetric keys; see warnings).")
                return False
            else:
                print(f"All tensors match and key sets align.")
                return True
        else:
            print(f"Found {mismatches} mismatches.")
            return False
    except Exception as e:
        print(f"Error during validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--safetensors-dir", type=str, required=True)
    args = parser.parse_args()

    print(f"Testing conversion from {args.ckpt_dir} to {args.safetensors_dir}")

    try:
        result = subprocess.run(["python3", CONVERSION_SCRIPT_PATH, "--ckpt-dir", args.ckpt_dir, "--out-dir", args.safetensors_dir], 
                                capture_output=True, text=True)
        if result.stderr:
            print(f"Script output (stderr):\n{result.stderr}")
        if result.returncode != 0:
            print("FAILURE: Conversion script failed.")
            return False

        validation_passed = validate_conversion(args.ckpt_dir, args.safetensors_dir)
        if validation_passed:
            print("\nSUCCESS")
        else:
            print("\nFAILURE")
        
        return validation_passed

    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print(f"Script stderr:\n{e.stderr}")
        return False


if __name__ == "__main__":
    main()
    
