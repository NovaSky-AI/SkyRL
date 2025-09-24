"""
Helper Script to convert FSDP shards to safetensor model files, compatible with Huggingface API

The main purpose is to be able to enable users who choose not to enable HF model saves during training, such as enable the `hf_save_interval` parameter, to
also be able to benefit from a way to create a HF safetensors model.

Example usage:
uv run --isolated --frozen --extra vllm scripts/convert_fsdp_to_hf.py --ckpt-dir /home/ubuntu/ckpts/gsm8k_0.5B_ckpt/global_step_10 --out-dir /home/ubuntu/hf/glob_step_10
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
import shutil
from safetensors.torch import save_file, save_model
from typing import Dict, Any, Optional
#from loguru import logger

import torch
from safetensors.torch import save_file as save_safetensors

def find_policy_dir(chkpt_dir):
    pol = chkpt_dir / "policy"
    if not pol.exists():
        #logger.error(f"Expected 'policy/' under {chkpt_dir}")
        print(f"[error] Expected 'policy/' under {chkpt_dir}")
        raise FileNotFoundError(f"Expected 'policy/' under {chkpt_dir}")
    return pol

def get_model_shards(policy_dir: Path):
    # policy_dir.glob returns a generator iterating over all matches 
    shards = sorted(policy_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        # check for possibility that model is saved as follows as well
        shards = sorted(policy_dir.glob("model*.pt"))
    if not shards:
        #logger.error(f"No model shards found under {policy_dir}")
        print(f"[error] No model shards found under {policy_dir}")
        raise FileNotFoundError(f"No model shards found under {policy_dir}")
    return shards

def normalize_key(k:str) -> str:
    # Remove certain leading keys / strings
    k = re.sub(r"^(module|model)\.", "", k)
    k = k.replace("_fsdp_wrapped_module.", "")
    return k

def load_single_shard(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # Common containers to unwrap
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format at {path} (type={type(obj)})")
    # Filter to tensors only
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}



def merge_shards(shards) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        sd = load_single_shard(shard)
        for k, v in sd.items():
            nk = normalize_key(k)
            if nk in merged:
                if merged[nk].shape != v.shape or merged[nk].dtype != v.dtype:
                    # TODO: this should probably be a warning
                    # logger.error(f"Key collision with mismatch for '{nk}' between shards "
                    #     f"(existing {merged[nk].shape}/{merged[nk].dtype} vs {v.shape}/{v.dtype})"
                    # )
                    print(f"[error] Key collision with mismatch for '{nk}' between shards "
                        f"(existing {merged[nk].shape}/{merged[nk].dtype} vs {v.shape}/{v.dtype})"
                    )
                    raise ValueError(f"Key collision with mismatch for '{nk}' between shards "
                        f"(existing {merged[nk].shape}/{merged[nk].dtype} vs {v.shape}/{v.dtype})"
                    )
            else:
                merged[nk] = v.detach().cpu().contiguous()
    if not merged:
        #logger.error("No tensors found in shards")
        print("[error] No tensors found in shards")
        raise RuntimeError("No tensors found in shards")
    return merged


def copy_hf_artifacts(policy_dir: Path, out_dir: Path):
    # copy all files from policy dir to out dir
    hf_src = policy_dir / "huggingface"
    # create missing parents and exist_ok is fine
    out_dir.mkdir(parents=True, exist_ok=True)
    if hf_src.exists():
        for p in hf_src.iterdir():
            dst = out_dir / p.name
            if p.is_file():
                shutil.copy2(p, dst)
            elif p.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(p, dst)
    else:
        print("[warn] policy/huggingface not found; you must supply a proper config/tokenizer.", file=sys.stderr)
        #logger.warn("[warn] policy/huggingface not found; you must supply a proper config/tokenizer.", file=sys.stderr)

def _materialize_for_safetensors(state_dict):
    # Make sure all the tensors are VALID
    # This is the method that is supposed to
    # sanitize the state_dict and remove everything
    # meta / fake / DTensor / ShardedTensor or not materialized
    import torch

    new_sd = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            new_sd[k] = v
            continue
        
        t = v

        # Handle DTensor (Python Distributed Tensor) -> local dense tensor
        if type(t).__name__ == "DTensor" and hasattr(t, "to_local"):
            t = t.to_local()
        
        # Handle ShardedTensor (older API) -> local tensor
        if type(t).__name__ == "ShardedTensor" and hasattr(t, "local_tensor"):
            t = t.local_tensor()
        
        # Disallow Meta / Fake tensors
        if getattr(t, "is_meta", False):
            raise RuntimeError(f"Tensor {k} is on meta device; load the real weights before saving.")
        if type(t).__name__ == "FakeTensor":
            raise RuntimeError(f"Tensor {k} is a FakeTensor; disable fake tensor mode for save")
        
        if t.device.type != "cpu":
            # not on the cpu
            t = t.to("cpu", non_blocking=False)
        t = t.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        
        # reassign the (fixed) tensor to the new state dict
        new_sd[k] = t 
    return new_sd

def _untie_shared_tensors(sd):
    # map the storage_ptr -> first (key, tensor)
    seen = {}
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        # get stable id for underlying storage
        try:
            ptr = v.storage().data_ptr()
        except Exception as e:
            # some tensor still doesn't have a real storage, handle later
            continue
        
        if ptr in seen:
            sd[k] = v.clone()
        else:
            seen[ptr] = (k, v)
    return sd



def main():
    ap = argparse.ArgumentParser(description="Convert FSDP checkpoint shards to a HuggingFace safetensors model.")
    ap.add_argument("--ckpt-dir", type=str, required=True,
                    help="Path to the checkpoint directory, containing trainer_state.pt")
    ap.add_argument("--out-dir", type=str, required=True, 
                    help="Output for HF model folder")
    ap.add_argument("--validate-load", action="store_true", help="Try loading with transformers after saving")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    output_dir = Path(args.out_dir).resolve()

    policy_dir = find_policy_dir(ckpt_dir)
    shards = get_model_shards(policy_dir)

    #logger.info(f"[info] Found {len(shards)} model shard(s). ")
    print(f"[info] Found {len(shards)} model shard(s). ")
    for s in shards:
        print(f"[info] - {s}")
        #logger.info(f" - {s}")

    # load shards and merge 
    #logger.info(f"[info] Merging shards...")
    print(f"[info] Merging shards...")
    state_dict = merge_shards(shards)
    print(f"[info] Merged {len(state_dict)} tensors.")
    #logger.info(f"[info] Merged {len(state_dict)} tensors.")

    copy_hf_artifacts(policy_dir, output_dir)

    # save the weights
    weights_path = output_dir / "model.safetensors"

    # import pdb
    # pdb.set_trace()

    clean_sd = _materialize_for_safetensors(state_dict)
    clean_sd = _untie_shared_tensors(clean_sd)

    #save_model(clean_sd, str(weights_path))
    save_file(clean_sd, str(weights_path))
    #save_safetensors(state_dict, str(weights_path))
    print(f"[success] Saved weights to {weights_path}")
    #logger.info(f"[success] Saved weights to {weights_path}")

    if args.validate_load:
        validate_load(output_dir)

if __name__ == "__main__":
    main()