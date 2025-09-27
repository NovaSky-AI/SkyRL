"""
Helper Script to convert FSDP shards to safetensor model files, compatible with Huggingface API

The main purpose is to be able to enable users who choose not to enable HF model saves during training, such as enable the `hf_save_interval` parameter, to
also be able to benefit from a way to create a HF safetensors model.

For FSDP2 model shards, the output directory will be created with the following structure:
.
├── added_tokens.json
├── chat_template.jinja (optional: this file is for chat specific tasks)
├── config.json
├── generation_config.json (optional: default decoding parameters)
├── merges.txt
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json

Example usage:
uv run --isolated --frozen --extra vllm scripts/convert_fsdp_to_hf.py --ckpt-dir /home/ubuntu/ckpts/gsm8k_0.5B_ckpt/global_step_10 --out-dir /home/ubuntu/hf/glob_step_10
"""

import argparse
import re
import sys
from pathlib import Path
import shutil
from safetensors.torch import save_file
from typing import Dict



import torch


def find_policy_dir(chkpt_dir):
    pol = chkpt_dir / "policy"
    if not pol.exists():
        print(f"[error] Expected 'policy/' under {chkpt_dir}")
        raise FileNotFoundError(f"Expected 'policy/' under {chkpt_dir}")
    return pol


def get_model_shards(policy_dir: Path):
    shards = sorted(policy_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        shards = sorted(policy_dir.glob("model*.pt"))
    if not shards:
        print(f"[error] No model shards found under {policy_dir}")
        raise FileNotFoundError(f"No model shards found under {policy_dir}")
    return shards


def normalize_key(k: str) -> str:
    k = re.sub(r"^(module|model)\.", "", k)
    k = k.replace("_fsdp_wrapped_module.", "")
    return k


def load_single_shard(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format at {path} (type={type(obj)})")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def merge_shards(shards) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        sd = load_single_shard(shard)
        for k, v in sd.items():
            nk = normalize_key(k)
            if nk in merged:
                if merged[nk].shape != v.shape or merged[nk].dtype != v.dtype:
                    print(
                        f"[error] Key collision with mismatch for '{nk}' between shards "
                        f"(existing {merged[nk].shape}/{merged[nk].dtype} vs {v.shape}/{v.dtype})"
                    )
                    raise ValueError(
                        f"Key collision with mismatch for '{nk}' between shards "
                        f"(existing {merged[nk].shape}/{merged[nk].dtype} vs {v.shape}/{v.dtype})"
                    )
            else:
                merged[nk] = v.detach().cpu().contiguous()
    if not merged:
        
        print("[error] No tensors found in shards")
        raise RuntimeError("No tensors found in shards")
    return merged


def copy_hf_artifacts(policy_dir: Path, out_dir: Path):
    
    hf_src = policy_dir / "huggingface"
    
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
        


def _materialize_for_safetensors(state_dict):
    import torch
    new_sd = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            new_sd[k] = v
            continue
        t = v
        if type(t).__name__ == "DTensor" and hasattr(t, "to_local"):
            t = t.to_local()
        if type(t).__name__ == "ShardedTensor" and hasattr(t, "local_tensor"):
            t = t.local_tensor()
        if getattr(t, "is_meta", False):
            raise RuntimeError(f"Tensor {k} is on meta device; load the real weights before saving.")
        if type(t).__name__ == "FakeTensor":
            raise RuntimeError(f"Tensor {k} is a FakeTensor; disable fake tensor mode for save")
        if t.device.type != "cpu":  
            t = t.to("cpu", non_blocking=False)
        t = t.detach()
        if not t.is_contiguous():
            t = t.contiguous()
        new_sd[k] = t
    return new_sd

def _untie_shared_tensors(sd):
    seen = {}
    for k, v in list(sd.items()):
        if not isinstance(v, torch.Tensor):
            continue
        try:
            ptr = v.storage().data_ptr()
        except Exception:
            continue
        if ptr in seen:
            sd[k] = v.clone()
        else:
            seen[ptr] = (k, v)
    return sd


def main():
    ap = argparse.ArgumentParser(description="Convert FSDP checkpoint shards to a HuggingFace safetensors model.")
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, help="Path to the checkpoint directory, containing trainer_state.pt"
    )
    ap.add_argument("--out-dir", type=str, required=True, help="Output for HF model folder")
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    policy_dir = find_policy_dir(ckpt_dir)
    shards = get_model_shards(policy_dir)
    print(f"[info] Found {len(shards)} model shard(s). ")
    for s in shards:
        print(f"[info] - {s}")
    
    print("[info] Merging shards...")
    state_dict = merge_shards(shards)
    print(f"[info] Merged {len(state_dict)} tensors.")
    
    copy_hf_artifacts(policy_dir, output_dir)
    weights_path = output_dir / "model.safetensors"
    
    clean_sd = _materialize_for_safetensors(state_dict)
    clean_sd = _untie_shared_tensors(clean_sd)

    save_file(clean_sd, str(weights_path))
    print(f"[success] Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
