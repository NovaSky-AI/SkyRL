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
from typing import Dict, List, Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
import torch


def find_policy_dir(chkpt_dir: Path) -> Path:
    """
    Return a Path object to the policy directory
        - Path to the policy directory
    """
    pol = chkpt_dir / "policy"
    if not pol.exists():
        print(f"[error] Expected 'policy/' under {chkpt_dir}")
        raise FileNotFoundError(f"Expected 'policy/' under {chkpt_dir}")
    return pol


def get_model_shards(policy_dir: Path) -> List[Path]:
    """
    Return a list of model Path objects
        - List[Path] of the model shards (the model*.pt files)
    """
    shards_paths = sorted(policy_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards_paths:
        shards_paths = sorted(policy_dir.glob("model*.pt"))
    if not shards_paths:
        print(f"[error] No model shards found under {policy_dir}")
        raise FileNotFoundError(f"No model shards found under {policy_dir}")
    return shards_paths


# Not used at the moment
def normalize_key(k: str) -> str:
    """
    Return a normalized key to ensure consistency across checkpointing frameworks
    Example - Attention layer training:
        "module.encoder.layer.0.attention.self.query.weight"
        "model.module.encoder.layer.0.attention.self.query.weight"
        "encoder.layer.0.attention.self.query.weight"
    These 3 should refer to the same thing, so they should be normalized.

    Function takes string and removes all possible prefixes.
    """
    k = re.sub(r"^(module|model)\.", "", k)
    k = k.replace("_fsdp_wrapped_module.", "")
    return k


def load_single_shard(path: Path) -> Dict[str, torch.Tensor]:
    """
    Load a single model shard and return a dictionary of tensors
        - Dict[str, torch.Tensor]
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format at {path} (type={type(obj)})")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def is_vocab_key(k: str) -> bool:
    """
    Check for the keys in the state_dict that we want to merge shards for, in a MultiGPU setup

    These keys are the keys in the state_dict that should be merged into a single tensor (from all the shards)
    """
    suffixes = (
        "embed_tokens.weight",
        "lm_head.weight",
        "word_embeddings.weight",
        "wte.weight",
    )
    return any(k.endswith(sfx) for sfx in suffixes)


def merge_two_shards(
    existing: torch.Tensor, new_shard: torch.Tensor, key: str, merge_type: Optional[str] = None
) -> torch.Tensor:
    """
    Merge two tensor shards into a single tensor, containing both the existing and new shards
    Merge tensors with key in the vocabulary as the default case, always.
    If key (tensor name) is not in the vocabulary then we merge based on the `merge_type` parameter
        - (str=default) Default: On the first dimension of the tensor
        - (str=vocab_parallel) Vocab-parallel (embeddings/lm_head): cat_dim = 0 (batch / words)
        - (str=col_parallel) Column-parallel linear (weights split output features): cat_dim = 1 (output features)
        - (str=row_parallel) Row-parallel linear ( weights split input features): cat_dim = 2 (input features)
        - (str=other) Replicated Parameters (LayerNorm, RMSNorm, rotary, etc): sum existing and new_shard

    If the merge_type is equal to None, then we use heuristic fallback.

    Heuristic Fallback:
      * If only dim0 differs -> cat dim=0
      * If only dim1 differs -> cat dim=1
      * If only dim2 differs -> cat dim=2
      * If shapes equal -> add (sum)  (useful for row-parallel biases)
    """
    if is_vocab_key(key) or merge_type == "default":
        return torch.cat([existing, new_shard], dim=0)

    if merge_type == "vocab_parallel":
        return torch.cat([existing, new_shard], dim=0)
    elif merge_type == "col_parallel":
        if existing.ndim >= 2 and new_shard.ndim >= 2:
            return torch.cat([existing, new_shard], dim=1)
        else:
            raise ValueError(
                f"existing.ndim={existing.ndim}, new_shard.ndim={new_shard.ndim}, cannot do col_parallel merging because at least 2 dimensions of both tensors are required"
            )
    elif merge_type == "row_parallel":
        if existing.ndim >= 3 and new_shard.ndim >= 3:
            return torch.cat([existing, new_shard], dim=2)
        else:
            raise ValueError(
                f"existing.ndim={existing.ndim}, new_shard.ndim={new_shard.ndim}, cannot do row_parallel merging because at least 3 dimensions of both tensors are required"
            )
        return torch.cat([existing, new_shard], dim=2)
    elif merge_type == "other":
        return existing
    else:
        ## merge_type = None or unknown, then we simply merge by heuristic
        if existing.ndim >= 2 and existing.shape[0] != new_shard.shape[0] and existing.shape[1] == new_shard.shape[1]:
            # Likely word-parallel linear weight (PyTorch Linear is [out, in])
            return torch.cat((existing, new_shard), dim=0)
        if existing.ndim >= 2 and existing.shape[0] == new_shard.shape[0] and existing.shape[1] != new_shard.shape[1]:
            # Likely col-parallel linear weight
            return torch.cat((existing, new_shard), dim=1)
        if (
            existing.ndim >= 3
            and existing.shape[0] == new_shard.shape[0]
            and existing.shape[1] == new_shard.shape[1]
            and existing.shape[2] != new_shard.shape[2]
        ):
            # Likely row-parallel linear weight
            return torch.cat((existing, new_shard), dim=2)
        if existing.shape == new_shard.shape:
            # Could be row-parallel bias or replicated tensors.
            # Try SUM
            return existing + new_shard
        raise ValueError(f"Don't know how to merge key '{key}' with shapes {existing.shape} and {new_shard.shape}")


def merge_shards(shards_paths: List[Path]) -> Dict[str, torch.Tensor]:
    """
    Merge all model shards into a single dictionary of string-based keys to their corresponding tensors
        - Dict[str, torch.Tensor]
    """
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards_paths:
        sd = load_single_shard(shard)
        for k, v in sd.items():
            # nk = normalize_key(k)
            nk = k
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
                    ## Merged here
                    merged[nk] = merge_two_shards(merged[nk], v.detach().cpu().contiguous())
            else:
                merged[nk] = v.detach().cpu().contiguous()
    if not merged:
        print("[error] No tensors found in shards")
        raise RuntimeError("No tensors found in shards")
    return merged


def copy_hf_artifacts(policy_dir: Path, out_dir: Path) -> None:
    """
    Copy huggingface artifacts from the policy directory to the output directory
    - A utility function that copies huggingface artifacts from the policy directory to the output directory
    """
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


def _materialize_for_safetensors(state_dict) -> Dict[str, torch.Tensor]:
    """
    Materialize the state dict for safetensors
    - A utility function that materializes the state dict for safetensors
    Essentially converts all torch tensors to local tensors so they can actually be saved.
    1) DTensor to local tensor
    2) ShardedTensor to local tensor
    We do not save meta tensors because they have no data and are not materializable.

    Then after that, convert these local tensors to cpu tensors, and create a new dictionary of keys -> Tensors.
    """
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


def guess_hf_class(cfg: AutoConfig):
    """
    Tries to find a reasonable HF class from config
    Falls back to the AutoModel architecture if an LM head can't be detected
    """
    if getattr(cfg, "is_encoder_decoder", False):
        return AutoModelForSeq2SeqLM
    archs = getattr(cfg, "architectures", []) or []
    if any(a.endswith("ForCausalLM") for a in archs):
        return AutoModelForCausalLM
    decoders = {"gpt2", "gpt_bigcode", "llama", "mistral", "qwen", "qwen2", "internlm", "mpt", "phi", "falcon"}
    if getattr(cfg, "model_type", "") in decoders:
        return AutoModelForCausalLM
    return AutoModel


def validate_load(out_dir: Path):
    """
    Optional: sanity-load with HF to ensure the saved safetensors is consumable
    Loads on the CPU to avoid device / dtype quirk (this may be a problem for loading on GPU which could cause data loading issues)
    """
    try:
        cfg = AutoConfig.from_pretrained(out_dir, local_files_only=True, trust_remote_code=True)
        HFClass = guess_hf_class(cfg)
        _ = HFClass.from_pretrained(
            out_dir, local_files_only=True, device_map=None, dtype="auto", trust_remote_code=True
        )
        print("[validate] HF Load OK")
    except Exception as e:
        print("[validate][error] HF Load failed: {e} ", e)
        raise RuntimeError("HF Load failed")


def _untie_shared_tensors(sd) -> Dict[str, torch.Tensor]:
    """
    Untie shared tensors
    - A utility function that unties shared tensors
    Some tensors may be shared by different keys in that the tensors they point to have the same data pointer.
    This function takes a state dict and returns a new state dict where the shared tensors have been untied.
    This is done by creating a new tensor (clone it) for each shared tensor.
    This allows each key to refer to a UNIQUE tensor
    """
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
    shards_paths = get_model_shards(policy_dir)
    print(f"[info] Found {len(shards_paths)} model shard(s). ")
    for s in shards_paths:
        print(f"[info] - {s}")

    print("[info] Merging shards...")
    state_dict = merge_shards(shards_paths)
    print(f"[info] Merged {len(state_dict)} tensors.")

    copy_hf_artifacts(policy_dir, output_dir)

    clean_sd = _materialize_for_safetensors(state_dict)
    clean_sd = _untie_shared_tensors(clean_sd)

    # save_file(clean_sd, str(weights_path))
    # print(f"[success] Saved weights to {weights_path}")

    cfg = AutoConfig.from_pretrained(output_dir, local_files_only=True, trust_remote_code=True)
    HFClass = guess_hf_class(cfg)
    hf_model = HFClass.from_config(cfg)
    hf_model.save_pretrained(
        save_directory=output_dir,
        state_dict=clean_sd,
    )

    if args.validate_load:
        validate_load(output_dir)


if __name__ == "__main__":
    main()
