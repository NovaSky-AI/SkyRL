"""
Systematic converter: DeepSpeed ZeRO checkpoint → Hugging Face safetensors model.

Assumptions:
- You have a structure like:
    data.pt
    trainer_state.pt
    policy/
      ├── global_step_x/
      │     ├── zero_pp_rank_0_mp_rank_00_model_states.pt
      │     └── zero_pp_rank_0_mp_rank_00_optim_states.pt
      ├── huggingface/
      │     ├── config.json, tokenizer.json, etc.
      └── zero_to_fp32.py
      └── latest


Output:
    policy/huggingface_converted/model.safetensors  (+ copied config/tokenizer)

For Deepspeed model shards, the output directory will be created with the following structure:
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
uv run --isolated --frozen --extra vllm scripts/convert_deepspeed_to_hf.py --ckpt-dir [local_checkpoint] --out-dir [output_directory]
"""

import json
import shutil
import os
import subprocess
import argparse
import torch
from pathlib import Path
from safetensors.torch import save_model
from transformers import AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM, AutoModel

# === Directories ===
def main(deepspeed_model_path: Path, out_dir:Path = None) -> Path:
    ROOT = deepspeed_model_path
    POLICY_DIR = ROOT / "policy"
    HF_BASE = POLICY_DIR / "huggingface"
    OUT_DIR = POLICY_DIR / "huggingface_converted" if not out_dir else out_dir
    MERGED_FP32 = OUT_DIR / "merged_model" # directory that will store the ultimate pytorch weights. 

    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # === 1. Merge ZeRO shards into single FP32 checkpoint ===
    zero2fp32_script = POLICY_DIR / "zero_to_fp32.py"

    if not MERGED_FP32.exists():
        print(f"[1/5] Merging ZeRO shards from {POLICY_DIR} ...")
        cmd = f"python {zero2fp32_script} {POLICY_DIR} {MERGED_FP32}"
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError("zero_to_fp32.py merge failed.")
    else:
        print(f"[1/5] Merged model already exists → {MERGED_FP32}")

    # === 2. Load merged state dict ===
    print("[2/5] Loading merged model ...")
    state = torch.load(MERGED_FP32 / "pytorch_model.bin", map_location="cpu")

    # Handle possible wrapper keys
    if isinstance(state, dict):
        for key in ["module", "model_state_dict", "state_dict"]:
            if key in state:
                state = state[key]
                break

    merged_bin = MERGED_FP32 / "pytorch_model.bin"
    hf_model_bin = HF_BASE / "pytorch_model.bin"
    shutil.copy2(merged_bin, hf_model_bin)
    print(f"    Copied to: {hf_model_bin}")

    # === 3. Load HF config and initialize model ===
    print("[3/5] Initializing Hugging Face model ...")
    model = AutoModelForCausalLM.from_pretrained(HF_BASE, torch_dtype=torch.float16)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"    → Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # === 4. Save to safetensors ===
    print("[4/5] Saving model.safetensors ...")
    save_model(model, str(OUT_DIR / "model.safetensors"), metadata={"format": "pt"})

    # === 5. Copy tokenizer + config files ===
    print("[5/5] Copying tokenizer/config files ...")
    for fname in os.listdir(HF_BASE):
        if fname.endswith((".json", ".txt", ".jinja")):
            shutil.copy(HF_BASE / fname, OUT_DIR / fname)

    # === Summary ===
    print("\n✅ Conversion complete!")
    print(f"→ Hugging Face safetensors model located at: {OUT_DIR.resolve()}")
    print(f"→ Load it via:\n\n"
        f"from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f"model = AutoModelForCausalLM.from_pretrained('{OUT_DIR}')\n"
        f"tokenizer = AutoTokenizer.from_pretrained('{OUT_DIR}')\n")
    return Path(OUT_DIR)

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
        print(f"[validate][error] HF Load failed: {e} ")
        raise RuntimeError("HF Load failed")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Deepspeed checkpoint shards to a HuggingFace safetensors model.")
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, help="Path to the checkpoint directory, containing the trainer_state.pt file"
    )
    ap.add_argument("--out-dir", type=str, default=None, help="Output for HF model folder")
    ap.add_argument("--validate-load", action="store_true", help="Try loading with the Transformers Module after saving")
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    output_dir = Path(args.out_dir).resolve()
    out_path = main(ckpt_dir, output_dir)
    if args.validate_load:
        validate_load(out_path)



    