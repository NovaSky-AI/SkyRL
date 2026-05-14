import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

source_model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
save_folder = "/tmp/nemotron3-moe-tiny-random"

tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename="config.json", repo_type="model"), "r", encoding="utf-8") as f:
    raw = f.read()

config_json = json.loads(re.sub(r"\bInfinity\b", "1e30", raw))

# Full hybrid_override_pattern (52 layers):
#   MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
# M = Mamba, E = MoE, * = Attention
# Keep only the first repeating block: MEMEM*E (7 layers)
config_json["hybrid_override_pattern"] = "MEMEM*E"
config_json["num_hidden_layers"] = 7
config_json["n_routed_experts"] = 16
config_json["num_experts_per_tok"] = 4
config_json.pop("auto_map", None)

with open(f"{save_folder}/config.json", "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=2)

config = AutoConfig.from_pretrained(save_folder)
print(config)

torch.set_default_dtype(torch.bfloat16)
model = AutoModelForCausalLM.from_config(config)
torch.set_default_dtype(torch.float32)

try:
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    model.generation_config.do_sample = True
    print(model.generation_config)
except OSError:
    print("no generation_config.json available; using model default")

model = model.cpu()
set_seed(42)
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.1)
        print(name, p.shape)

# Scale lm_head ~7x so the random model produces confident pre-sync logprobs
# (~-2 to -3 nats instead of uniform -log(vocab) ≈ -11.8). The round-trip
# correctness check in test_logprobs_matching_roundtrip uses a teacher-forced
# post-sync forward over the SAME pre-sync tokens; a corrupted post-sync model
# (e.g. the vLLM conv_weights aliasing regression) drives the per-token logprob
# of those tokens back toward -11.8, giving a large pre/post diff. Without
# this scaling the pre-sync model is already near uniform, so corruption is
# invisible. The clean-run pre/post diff stays ~0.16 (parity only).
LM_HEAD_SCALE = 7.0
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.startswith("lm_head."):
            p.mul_(LM_HEAD_SCALE)
            print(f"scaled {name} by {LM_HEAD_SCALE}")

for i, block_type in enumerate(config.layers_block_type):
    if block_type == "moe":
        model.model.layers[i].mixer.gate.e_score_correction_bias = torch.rand_like(
            model.model.layers[i].mixer.gate.e_score_correction_bias
        ).float()

model.save_pretrained(save_folder)

# vLLM's NemotronH mapper converts "embeddings" -> "embed_tokens" but NOT
# "embedding" (singular).  The native transformers model uses "embeddings"
# (plural) which maps correctly, but NVIDIA's custom HF code uses
# "embedding" (singular).  Normalise the key so the checkpoint always works
# regardless of which code path created the model.
weights_path = Path(save_folder) / "model.safetensors"
state_dict = load_file(str(weights_path))
renamed = False
for old_key in list(state_dict):
    if ".embedding." in old_key or old_key.startswith("model.embedding."):
        new_key = old_key.replace(".embedding.", ".embeddings.", 1)
        if new_key != old_key:
            state_dict[new_key] = state_dict.pop(old_key)
            print(f"Renamed: {old_key} -> {new_key}")
            renamed = True
if renamed:
    save_file(state_dict, str(weights_path))

print(f"\nModel saved to {save_folder}")
print("Upload with: huggingface-cli upload <org_name>/nemotron3-moe-tiny-random " + save_folder)
