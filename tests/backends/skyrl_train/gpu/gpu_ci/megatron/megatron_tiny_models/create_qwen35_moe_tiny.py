import json
import re
from pathlib import Path

import torch
from huggingface_hub import file_exists, hf_hub_download

# Megatron Bridge expects fused expert weights (gate_up_proj of shape
# [num_experts, 2*intermediate, hidden] and down_proj of shape
# [num_experts, hidden, intermediate]) but HF saves per-expert tensors.
# Rewrite the safetensors file to match the published checkpoint layout.
from safetensors.torch import load_file, save_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
    Qwen3_5MoeForConditionalGeneration,
)

source_model_id = "Qwen/Qwen3.5-35B-A3B"
save_folder = "/tmp/qwen35-moe-tiny-random"

processor = AutoProcessor.from_pretrained(source_model_id, trust_remote_code=True)
processor.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename="config.json", repo_type="model"), "r", encoding="utf-8") as f:
    config_json = json.load(f)

num_hidden_layers = 4
config_json["text_config"]["num_hidden_layers"] = num_hidden_layers
config_json["text_config"]["layer_types"] = config_json["text_config"]["layer_types"][:num_hidden_layers]
config_json["text_config"]["num_experts"] = 16
config_json["text_config"]["num_experts_per_tok"] = 4
config_json["text_config"]["mtp_num_hidden_layers"] = 0
config_json["vision_config"]["depth"] = 2

with open(f"{save_folder}/config.json", "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=2)

config = AutoConfig.from_pretrained(
    save_folder,
    trust_remote_code=True,
)
print(config)
torch.set_default_dtype(torch.bfloat16)
model = Qwen3_5MoeForConditionalGeneration(config)

layer_types = config_json["text_config"]["layer_types"]
with torch.no_grad():
    for i, lt in enumerate(layer_types):
        if lt == "linear_attention":
            attn = model.model.language_model.layers[i].linear_attn
            attn.A_log = torch.nn.Parameter(attn.A_log.float())
            attn.norm.float()

if any(lt == "linear_attention" for lt in layer_types):
    print(model.state_dict()["model.language_model.layers.0.linear_attn.A_log"].dtype)
    print(model.state_dict()["model.language_model.layers.0.linear_attn.norm.weight"].dtype)

torch.set_default_dtype(torch.float32)
if file_exists(filename="generation_config.json", repo_id=source_model_id, repo_type="model"):
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    model.generation_config.do_sample = True
    print(model.generation_config)
model = model.cpu()
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.1)
        print(name, p.shape)
model.save_pretrained(save_folder)

weights_path = Path(save_folder) / "model.safetensors"
state_dict = load_file(str(weights_path))
num_experts = config_json["text_config"]["num_experts"]

expert_re = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts)" r"\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

layers: dict = {}
keys_to_remove: list[str] = []
for key in state_dict:
    m = expert_re.match(key)
    if m:
        prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
        layers.setdefault(prefix, {}).setdefault(idx, {})[proj] = state_dict[key]
        keys_to_remove.append(key)

if keys_to_remove:
    new_state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
    for prefix, experts in layers.items():
        gate_up = torch.stack(
            [torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]], dim=0) for i in range(num_experts)],
            dim=0,
        )
        down = torch.stack([experts[i]["down_proj"] for i in range(num_experts)], dim=0)
        new_state_dict[f"{prefix}.gate_up_proj"] = gate_up
        new_state_dict[f"{prefix}.down_proj"] = down
    save_file(new_state_dict, str(weights_path))
    print(f"Fused expert weights for {len(layers)} MoE layers")

print("Upload with: huggingface-cli upload <org_name>/qwen35-moe-tiny-random " + save_folder)
