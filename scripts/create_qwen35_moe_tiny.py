import json
from copy import deepcopy
from pathlib import Path

import torch
from huggingface_hub import file_exists, hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Qwen3_5MoeForConditionalGeneration,
    set_seed,
)

source_model_id = "Qwen/Qwen3.5-397B-A17B"
save_folder = "/tmp/erictang000/qwen35-moe-tiny-random"

processor = AutoProcessor.from_pretrained(source_model_id, trust_remote_code=True)
processor.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename='config.json', repo_type='model'), 'r', encoding='utf-8') as f:
    config_json = json.load(f)

# Dimensions are chosen so that per-GPU values with TP=2 are >= 64,
# which is the minimum safe size for vLLM's fused MoE Triton kernels
# (BLOCK_SIZE_K=64). The original tiny model used hidden_size=8 which
# resulted in hidden_size_per_gpu=4, far below kernel block sizes.
config_json['text_config'].update({
    'head_dim': 32,
    'hidden_size': 128,            # 64 per GPU with TP=2
    "layer_types": ['linear_attention'] * 3 + ['full_attention'],
    'intermediate_size': 256,      # 128 per GPU with TP=2
    'moe_intermediate_size': 128,  # 64 per GPU with TP=2
    'num_hidden_layers': 4,
    'num_attention_heads': 4,      # 2 per GPU with TP=2
    'num_key_value_heads': 2,      # 1 per GPU with TP=2
    'num_experts': 128,
    'num_experts_per_tok': 10,
    'shared_expert_intermediate_size': 256,  # 128 per GPU with TP=2
    "linear_key_head_dim": 32,
    "linear_num_key_heads": 4,     # 2 per GPU with TP=2
    "linear_num_value_heads": 4,   # 2 per GPU with TP=2
    "linear_value_head_dim": 32,
})
# partial_rotary_factor=0.25 from source → rotary_dim = 0.25 * 32 = 8
# mrope_section sum * 2 = (1+1+2)*2 = 8 = rotary_dim ✓
config_json['text_config']['rope_parameters']['mrope_section'] = [1, 1, 2]
config_json["tie_word_embeddings"] = False
config_json['vision_config'].update(
    {
        'hidden_size': 64,
        'intermediate_size': 128,
        'num_heads': 2,
        'out_hidden_size': 128,    # must match language model hidden_size
        'depth': 2,
    }
)
with open(f"{save_folder}/config.json", "w", encoding='utf-8') as f:
    json.dump(config_json, f, indent=2)

config = AutoConfig.from_pretrained(
    save_folder,
    trust_remote_code=True,
)
print(config)
torch.set_default_dtype(torch.bfloat16)
model = Qwen3_5MoeForConditionalGeneration(config)
with torch.no_grad():
    for i in range(3):
        attn = model.model.language_model.layers[i].linear_attn
        attn.A_log = torch.nn.Parameter(attn.A_log.float())
        attn.norm.float()

print(model.state_dict()['model.language_model.layers.0.linear_attn.A_log'].dtype)
print(model.state_dict()['model.language_model.layers.0.linear_attn.norm.weight'].dtype)

model.mtp = torch.nn.ModuleDict({
    "pre_fc_norm_embedding": torch.nn.RMSNorm(config.text_config.hidden_size),
    "fc": torch.nn.Linear(config.text_config.hidden_size * 2, config.text_config.hidden_size, bias=False),
    "layers": torch.nn.ModuleList([deepcopy(model.model.language_model.layers[3])]),
    "norm": torch.nn.RMSNorm(config.text_config.hidden_size),
    "pre_fc_norm_hidden": torch.nn.RMSNorm(config.text_config.hidden_size),
})
torch.set_default_dtype(torch.float32)
if file_exists(filename="generation_config.json", repo_id=source_model_id, repo_type='model'):
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id, trust_remote_code=True,
    )
    model.generation_config.do_sample = True
    print(model.generation_config)
model = model.cpu()
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.1)
        print(name, p.shape)
model.save_pretrained(save_folder)
