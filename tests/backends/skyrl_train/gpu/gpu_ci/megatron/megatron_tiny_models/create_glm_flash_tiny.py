import json
from copy import deepcopy

import torch
import torch.nn as nn
from huggingface_hub import file_exists, hf_hub_download
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    set_seed,
)

source_model_id = "zai-org/GLM-4.7-Flash"
save_folder = "/tmp/glm-4.7-flash-tiny-random"

processor = AutoProcessor.from_pretrained(source_model_id, trust_remote_code=True)
processor.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename="config.json", repo_type="model"), "r", encoding="utf-8") as f:
    config_json = json.load(f)

# All per-GPU values with TP=2 must be >= 64 for vLLM's fused MoE Triton
# kernels (BLOCK_SIZE_K=64).
config_json.update(
    {
        "hidden_size": 128,  # >= 64 (K dim for MoE first matmul, not split by TP)
        "intermediate_size": 256,  # dense MLP / shared expert: 128 per GPU with TP=2
        "moe_intermediate_size": 128,  # routed experts: 64 per GPU with TP=2
        "num_attention_heads": 4,  # 2 per GPU with TP=2
        "num_key_value_heads": 4,  # same as num_attention_heads for MLA
        "kv_lora_rank": 256,  # MLA head_dim = kv_lora_rank + qk_rope_head_dim
        "q_lora_rank": 64,  # compressed Q dim (not split by TP)
        "qk_nope_head_dim": 64,  # per-head (not split by TP)
        "qk_rope_head_dim": 64,  # per-head; head_dim = 256 + 64 = 320 (supported by vLLM MLA)
        "v_head_dim": 64,  # per-head (not split by TP)
        "n_routed_experts": 8,  # reduced for tiny model
        "num_experts_per_tok": 2,  # reduced for tiny model
        "num_hidden_layers": 2,
        "first_k_dense_replace": 1,  # layer 0 = dense MLP, layer 1+ = MoE
        "tie_word_embeddings": False,
        "use_cache": True,
    }
)

with open(f"{save_folder}/config.json", "w", encoding="utf-8") as f:
    json.dump(config_json, f, indent=2)

config = AutoConfig.from_pretrained(save_folder, trust_remote_code=True)
print(config)

torch.set_default_dtype(torch.bfloat16)
model = AutoModelForCausalLM.from_config(config)
torch.set_default_dtype(torch.float32)

if file_exists(filename="generation_config.json", repo_id=source_model_id, repo_type="model"):
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    model.generation_config.do_sample = True
    print(model.generation_config)

model = model.cpu()
set_seed(42)
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.1)
        print(name, p.shape)

# MTP layer: manually append following the same structure as the reference
set_seed(42)
model.model.layers.append(
    nn.ModuleDict(
        dict(
            embed_tokens=deepcopy(model.model.embed_tokens),
            shared_head=nn.ModuleDict(
                dict(
                    norm=nn.RMSNorm(config.hidden_size),
                    head=deepcopy(model.model.embed_tokens),
                )
            ),
            eh_proj=nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False),
            enorm=nn.RMSNorm(config.hidden_size),
            hnorm=nn.RMSNorm(config.hidden_size),
            input_layernorm=nn.RMSNorm(config.hidden_size),
            post_attention_layernorm=nn.RMSNorm(config.hidden_size),
            self_attn=deepcopy(model.model.layers[1].self_attn),
            mlp=deepcopy(model.model.layers[1].mlp),
        )
    )
)

for i in range(1, len(model.model.layers)):
    model.model.layers[i].mlp.gate.e_score_correction_bias = torch.rand_like(
        model.model.layers[i].mlp.gate.e_score_correction_bias
    ).float()

model.save_pretrained(save_folder)
print(f"\nModel saved to {save_folder}")
print("Upload with: hf upload <org_name>/glm-4.7-flash-tiny-random " + save_folder)
