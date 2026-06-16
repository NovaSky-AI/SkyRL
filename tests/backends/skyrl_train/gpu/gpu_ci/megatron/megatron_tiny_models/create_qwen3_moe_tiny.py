import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)

source_model_id = "Qwen/Qwen3-235B-A22B"
save_folder = "/tmp/qwen3-moe-tiny-random"

tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_folder)

config = AutoConfig.from_pretrained(source_model_id, trust_remote_code=True)
config._name_or_path = source_model_id

# All per-GPU values with TP=2 must be >= 64 for vLLM's fused MoE Triton
# kernels (BLOCK_SIZE_K=64).
config.hidden_size = 128  # 64 per GPU with TP=2
config.intermediate_size = 256  # 128 per GPU with TP=2 (dense MLP)
config.moe_intermediate_size = 128  # 64 per GPU with TP=2
config.head_dim = 32
config.num_attention_heads = 4  # 2 per GPU with TP=2
config.num_key_value_heads = 2  # 1 per GPU with TP=2
config.num_hidden_layers = 2
config.decoder_sparse_step = 1  # all layers are MoE (matches Qwen3MoEBridge assumption)
config.num_experts = 8
config.num_experts_per_tok = 2
config.max_window_layers = 2
config.tie_word_embeddings = False

print(config)

model = AutoModelForCausalLM.from_config(
    config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.generation_config = GenerationConfig.from_pretrained(
    source_model_id,
    trust_remote_code=True,
)
model.generation_config.do_sample = True

set_seed(42)
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        torch.nn.init.normal_(p, 0, 0.5)
        print(name, p.shape)

model.save_pretrained(save_folder)
print(f"\nModel saved to {save_folder}")
print("Upload with: huggingface-cli upload <org_name>/qwen3-moe-tiny-random " + save_folder)
