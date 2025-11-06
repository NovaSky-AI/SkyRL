from tx.models.layers import RMSNorm, SwiGLUMLP, apply_rope
from tx.models.llama3 import Llama3ForCausalLM
from tx.models.mnist import Mnist
from tx.models.outputs import CausalLMOutput, ModelOutput
from tx.models.qwen3 import Qwen3ForCausalLM

# Aliases for HuggingFace architecture names
Qwen3MoeForCausalLM = Qwen3ForCausalLM
LlamaForCausalLM = Llama3ForCausalLM

__all__ = [
    # Models
    Llama3ForCausalLM,
    LlamaForCausalLM,  # HuggingFace alias
    Mnist,
    Qwen3ForCausalLM,
    Qwen3MoeForCausalLM,
    # Outputs
    CausalLMOutput,
    ModelOutput,
    # Shared layers
    RMSNorm,
    SwiGLUMLP,
    apply_rope,
]
