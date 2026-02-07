"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """Configuration for tx models with LoRA support.

    Wraps a HuggingFace PretrainedConfig with additional parameters
    for Multi-LoRA training and tensor parallelism.

    Args:
        config: A HuggingFace PretrainedConfig object (e.g., from AutoConfig.from_pretrained())
        max_lora_adapters: Maximum number of concurrent LoRA adapters
        max_lora_rank: Maximum rank for LoRA adapters
        shard_attention_heads: Whether to shard attention across tensor parallel devices
        loss_chunk_size: Chunk size for cross-entropy loss computation (0 = no chunking)
        gradient_checkpointing: Whether to use gradient checkpointing for chunked loss
    """

    # Type hints for config attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool
    loss_chunk_size: int
    gradient_checkpointing: bool

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        loss_chunk_size: int = 0,
        gradient_checkpointing: bool = False,
    ):
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.loss_chunk_size = loss_chunk_size
        self.gradient_checkpointing = gradient_checkpointing

    def get_num_experts(self):
        return getattr(self, "num_experts", None) or getattr(self, "n_routed_experts", None)


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
DeepseekV3Config = ModelConfig
