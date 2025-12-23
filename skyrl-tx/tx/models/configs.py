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
        use_paged_attention: Whether to use paged attention for KV cache management
        page_size: Number of tokens per page in paged attention (default: 16)
    """

    # Type hints for LoRA attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool
    use_paged_attention: bool
    page_size: int

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        use_paged_attention: bool = False,
        page_size: int = 16,
    ):
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.use_paged_attention = use_paged_attention
        self.page_size = page_size


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
