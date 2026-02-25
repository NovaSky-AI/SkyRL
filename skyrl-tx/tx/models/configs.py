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
        gradient_checkpointing: Recompute activations during backward to save memory
        mhc_expansion_rate: mHC expansion rate. Connectors are trainable when this is > 1.
    """

    # Type hints for config attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool
    loss_chunk_size: int
    gradient_checkpointing: bool
    mhc_expansion_rate: int

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        loss_chunk_size: int = 0,
        gradient_checkpointing: bool = False,
        mhc_expansion_rate: int = 1,
    ):
        # Copy attributes from the base config.
        # Some configs (especially multimodal wrappers) keep language-model fields
        # under nested dicts like "text_config". Merge these as fallbacks so
        # model code can consistently access top-level attributes.
        config_dict = config.to_dict()
        for nested_key in ("text_config", "language_config"):
            nested = config_dict.get(nested_key)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    config_dict.setdefault(key, value)

        super().__init__(**config_dict)

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.loss_chunk_size = loss_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.mhc_expansion_rate = mhc_expansion_rate

    def get_num_experts(self):
        return getattr(self, "num_experts", None) or getattr(self, "n_routed_experts", None)


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
Qwen3NextConfig = ModelConfig
DeepseekV3Config = ModelConfig
