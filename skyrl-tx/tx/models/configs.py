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
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.loss_chunk_size = loss_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.mhc_expansion_rate = mhc_expansion_rate

    def get_num_experts(self):
        # Most models expose experts at top-level config.
        experts = getattr(self, "num_experts", None) or getattr(
            self, "n_routed_experts", None
        )
        if experts is not None:
            return experts

        # VL-MoE stores expert config under text_config (object or dict).
        text_config = getattr(self, "text_config", None)
        if isinstance(text_config, dict):
            return text_config.get("num_experts") or text_config.get("n_routed_experts")
        if text_config is not None:
            return getattr(text_config, "num_experts", None) or getattr(
                text_config, "n_routed_experts", None
            )
        return None


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
DeepseekV3Config = ModelConfig
Qwen3VLMoeConfig = ModelConfig
Qwen3VLModelConfig = ModelConfig
