"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig

from tx.models.qwen3_vl_configs import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig


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


class Qwen3VLModelConfig(ModelConfig):
    """Qwen3-VL configuration with LoRA support.

    Wraps Qwen3VLConfig (or a compatible PretrainedConfig from HuggingFace)
    and adds LoRA parameters. Ensures text_config and vision_config are
    proper config objects for the model to use.

    Use with base models like "Qwen/Qwen3-VL-4B-Instruct".
    """

    def __init__(
        self,
        config: PretrainedConfig | Qwen3VLConfig,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        loss_chunk_size: int = 0,
        gradient_checkpointing: bool = False,
    ):
        # Build base dict, ensuring nested configs are proper objects
        config_dict = config.to_dict()

        # Ensure text_config and vision_config are proper config objects
        # (they may be dicts when loaded from JSON)
        if "text_config" in config_dict:
            tc = config_dict["text_config"]
            if isinstance(tc, dict):
                config_dict["text_config"] = Qwen3VLTextConfig(**tc)
        if "vision_config" in config_dict:
            vc = config_dict["vision_config"]
            if isinstance(vc, dict):
                config_dict["vision_config"] = Qwen3VLVisionConfig(**vc)

        super(ModelConfig, self).__init__(**config_dict)

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.loss_chunk_size = loss_chunk_size
        self.gradient_checkpointing = gradient_checkpointing

    def get_num_experts(self):
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            return getattr(text_config, "num_experts", None) or getattr(
                text_config, "n_routed_experts", None
            )
        return None


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
DeepseekV3Config = ModelConfig
