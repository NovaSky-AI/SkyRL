"""Configuration classes for models with LoRA support."""

from transformers import Qwen3Config as HfQwen3Config


class Qwen3Config(HfQwen3Config):
    """Qwen3 configuration for tx.

    Extends the standard HuggingFace Qwen3Config with additional parameters
    for Multi-LoRA training and tensor parallelism.

    Args:
        max_lora_adapters: Maximum number of concurrent LoRA adapters (default: 32)
        max_lora_rank: Maximum rank for LoRA adapters (default: 32)
        shard_attention_heads: Whether to shard attention across tensor parallel devices (default: True)
        **kwargs: Additional arguments passed to Qwen3Config
    """

    # Type hints for LoRA attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool

    def __init__(
        self,
        max_lora_adapters: int = 32,
        max_lora_rank: int = 32,
        shard_attention_heads: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads

    @classmethod
    def from_pretrained_with_lora(
        cls,
        pretrained_model_name_or_path: str,
        max_lora_adapters: int = 32,
        max_lora_rank: int = 32,
        shard_attention_heads: bool = True,
        **kwargs
    ):
        """Load config from HuggingFace with LoRA parameters.

        Args:
            pretrained_model_name_or_path: Model ID from HuggingFace Hub or local path
            max_lora_adapters: Maximum number of concurrent LoRA adapters
            max_lora_rank: Maximum rank for LoRA adapters
            shard_attention_heads: Whether to shard attention heads
            **kwargs: Additional arguments for from_pretrained

        Returns:
            Qwen3Config instance
        """
        base_config = HfQwen3Config.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shard_attention_heads=shard_attention_heads,
            **base_config.to_dict()
        )
