"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig


class Qwen3Config(PretrainedConfig):
    """Qwen3 configuration for tx.

    Wraps a HuggingFace PretrainedConfig with additional parameters
    for Multi-LoRA training and tensor parallelism.

    Args:
        config: A HuggingFace PretrainedConfig object (e.g., from Qwen3Config.from_pretrained())
        max_lora_adapters: Maximum number of concurrent LoRA adapters
        max_lora_rank: Maximum rank for LoRA adapters
        shard_attention_heads: Whether to shard attention across tensor parallel devices
        train_attn: Whether to train attention layers with LoRA (default: True)
        train_mlp: Whether to train MLP layers with LoRA (default: True)
        train_unembed: Whether to train unembedding/LM head layer with LoRA (default: False)
    """

    # Type hints for LoRA attributes
    max_lora_adapters: int
    max_lora_rank: int
    shard_attention_heads: bool
    train_attn: bool
    train_mlp: bool
    train_unembed: bool

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        train_attn: bool = True,
        train_mlp: bool = True,
        train_unembed: bool = False
    ):
        # Copy all attributes from the base config
        super().__init__(**config.to_dict())

        # Add LoRA-specific parameters
        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.train_attn = train_attn
        self.train_mlp = train_mlp
        self.train_unembed = train_unembed
