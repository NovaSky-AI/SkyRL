"""Configuration classes for models with LoRA support."""

from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    """Configuration for skyrl models with LoRA support.

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
        config: PretrainedConfig | dict,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shard_attention_heads: bool,
        loss_chunk_size: int = 0,
        gradient_checkpointing: bool = False,
        mhc_expansion_rate: int = 1,
    ):
        super().__init__(**(config if isinstance(config, dict) else config.__dict__))

        # In transformers v5, rope_parameters may not contain rope_theta
        # even when it exists as a top-level config attribute (e.g. DeepSeek v3).
        # Inject it so model code can always use config.rope_parameters["rope_theta"].
        rope_params = getattr(self, "rope_parameters", None) or {}
        if "rope_theta" not in rope_params:
            rope_theta = getattr(self, "rope_theta", None)
            if rope_theta is not None:
                rope_params["rope_theta"] = rope_theta
        if rope_params:
            self.rope_parameters = rope_params

        self.max_lora_adapters = max_lora_adapters
        self.max_lora_rank = max_lora_rank
        self.shard_attention_heads = shard_attention_heads
        self.loss_chunk_size = loss_chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.mhc_expansion_rate = mhc_expansion_rate

    def get_config(self) -> PretrainedConfig:
        """Return `text_config` when present, otherwise return this config."""
        return self.get_text_config() if hasattr(self, "text_config") else self

    def get_text_config(self, decoder=None, encoder=None) -> "ModelConfig":
        """Return a wrapped config built from `self.text_config`."""
        text_cfg = super().get_text_config(decoder=decoder, encoder=encoder)
        if text_cfg is self or isinstance(text_cfg, ModelConfig):
            return text_cfg
        return type(self)(
            text_cfg,
            max_lora_adapters=self.max_lora_adapters,
            max_lora_rank=self.max_lora_rank,
            shard_attention_heads=self.shard_attention_heads,
            loss_chunk_size=self.loss_chunk_size,
            gradient_checkpointing=self.gradient_checkpointing,
            mhc_expansion_rate=self.mhc_expansion_rate,
        )

    def get_num_experts(self):
        # TODO: Change this if there can be different numbers of experts in text_config and vision_config
        config = self.get_config()
        return getattr(config, "num_experts", None) or getattr(config, "n_routed_experts", None)


# Model-specific aliases for clarity and backwards compatibility
Llama3Config = ModelConfig
Qwen3Config = ModelConfig
Qwen3_5Config = ModelConfig
Qwen3_5TextConfig = ModelConfig
DeepseekV3Config = ModelConfig
