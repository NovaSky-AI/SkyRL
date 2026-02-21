"""Qwen3-VL configuration classes.

Compatible with HuggingFace Qwen3-VL config structure for loading checkpoints.
Aligned with transformers.models.qwen3_vl.configuration_qwen3_vl.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class Qwen3VLVisionConfig(PretrainedConfig):
    """Vision encoder (ViT) configuration for Qwen3-VL."""

    model_type = "qwen3_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 3584,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: list[int] | None = None,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes or [8, 16, 24]


class Qwen3VLTextConfig(PreTrainedConfig):
    """Text backbone configuration for Qwen3-VL (same as Qwen3 LLM)."""

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"
    default_theta = 500000.0

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 22016,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 128000,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        super().__init__(
            ignore_keys_at_rope_validation={"mrope_section", "mrope_interleaved"},
            **kwargs,
        )


class Qwen3VLConfig(PretrainedConfig):
    """Top-level Qwen3-VL configuration with text and vision subconfigs."""

    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Qwen3VLTextConfig | dict[str, Any] | None = None,
        vision_config: Qwen3VLVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
        vision_start_token_id: int = 151652,
        vision_end_token_id: int = 151653,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Qwen3VLConfig", "Qwen3VLTextConfig", "Qwen3VLVisionConfig"]
