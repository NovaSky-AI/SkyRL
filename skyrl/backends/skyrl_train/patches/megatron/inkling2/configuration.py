"""Hugging Face config classes for Inkling2 (``model_type: inkling2``).

The release ships no ``transformers`` implementation, so these classes give ``AutoConfig`` (and
through it Megatron-Bridge) a typed view of ``config.json``. Field defaults are the released
checkpoint's values; fields the checkpoint does not set but the SGLang fork fixes
(``sglang/tml/config.py``) are recorded as constants on the text config.

Pre-release checkpoints used ``model_type: inkling2_mm_model`` and the key
``n_layers_per_attnres_block``; both are still read (:data:`LEGACY_MODEL_TYPE`). The vLLM side
accepts only the release names: :func:`release_config` converts such a checkpoint's
``config.json`` (``examples/train/megatron/dequantize_inkling2.py`` writes release names).
"""

from transformers import AutoConfig, PretrainedConfig

MODEL_TYPE = "inkling2"
LEGACY_MODEL_TYPE = "inkling2_mm_model"
TEXT_MODEL_TYPE = "inkling2_model"
# Renamed in the release config.
LEGACY_KEYS = {"n_layers_per_attnres_block": "n_layers_per_attn_res_block"}


class Inkling2TextConfig(PretrainedConfig):
    model_type = TEXT_MODEL_TYPE

    # Fixed by the SGLang fork for Inkling2 (``tml/config.py``); not read from config.json.
    use_embed_norm = True
    gate_activation = "sigmoid"
    w13_interleaved = True

    def __init__(
        self,
        vocab_size=201024,
        unpadded_vocab_size=200058,
        hidden_size=4608,
        num_hidden_layers=48,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        swa_num_attention_heads=48,
        swa_num_key_value_heads=8,
        swa_head_dim=128,
        sliding_window_size=1024,
        local_layer_ids=None,
        d_rel=16,
        rel_extent=1024,
        q_bias=False,
        o_bias=False,
        log_scaling_n_floor=128000,
        log_scaling_alpha=0.1,
        rms_norm_eps=1e-6,
        pre_norm_pre_affine=False,
        final_norm_pre_affine=False,
        use_value_norm_no_gamma=True,
        qk_l2_clip_threshold=22.627416997969522,
        rel_l2_clip_threshold=8.0,
        sconv_kernel_size=4,
        final_logit_softcapping=30.0,
        n_layers_per_attn_res_block=6,
        attn_res_normalize_values=True,
        n_routed_experts=384,
        num_experts_per_tok=8,
        n_shared_experts=0,
        intermediate_size=2304,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
        route_scale=8.0,
        use_gate_bias=True,
        model_max_length=1048576,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if "n_layers_per_attnres_block" in kwargs:
            n_layers_per_attn_res_block = kwargs.pop("n_layers_per_attnres_block")
        if "max_position_embeddings" in kwargs and model_max_length == 1048576:
            model_max_length = kwargs["max_position_embeddings"]
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        if local_layer_ids is None:
            local_layer_ids = [i for i in range(num_hidden_layers) if i % 6 != 5]
        self.local_layer_ids = list(local_layer_ids)
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.rms_norm_eps = rms_norm_eps
        self.pre_norm_pre_affine = pre_norm_pre_affine
        self.final_norm_pre_affine = final_norm_pre_affine
        self.use_value_norm_no_gamma = use_value_norm_no_gamma
        self.qk_l2_clip_threshold = qk_l2_clip_threshold
        self.rel_l2_clip_threshold = rel_l2_clip_threshold
        self.sconv_kernel_size = sconv_kernel_size
        self.final_logit_softcapping = final_logit_softcapping
        self.n_layers_per_attn_res_block = n_layers_per_attn_res_block
        self.attn_res_normalize_values = attn_res_normalize_values
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.intermediate_size = intermediate_size
        self.mlp_gate_softcap = mlp_gate_softcap
        self.mlp_up_softcap = mlp_up_softcap
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.model_max_length = model_max_length
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_local_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids


def _set_sub_configs(config, text_config, vision_config, mtp_config) -> None:
    if isinstance(text_config, dict) or text_config is None:
        text_config = Inkling2TextConfig(**(text_config or {}))
    config.text_config = text_config
    config.vision_config = vision_config
    config.mtp_config = mtp_config


# The release and pre-release top-level configs are two classes that each define ``__init__``:
# Transformers applies a config class's own constructor specially, and one that a class inherits
# (from a base or a mixin) leaves ``text_config`` a dict.


class Inkling2Config(PretrainedConfig):
    """Top-level config. ``vision_config`` and ``mtp_config`` are kept as plain dicts: only the
    language model is ported."""

    model_type = MODEL_TYPE
    sub_configs = {"text_config": Inkling2TextConfig}

    def __init__(self, text_config=None, vision_config=None, mtp_config=None, **kwargs):
        _set_sub_configs(self, text_config, vision_config, mtp_config)
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config


class Inkling2LegacyConfig(PretrainedConfig):
    """A pre-release checkpoint (``model_type: inkling2_mm_model``); same fields."""

    model_type = LEGACY_MODEL_TYPE
    sub_configs = {"text_config": Inkling2TextConfig}

    def __init__(self, text_config=None, vision_config=None, mtp_config=None, **kwargs):
        _set_sub_configs(self, text_config, vision_config, mtp_config)
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs):
        return self.text_config


def register() -> None:
    """Register the config classes with ``AutoConfig`` (idempotent)."""
    for model_type, cls in (
        (MODEL_TYPE, Inkling2Config),
        (LEGACY_MODEL_TYPE, Inkling2LegacyConfig),
        (TEXT_MODEL_TYPE, Inkling2TextConfig),
    ):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            pass  # already registered


def release_config(config: dict) -> dict:
    """A ``config.json`` dict with the release's names: ``model_type: inkling2``, the text
    config's ``inkling2_model`` and ``n_layers_per_attn_res_block``. Values are kept."""
    config = dict(config)
    if config.get("model_type") == LEGACY_MODEL_TYPE:
        config["model_type"] = MODEL_TYPE
    text = dict(config.get("text_config") or {})
    for old, new in LEGACY_KEYS.items():
        if old in text:
            text[new] = text.pop(old)
    text["model_type"] = TEXT_MODEL_TYPE
    config["text_config"] = text
    return config
