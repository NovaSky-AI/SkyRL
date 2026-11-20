"""Hugging Face config classes for Inkling2 (``model_type: inkling2``, and ``inkling2_loop`` for looped checkpoints).

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
# Looped checkpoints (``text_config.loop_config``) have their own model type; plain ones have no loop_config.
LOOP_MODEL_TYPE = "inkling2_loop"
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
        loop_config=None,
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
        self.loop_config = validate_loop_config(
            loop_config, num_hidden_layers, n_layers_per_attn_res_block, attn_res_normalize_values
        )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def is_local_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.local_layer_ids

    @property
    def loop_core_layers(self) -> range:
        """The looped layers (empty without ``loop_config``)."""
        return loop_core_layers(self.loop_config)

    @property
    def num_layer_visits(self) -> int:
        """Layer runs per forward pass: each core layer runs ``num_passes`` times."""
        return num_layer_visits(self.loop_config, self.num_hidden_layers)


LOOP_CONFIG_FIELDS = ("core_start", "core_end", "num_passes", "gate_rank")
# Whether the checkpoint was trained to run with other pass counts (a serving option of variable-loop checkpoints).
# Training here always runs the checkpoint's ``num_passes``, so both values train the same way.
LOOP_CONFIG_FLAGS = ("is_varloop",)


def validate_loop_config(loop_config, num_layers: int, block_size: int, normalize_values: bool):
    """``loop_config`` (``{core_start, core_end, num_passes, gate_rank, is_varloop}`` or None) checked as the SGLang
    fork does (``tml/config.py``): layers ``[core_start, core_end)`` run ``num_passes`` times with shared weights, and
    the core starts and ends on attn-res block boundaries."""
    if loop_config is None:
        return None
    loop_config = dict(loop_config)
    if (
        set(loop_config) != set(LOOP_CONFIG_FIELDS) | set(LOOP_CONFIG_FLAGS)
        or any(type(loop_config[k]) is not int for k in LOOP_CONFIG_FIELDS)
        or any(type(loop_config[k]) is not bool for k in LOOP_CONFIG_FLAGS)
    ):
        raise ValueError(
            f"loop_config requires integer fields {sorted(LOOP_CONFIG_FIELDS)} and boolean "
            f"{sorted(LOOP_CONFIG_FLAGS)}, got {loop_config} (configs from before is_varloop need re-downloading)"
        )
    if not normalize_values:
        raise ValueError("Loop K/V messages require attn_res_normalize_values=True")
    start, end = loop_config["core_start"], loop_config["core_end"]
    if not 0 <= start < end <= num_layers:
        raise ValueError(f"Loop core [{start}, {end}) must be within the {num_layers} layers")
    if start % block_size or end % block_size:
        raise ValueError(f"Loop core boundaries must align with attention residual blocks of {block_size}")
    if loop_config["num_passes"] < 2 or loop_config["gate_rank"] <= 0:
        raise ValueError("Looping requires at least two passes and a positive gate rank")
    return loop_config


def loop_core_layers(loop_config) -> range:
    return range(0) if loop_config is None else range(loop_config["core_start"], loop_config["core_end"])


def num_layer_visits(loop_config, num_layers: int) -> int:
    if loop_config is None:
        return num_layers
    return num_layers + (loop_config["num_passes"] - 1) * len(loop_core_layers(loop_config))


def layer_run_order(loop_config, num_layers: int):
    """``(layer_idx, pass_idx)`` of every layer run, in execution order: prelude, core x passes, coda."""
    core = loop_core_layers(loop_config)
    if not core:
        return [(i, 0) for i in range(num_layers)]
    runs = [(i, 0) for i in range(core.start)]
    for p in range(loop_config["num_passes"]):
        runs += [(i, p) for i in core]
    return runs + [(i, 0) for i in range(core.stop, num_layers)]


def _set_sub_configs(config, text_config, vision_config, mtp_config) -> None:
    # A bare ``cls()`` (Transformers builds one to diff against when serializing) is not checked.
    given = text_config is not None
    if isinstance(text_config, dict) or text_config is None:
        text_config = Inkling2TextConfig(**(text_config or {}))
    if (
        given
        and config.model_type != LEGACY_MODEL_TYPE
        and (text_config.loop_config is not None) != (config.model_type == LOOP_MODEL_TYPE)
    ):
        raise ValueError(
            f"Looped Inkling2 checkpoints use model_type {LOOP_MODEL_TYPE} with a loop_config; "
            f"model_type {MODEL_TYPE} has none"
        )
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


class Inkling2LoopConfig(PretrainedConfig):
    """A looped checkpoint (``model_type: inkling2_loop``); same fields, ``text_config.loop_config`` required."""

    model_type = LOOP_MODEL_TYPE
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
        (LOOP_MODEL_TYPE, Inkling2LoopConfig),
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
    text = dict(config.get("text_config") or {})
    if config.get("model_type") == LEGACY_MODEL_TYPE:
        config["model_type"] = LOOP_MODEL_TYPE if text.get("loop_config") is not None else MODEL_TYPE
    for old, new in LEGACY_KEYS.items():
        if old in text:
            text[new] = text.pop(old)
    text["model_type"] = TEXT_MODEL_TYPE
    config["text_config"] = text
    return config
