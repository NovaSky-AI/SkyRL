"""Beam (``model_type="beam"``) configuration for the trainer side.

transformers ships ``BeamConfig`` from 5.22.0 (``transformers.models.beam``), but SkyRL pins
``transformers<=5.16.1`` (vLLM 0.33's cap). This vendors that config: same fields and defaults
as transformers 5.22's ``BeamConfig``, written as an ``__init__``-style ``PretrainedConfig`` so it
loads on the pinned transformers.

``rope_parameters`` stays keyed by layer type, with ``None`` for the NoPE (full-attention) layer
type, as in the checkpoint: Megatron-Bridge's ``BeamBridge`` reads one entry per layer type to set
``no_rope_freq`` and ``rotary_base``. vLLM's own ``BeamConfig`` flattens it to a single entry, which
is why the vLLM class cannot be registered here instead (vLLM keeps using its class through its
own config registry, whatever ``AutoConfig`` resolves to). The pinned transformers' RoPE
validation cannot read a ``None`` entry, so the keyed dict is flattened for ``super().__init__``
and restored afterwards.

Remove this module (and ``register_beam_config``) once the transformers pin reaches 5.22.
"""

from typing import Any, Optional

from transformers import PretrainedConfig

LAYER_TYPES = ("sliding_attention", "full_attention")
MLP_LAYER_TYPES = ("dense", "sparse")


def _flatten_rope_parameters(rope_parameters: dict) -> dict:
    """The single RoPE entry of a layer-type-keyed ``rope_parameters`` (``{}`` if every type is NoPE)."""
    rotary = [p for p in rope_parameters.values() if p is not None]
    if any(p != rotary[0] for p in rotary[1:]):
        raise ValueError(f"Beam requires identical RoPE parameters across layer types, got {rope_parameters!r}.")
    return dict(rotary[0]) if rotary else {}


class BeamConfig(PretrainedConfig):
    """Configuration for Reflection's Beam decoder-only MoE causal LM (``BeamForCausalLM``).

    Sliding-window layers use RoPE; full-attention layers are NoPE with SSMax query scaling.
    ``mlp_layer_types`` marks each layer's feed-forward as ``"dense"`` or ``"sparse"`` (grouped
    sigmoid ``noaux_tc`` MoE with an fp32 router). Defaults match transformers 5.22's
    ``BeamConfig`` (``reflection/Beam-1.0-501B-A23B``).
    """

    model_type = "beam"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 5120,
        intermediate_size: int = 20480,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        sliding_window: Optional[int] = 512,
        attention_dropout: float = 0.0,
        moe_intermediate_size: int = 2560,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        layer_types: Optional[list[str]] = None,
        mlp_layer_types: Optional[list[str]] = None,
        n_group: Optional[int] = 8,
        topk_group: Optional[int] = 4,
        routed_scaling_factor: float = 2.5,
        final_logit_softcapping: Optional[float] = 30.0,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int | list[int]] = None,
        **kwargs: Any,
    ):
        if layer_types is None:
            # Every sixth layer runs full attention, and so does the last one.
            layer_types = ["sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(num_hidden_layers)]
            layer_types[-1] = "full_attention"
        if len(layer_types) != num_hidden_layers or any(t not in LAYER_TYPES for t in layer_types):
            raise ValueError(f"layer_types must hold one of {LAYER_TYPES} per layer, got {layer_types!r}.")
        if mlp_layer_types is None:
            num_dense_layers = min(3, num_hidden_layers)
            mlp_layer_types = ["dense"] * num_dense_layers + ["sparse"] * (num_hidden_layers - num_dense_layers)
        if len(mlp_layer_types) != num_hidden_layers or any(t not in MLP_LAYER_TYPES for t in mlp_layer_types):
            raise ValueError(f"mlp_layer_types must hold one of {MLP_LAYER_TYPES} per layer, got {mlp_layer_types!r}.")
        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{num_attention_heads} % {num_key_value_heads} != 0"
            )
        if rope_parameters is None:
            rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": None,
            }
        keyed_rope_parameters = None
        if rope_parameters and set(rope_parameters) <= set(LAYER_TYPES):
            keyed_rope_parameters = {k: (dict(v) if v is not None else None) for k, v in rope_parameters.items()}
            rope_parameters = _flatten_rope_parameters(rope_parameters)
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["BeamForCausalLM"]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.layer_types = list(layer_types)
        self.mlp_layer_types = list(mlp_layer_types)
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.final_logit_softcapping = final_logit_softcapping

        super().__init__(
            rope_parameters=rope_parameters or None,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        if keyed_rope_parameters is not None:
            self.rope_parameters = keyed_rope_parameters


def register_beam_config() -> None:
    """Register :class:`BeamConfig` with ``AutoConfig`` so ``model_type="beam"`` resolves. Idempotent."""
    from transformers import AutoConfig

    AutoConfig.register("beam", BeamConfig, exist_ok=True)
