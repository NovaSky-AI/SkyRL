"""Model provider for Inkling2 (``inkling2_mm_model``, language model only)."""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

import torch.nn.functional as F
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

from skyrl.backends.skyrl_train.patches.megatron.inkling2.layer_specs import (
    build_inkling2_layer_spec,
)
from skyrl.backends.skyrl_train.patches.megatron.inkling2.modules import (
    Inkling2EmbeddingNormMixin,
    Inkling2OutputLayerMixin,
    Inkling2RMSNorm,
    extend_instance,
)


@dataclass
class Inkling2ModelProvider(GPTModelProvider):
    """Megatron configuration and provider for the Inkling2 language model.

    The MoE is megatron-core's: sigmoid router whose expert bias only affects selection, top-k
    weights renormalized and scaled by ``moe_router_topk_scaling_factor``, and SiTU-GLU experts
    (``4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)`` via ``activation_func_tanh_clamp_scale*``).
    The expert bias is frozen (``moe_router_bias_update_rate = 0``). Attention, short
    convolutions and attn-res are ``inkling2.modules``; the fields below configure them.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = build_inkling2_layer_spec

    normalization: str = "RMSNorm"
    layernorm_epsilon: float = 1e-6
    position_embedding_type: str = "none"
    share_embeddings_and_output_weights: bool = False
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    gated_linear_unit: bool = True
    activation_func: Callable = F.silu
    activation_func_tanh_clamp_scale: float = 4.0
    activation_func_tanh_clamp_scale_linear: float = 25.0
    bias_activation_fusion: bool = True

    num_moe_experts: int = 384
    moe_router_topk: int = 8
    moe_ffn_hidden_size: int = 2304
    moe_layer_freq: int = 1
    # Layers differ in their params (layer 0 has no attn_query; it is [H] in local and [3, H] in global
    # layers, which alone have k/v_input_layernorm), so a distributed checkpoint keys each layer separately
    # (decoder.layers.<i>.*) instead of stacking every layer's param into one tensor. (Megatron's spelling.)
    hetereogenous_dist_checkpoint: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0.0
    moe_router_topk_scaling_factor: float = 8.0
    moe_router_dtype: str = "fp32"
    moe_router_load_balancing_type: str = "none"
    moe_aux_loss_coeff: float = 0.0
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    # TE's fused router (sigmoid scores, expert-bias top-k, renormalization and scaling in one kernel); ~3% of a
    # training step's model time on the 8-layer slice.
    moe_router_fusion: bool = True

    # Attention.
    local_layer_ids: List[int] = field(default_factory=lambda: [i for i in range(48) if i % 6 != 5])
    sliding_window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    log_scaling_n_floor: float = 8192
    log_scaling_alpha: float = 0.1
    # "flex" (compiled FlexAttention, BF16 output), "flex_fp32" (FP32 inputs and output, as the
    # serving kernel's output) or "dense" (FP32 materialized scores; tests only).
    inkling2_attention_backend: str = "flex"
    # Compile the FP32 elementwise pieces (attn-res, RMSNorm, L2 clips, short convolutions, attention gating, logit
    # soft-cap) into fused kernels (``modules.fused``); False runs them eagerly, op by op.
    inkling2_fuse_elementwise: bool = True
    # Under sequence parallelism, all-gather each attention input once and run its column-parallel projections
    # as local GEMMs, instead of one gather per projection (``Inkling2Attention._project``).
    inkling2_shared_sequence_gather: bool = True
    # Short convolutions and attn-res.
    sconv_kernel_size: int = 4
    n_layers_per_attn_res_block: int = 6
    attn_res_normalize_values: bool = True
    # Output.
    final_logit_softcapping: float = 30.0
    unpadded_vocab_size: int = 200058

    # LoRA ``target_modules="all-linear"``: the attention projections, added to the standard
    # names (``linear_proj``, ``linear_fc1``, ``linear_fc2``) that already cover the rest.
    lora_all_linear_extra_targets: Tuple[str, ...] = (
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_r",
        "linear_gate",
        "linear_sink",
    )

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if model.pre_process:
            model.embedding.embed_norm = Inkling2RMSNorm(self, self.hidden_size, self.layernorm_epsilon)
            extend_instance(model.embedding, Inkling2EmbeddingNormMixin)
        if model.post_process:
            extend_instance(model.output_layer, Inkling2OutputLayerMixin)
        return model
