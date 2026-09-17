"""Hugging Face <-> Megatron bridge for GLM-5.3-Flash (``Glm5NextForConditionalGeneration``).

Only the language model is bridged (``model.language_model.*`` and ``lm_head``); the vision
tower of the unified VL checkpoint has no Megatron counterpart and is left untouched.
"""

from typing import Any, Dict, Optional

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.core.models.gpt.gpt_model import GPTModel
from torch import nn

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.provider import (
    Glm5NextModelProvider,
)

# Sinkhorn epsilon hard-coded in megatron-core's HyperConnectionModule (``_MHC_SINKHORN_EPS``).
_MCORE_MHC_EPS = 1e-6


class HyperConnectionScaleMapping(MegatronParamMapping):
    """Map the HF ``hc_*_scale`` tensor ([3] = [alpha_pre, alpha_post, alpha_res]) to Megatron's
    three scalar ``alpha_*`` parameters.

    Registered on ``alpha_pre``; export concatenates all three alphas from the owning module.
    :class:`HyperConnectionScaleSliceMapping` covers ``alpha_post`` / ``alpha_res`` on import.
    """

    def __init__(self, megatron_pre: str, megatron_post: str, megatron_res: str, hf_param: str):
        super().__init__(megatron_param=megatron_pre, hf_param=hf_param)
        self._megatron_post = megatron_post
        self._megatron_res = megatron_res

    @staticmethod
    def _resolve_single(pattern: str, captures) -> str:
        result = pattern
        for capture in captures:
            if "*" not in result:
                break
            result = result.replace("*", capture, 1)
        return result

    def resolve(self, captures):
        resolved_megatron, resolved_hf = self._resolve_names(captures)
        return HyperConnectionScaleMapping(
            megatron_pre=resolved_megatron,
            megatron_post=self._resolve_single(self._megatron_post, captures),
            megatron_res=self._resolve_single(self._megatron_res, captures),
            hf_param=resolved_hf,
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return hf_weights.to(megatron_module.alpha_pre.device)[0:1]

    def megatron_to_hf(
        self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        post = megatron_module.alpha_post.detach() if megatron_module is not None else None
        res = megatron_module.alpha_res.detach() if megatron_module is not None else None
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        post = self.broadcast_from_pp_rank(post, cache_key=str(self.hf_param) + "_post")
        res = self.broadcast_from_pp_rank(res, cache_key=str(self.hf_param) + "_res")
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)
        return {str(self.hf_param): torch.cat([megatron_weights.float(), post.float(), res.float()])}


class HyperConnectionScaleSliceMapping(MegatronParamMapping):
    """Import-only mapping of ``hc_*_scale[index]`` into ``alpha_post`` (1) or ``alpha_res`` (2).

    Export is a no-op because :class:`HyperConnectionScaleMapping` already emits the full tensor.
    """

    def __init__(self, megatron_param: str, hf_param: str, index: int):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._index = index
        self.allow_hf_name_mismatch = True

    def resolve(self, captures):
        resolved_megatron, resolved_hf = self._resolve_names(captures)
        return HyperConnectionScaleSliceMapping(resolved_megatron, resolved_hf, self._index)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        target = megatron_module.alpha_post if self._index == 1 else megatron_module.alpha_res
        return hf_weights.to(target.device)[self._index : self._index + 1]

    def megatron_to_hf(self, megatron_weights, megatron_module) -> Dict[str, torch.Tensor]:
        return {}


@MegatronModelBridge.register_bridge(
    source="Glm5NextForConditionalGeneration",
    target=GPTModel,
    provider=Glm5NextModelProvider,
    model_type="glm5_next",
)
class Glm5NextBridge(MegatronModelBridge):
    """Megatron Bridge for the GLM-5.3-Flash language model.

    Handles conversion between the HF ``Glm5NextForConditionalGeneration`` checkpoint (text
    fields under ``text_config``, weights under ``model.language_model.*``) and a Megatron-Core
    ``GPTModel`` built from :func:`~.layer_specs.build_glm5_next_layer_spec`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5.3-Flash")
        >>> provider = bridge.to_megatron_provider()
    """

    def _should_map_hf_config_field(self, hf_config: Any, hf_name: str, megatron_name: str, value: Any) -> bool:
        # ``head_dim`` is the RoPE width, which GLM-5.3-Flash pins to 0 (NoPE); it must not
        # become ``kv_channels``. MLA reads its head geometry from the ``*_head_dim`` fields.
        if hf_name == "head_dim":
            return False
        return super()._should_map_hf_config_field(hf_config, hf_name, megatron_name, value)

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        """Map the nested text configuration with the common config mappings."""
        return super().hf_config_to_provider_kwargs(getattr(hf_config, "text_config", hf_config))

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Glm5NextModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        provider = super().provider_bridge(hf_pretrained)

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = bool(getattr(hf_config, "tie_word_embeddings", False))
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = False
        # The gate/up clamp of GLM's SwiGLU lives on the unfused GLU path only.
        provider.activation_func_clamp_value = text_config.swiglu_limit
        provider.bias_activation_fusion = False
        provider.use_te_activation_func = False
        provider.mtp_num_layers = None

        # Attention pattern: KDA (1) vs DSA (0) per layer.
        layer_types = text_config.layer_types
        provider.linear_attention_freq = [1 if t == "linear_attention" else 0 for t in layer_types]
        provider.linear_num_key_heads = text_config.linear_num_heads
        provider.linear_num_value_heads = text_config.linear_num_heads
        provider.linear_key_head_dim = text_config.linear_head_dim
        provider.linear_value_head_dim = text_config.linear_head_dim
        provider.linear_conv_kernel_dim = text_config.linear_conv_kernel_dim
        provider.kda_gate_lower_bound = text_config.linear_lower_bound

        # NoPE MLA + DSA on the full-attention layers.
        provider.multi_latent_attention = True
        provider.qk_layernorm = True
        provider.rope_type = "rope"
        provider.rotary_scaling_factor = 1.0
        provider.mscale = 1.0
        provider.mscale_all_dim = 1.0
        provider.apply_rope_fusion = False
        provider.experimental_attention_variant = "dsa"
        provider.dsa_indexer_head_dim = text_config.index_head_dim
        provider.dsa_indexer_n_heads = text_config.index_n_heads
        provider.dsa_indexer_topk = text_config.index_topk
        provider.dsa_indexer_rotate_activation = False
        provider.dsa_indexer_scoring_relu = True
        provider.dsa_indexer_k_norm_epsilon = 1e-6
        provider.dsa_indexer_rope_interleaved = bool(getattr(text_config, "indexer_rope_interleave", False))
        provider.dsa_indexer_loss_coeff = 0.0
        provider.dsa_indexer_kpool = text_config.index_kpool
        provider.dsa_indexer_kpool_always_select_tail = text_config.index_kpool_always_select_tail
        indexer_types = [
            t for t, layer_type in zip(text_config.indexer_types, layer_types) if layer_type != "linear_attention"
        ]
        if any(t != "full" for t in indexer_types):
            raise NotImplementedError(
                "GLM-5.3-Flash cross-layer DSA index sharing (indexer_types containing 'shared') is not "
                "supported by the Megatron backend yet."
            )
        provider.dsa_indexer_topk_freq = 1
        provider.dsa_indexer_skip_topk_offset = 0

        # Sigmoid-scored MoE with a frozen routing-correction bias and one shared expert.
        provider.moe_layer_freq = [1 if t == "sparse" else 0 for t in text_config.mlp_layer_types]
        provider.moe_shared_expert_intermediate_size = text_config.moe_intermediate_size * text_config.n_shared_experts
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_pre_softmax = True
        provider.moe_router_dtype = "fp32"
        provider.moe_grouped_gemm = True
        if text_config.n_group in (None, 1):
            provider.moe_router_num_groups = None
            provider.moe_router_group_topk = None

        # Manifold-Constrained Hyper-Connections.
        provider.enable_mhc_connections = bool(getattr(text_config, "mhc", True))
        provider.mhc_num_residual_streams = text_config.hc_mult
        provider.mhc_sinkhorn_iterations = text_config.hc_sinkhorn_iters
        provider.mhc_norm_eps = text_config.rms_norm_eps
        provider.mhc_norm_eps_inside_sqrt = True
        if text_config.hc_eps != _MCORE_MHC_EPS:
            raise ValueError(f"hc_eps={text_config.hc_eps} differs from megatron-core's mHC epsilon {_MCORE_MHC_EPS}.")
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_prefix = "model.language_model"
        hf_layer = f"{hf_prefix}.layers.*"
        hf_attn = f"{hf_layer}.self_attn"
        megatron_layer = "decoder.layers.*"
        megatron_attn = f"{megatron_layer}.self_attention"

        # Standard megatron-core modules: AutoMapping infers the TP layout from the module type.
        auto_mappings = {
            "embedding.word_embeddings.weight": f"{hf_prefix}.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": f"{hf_prefix}.norm.weight",
            f"{megatron_layer}.input_layernorm.weight": f"{hf_layer}.input_layernorm.weight",
            # MoE layers keep a standalone pre-MLP norm; the dense TE MLP fuses it into linear_fc1.
            f"{megatron_layer}.pre_mlp_layernorm.weight": f"{hf_layer}.post_attention_layernorm.weight",
            f"{megatron_layer}.mlp.linear_fc1.layer_norm_weight": f"{hf_layer}.post_attention_layernorm.weight",
            # NoPE MLA (DSA layers).
            f"{megatron_attn}.linear_q_down_proj.weight": f"{hf_attn}.q_a_proj.weight",
            f"{megatron_attn}.linear_q_up_proj.weight": f"{hf_attn}.q_b_proj.weight",
            f"{megatron_attn}.q_layernorm.weight": f"{hf_attn}.q_a_layernorm.weight",
            f"{megatron_attn}.linear_kv_down_proj.weight": f"{hf_attn}.kv_a_proj_with_mqa.weight",
            f"{megatron_attn}.linear_kv_up_proj.weight": f"{hf_attn}.kv_b_proj.weight",
            f"{megatron_attn}.kv_layernorm.weight": f"{hf_attn}.kv_a_layernorm.weight",
            f"{megatron_attn}.linear_proj.weight": f"{hf_attn}.o_proj.weight",
            # DSA lightning indexer (the k-pool compression parameters have no Megatron counterpart).
            f"{megatron_attn}.core_attention.indexer.linear_wq_b.weight": f"{hf_attn}.indexer.wq_b.weight",
            f"{megatron_attn}.core_attention.indexer.linear_wk.weight": f"{hf_attn}.indexer.wk.weight",
            f"{megatron_attn}.core_attention.indexer.k_norm.weight": f"{hf_attn}.indexer.k_norm.weight",
            f"{megatron_attn}.core_attention.indexer.k_norm.bias": f"{hf_attn}.indexer.k_norm.bias",
            f"{megatron_attn}.core_attention.indexer.linear_weights_proj.weight": f"{hf_attn}.indexer.weights_proj.weight",
            # MoE router and down projections.
            f"{megatron_layer}.mlp.router.weight": f"{hf_layer}.mlp.gate.weight",
            f"{megatron_layer}.mlp.router.expert_bias": f"{hf_layer}.mlp.gate.e_score_correction_bias",
            f"{megatron_layer}.mlp.linear_fc2.weight": f"{hf_layer}.mlp.down_proj.weight",
            f"{megatron_layer}.mlp.shared_experts.linear_fc2.weight": f"{hf_layer}.mlp.shared_experts.down_proj.weight",
            f"{megatron_layer}.mlp.experts.linear_fc2.weight*": f"{hf_layer}.mlp.experts.*.down_proj.weight",
            f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc2.weight": f"{hf_layer}.mlp.experts.*.down_proj.weight",
        }
        mappings = [AutoMapping(megatron_param=m, hf_param=h) for m, h in auto_mappings.items()]

        # KDA layers: explicit TP layouts for the custom module.
        mappings += [
            ColumnParallelMapping(f"{megatron_attn}.{name}", f"{hf_attn}.{name}")
            for name in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "f_b_proj.weight",
                "g_b_proj.weight",
                "b_proj.weight",
                "A_log",
                "dt_bias",
            )
        ]
        mappings += [
            ReplicatedMapping(f"{megatron_attn}.{name}", f"{hf_attn}.{name}")
            for name in ("f_a_proj.weight", "g_a_proj.weight", "o_norm.weight")
        ]
        mappings.append(RowParallelMapping(f"{megatron_attn}.o_proj.weight", f"{hf_attn}.o_proj.weight"))

        # Gated MLPs: dense, shared expert, routed experts (per-expert HF layout).
        mappings += [
            GatedMLPMapping(
                f"{megatron_layer}.mlp.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.gate_proj.weight",
                up=f"{hf_layer}.mlp.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.shared_experts.gate_proj.weight",
                up=f"{hf_layer}.mlp.shared_experts.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.linear_fc1.weight*",
                gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
            ),
            GatedMLPMapping(
                f"{megatron_layer}.mlp.experts.local_experts.*.linear_fc1.weight",
                gate=f"{hf_layer}.mlp.experts.*.gate_proj.weight",
                up=f"{hf_layer}.mlp.experts.*.up_proj.weight",
            ),
        ]

        # Hyper-connections (replicated; the fp32 alphas are packed as one [3] tensor in HF).
        for site, hf_site in (("self_attention_hyper_connection", "hc_attn"), ("mlp_hyper_connection", "hc_ffn")):
            megatron_site = f"{megatron_layer}.{site}"
            mappings += [
                ReplicatedMapping(f"{megatron_site}.mapping_proj.weight", f"{hf_layer}.{hf_site}_fn"),
                ReplicatedMapping(f"{megatron_site}.bias", f"{hf_layer}.{hf_site}_base"),
                HyperConnectionScaleMapping(
                    megatron_pre=f"{megatron_site}.alpha_pre",
                    megatron_post=f"{megatron_site}.alpha_post",
                    megatron_res=f"{megatron_site}.alpha_res",
                    hf_param=f"{hf_layer}.{hf_site}_scale",
                ),
                HyperConnectionScaleSliceMapping(f"{megatron_site}.alpha_post", f"{hf_layer}.{hf_site}_scale", 1),
                HyperConnectionScaleSliceMapping(f"{megatron_site}.alpha_res", f"{hf_layer}.{hf_site}_scale", 2),
            ]

        return MegatronMappingRegistry(*mappings)
