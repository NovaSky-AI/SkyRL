"""Megatron block spec for GLM-5.3-Flash: KDA or DSA attention, dense or MoE MLP, mHC residuals."""

import copy
from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_dsa_module_spec_for_backend,
    get_linear_attention_pattern,
    get_moe_layer_pattern,
)
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from skyrl.backends.skyrl_train.workers.megatron.glm5_next.dsa import (
    Glm5NextDSAttention,
)
from skyrl.backends.skyrl_train.workers.megatron.mcore_ext.kda import (
    KimiDeltaAttention,
    KimiDeltaAttentionSubmodules,
)
from skyrl.backends.skyrl_train.workers.megatron.mcore_ext.mhc_transformer_layer import (
    HyperConnectionTransformerLayer,
)


def get_kda_module_spec(backend: TESpecProvider) -> ModuleSpec:
    """Module spec for a KDA linear-attention layer."""
    column = backend.column_parallel_linear()
    return ModuleSpec(
        module=KimiDeltaAttention,
        submodules=KimiDeltaAttentionSubmodules(
            q_proj=column,
            k_proj=column,
            v_proj=column,
            f_a_proj=backend.linear(),
            f_b_proj=column,
            g_a_proj=backend.linear(),
            g_b_proj=column,
            b_proj=column,
            o_proj=backend.row_parallel_linear(),
        ),
        metainfo={"fuse_input_layernorm": False},
    )


def build_glm5_next_layer_spec(config: TransformerConfig, vp_stage: Optional[int] = None) -> TransformerBlockSubmodules:
    """Build the GLM-5.3-Flash decoder block spec for this pipeline stage.

    Layer ``i`` is KDA when ``config.linear_attention_freq[i]`` is 1 and DSA (NoPE MLA with the
    lightning indexer) otherwise; its MLP is MoE when ``config.moe_layer_freq[i]`` is 1 and a
    dense (layernorm-fused) MLP otherwise. Every layer is an mHC
    :class:`HyperConnectionTransformerLayer`.
    """
    if config.transformer_impl != "transformer_engine":
        raise ValueError("The GLM-5.3-Flash block spec requires transformer_impl='transformer_engine'.")
    backend = TESpecProvider()
    rms_norm = config.normalization == "RMSNorm"

    attention_pattern = get_linear_attention_pattern(config)
    moe_pattern = get_moe_layer_pattern(config)

    kda_spec = get_kda_module_spec(backend)
    dsa_spec = copy.deepcopy(get_dsa_module_spec_for_backend(config, backend))
    dsa_spec.submodules.core_attention.module = Glm5NextDSAttention

    moe_mlp = get_moe_module_spec_for_backend(
        backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    # The TE dense MLP fuses the pre-MLP layernorm into linear_fc1.
    dense_mlp = get_mlp_module_spec_for_backend(
        backend, num_experts=None, use_te_activation_func=config.use_te_activation_func
    )

    layer_specs = []
    for layer_idx in range(config.num_layers):
        is_moe = bool(moe_pattern[layer_idx])
        layer_specs.append(
            ModuleSpec(
                module=HyperConnectionTransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=backend.layer_norm(rms_norm=rms_norm, for_qk=False),
                    self_attention=kda_spec if attention_pattern[layer_idx] else dsa_spec,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=(backend.layer_norm(rms_norm=rms_norm, for_qk=False) if is_moe else IdentityOp),
                    mlp=moe_mlp if is_moe else dense_mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            )
        )

    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]

    return TransformerBlockSubmodules(
        layer_specs=layer_specs, layer_norm=backend.layer_norm(rms_norm=rms_norm, for_qk=False)
    )
