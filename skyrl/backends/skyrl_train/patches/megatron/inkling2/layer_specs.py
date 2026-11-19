"""Megatron block spec for Inkling2: attn-res decoder layers with MoE, attn-res output norm."""

from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from skyrl.backends.skyrl_train.patches.megatron.inkling2.modules import (
    Inkling2Attention,
    Inkling2AttentionSubmodules,
    Inkling2FinalNorm,
    Inkling2RMSNorm,
    Inkling2TransformerLayer,
)


def build_inkling2_layer_spec(config: TransformerConfig, vp_stage: Optional[int] = None) -> TransformerBlockSubmodules:
    """Every layer is an :class:`Inkling2TransformerLayer` with a routed-expert MoE; whether its
    attention is local or global is read from ``config.local_layer_ids`` by the modules."""
    if config.transformer_impl != "transformer_engine":
        raise ValueError("The Inkling2 block spec requires transformer_impl='transformer_engine'.")
    if config.pipeline_model_parallel_size != 1:
        # The attn-res state grows from 1 to n_blocks + 1 hidden widths across layers, which the
        # fixed-shape pipeline p2p communication cannot carry.
        raise NotImplementedError("Inkling2 does not support pipeline parallelism.")
    backend = TESpecProvider()
    attention = ModuleSpec(
        module=Inkling2Attention,
        submodules=Inkling2AttentionSubmodules(
            linear_column=backend.column_parallel_linear(),
            linear_row=backend.row_parallel_linear(),
        ),
    )
    moe = get_moe_module_spec_for_backend(
        backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    layer = ModuleSpec(
        module=Inkling2TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=Inkling2RMSNorm,
            self_attention=attention,
            pre_mlp_layernorm=Inkling2RMSNorm,
            mlp=moe,
        ),
    )
    num_layers = get_num_layers_to_build(config, vp_stage=vp_stage)
    return TransformerBlockSubmodules(layer_specs=[layer] * num_layers, layer_norm=Inkling2FinalNorm)
