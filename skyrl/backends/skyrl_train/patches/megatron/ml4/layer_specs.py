"""Megatron block spec for Mistral Large 4: ML4 attention, dense or MoE MLP, vLLM-exact norms."""

import dataclasses
import functools
from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec_for_backend
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)

from skyrl.backends.skyrl_train.patches.megatron.ml4.experts import (
    ML4MoELayer,
    ML4NVFP4GroupedMLP,
)
from skyrl.backends.skyrl_train.patches.megatron.ml4.modules import (
    ML4Attention,
    ML4RMSNorm,
    ML4Router,
)


@dataclasses.dataclass
class ML4AttentionSubmodules:
    linear_q_proj: type
    linear_kv_down_proj: type
    linear_kv_up_proj: type
    linear_proj: type


class _UnfusedTESpecProvider(TESpecProvider):
    """TE linears without the layernorm fused into linear_fc1: ML4 keeps vLLM's RMSNorm."""

    def fuse_layernorm_and_linear(self) -> bool:
        return False


def build_ml4_layer_spec(config, vp_stage: Optional[int] = None) -> TransformerBlockSubmodules:
    """Decoder block spec for this pipeline stage.

    Layer ``i`` has ML4 attention and a MoE MLP when ``config.moe_layer_freq[i]`` is 1, else
    a dense SwiGLU MLP. Norms are :class:`ML4RMSNorm` (vLLM's RMSNorm numerics).
    """
    if config.transformer_impl != "transformer_engine":
        raise ValueError("The ML4 block spec requires transformer_impl='transformer_engine'.")
    if config.recompute_granularity == "full":
        raise ValueError(
            "ML4 carries cross-layer state (shared KV latent, shared index selection) through the "
            "forward; full-layer recompute would re-run layers out of that order. Use "
            "recompute_granularity='selective' with recompute_modules including 'core_attn'/'moe'."
        )
    if (config.virtual_pipeline_model_parallel_size or 1) > 1:
        raise ValueError("ML4 does not support virtual pipeline parallelism.")
    backend = _UnfusedTESpecProvider()
    attention = ModuleSpec(
        module=ML4Attention,
        submodules=ML4AttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_linear(),
            linear_proj=backend.row_parallel_linear(),
        ),
    )
    moe_mlp = get_moe_module_spec_for_backend(
        backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=False,
    )
    moe_mlp.keywords["submodules"].router = ML4Router
    if config.ml4_nvfp4_experts:
        # checkpoint NVFP4 bytes resident, FSDP-trainer (vLLM) expert math; FP32 per-token combine
        moe_mlp = functools.partial(ML4MoELayer, *moe_mlp.args, **moe_mlp.keywords)
        moe_mlp.keywords["submodules"].experts = ML4NVFP4GroupedMLP
    dense_mlp = get_mlp_module_spec_for_backend(backend, num_experts=None, use_te_activation_func=False)

    moe_pattern = config.moe_layer_freq
    layer_specs = []
    for i in range(config.num_layers):
        layer_specs.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=ML4RMSNorm,
                    self_attention=attention,
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=ML4RMSNorm,
                    mlp=moe_mlp if moe_pattern[i] else dense_mlp,
                    mlp_bda=get_bias_dropout_add,
                ),
            )
        )

    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    n = get_num_layers_to_build(config, vp_stage=vp_stage)
    stage = range(offset, offset + n)
    for i in stage:
        src = config.ml4_kv_source_layers[i]
        if src not in stage:
            raise ValueError(f"ML4 layer {i} shares the KV latent of layer {src}, which is on another pipeline stage")
        if config.ml4_index_topk is not None and config.ml4_index_sharing[i] == "shared" and i == offset:
            raise ValueError(f"ML4 shared-index layer {i} starts a pipeline stage; it needs the previous indexer")
    return TransformerBlockSubmodules(layer_specs=layer_specs[offset : offset + n], layer_norm=ML4RMSNorm)
