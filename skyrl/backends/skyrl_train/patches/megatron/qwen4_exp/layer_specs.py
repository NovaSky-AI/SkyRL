"""Megatron block spec for Qwen3.8-Flash-Next (``qwen4_exp``).

Every layer is a :class:`Qwen4ExpTransformerLayer`: gated low-rank hyper-connections around a
GDN-or-gated-attention mixer and a shared-expert MoE, with the PLE n-gram injection on the
``ple_layer_ids`` layers. The 4-stream residual is ``[s, b, n*C]`` between layers; the first
layer expands the embedding by replication and the block's final "layer norm" is the learned
``hyper_connection_mixer`` that contracts it (HF has no final RMSNorm).
"""

import copy
from typing import Optional

import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TEDotProductAttention,
)
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _get_self_attention_module_spec,
    get_gated_delta_net_module_spec,
    get_linear_attention_pattern,
)
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.mhc_transformer_layer import (
    chunked_mlp,
)
from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
    Qwen4ExpFinalMixer,
    Qwen4ExpGatedDeltaNet,
    Qwen4ExpGatedResidual,
    Qwen4ExpPLE,
    Qwen4ExpQSAIndexer,
    qsa_sparse_attention,
    sp_gather,
    sp_local,
)


class Qwen4ExpCoreAttention(TEDotProductAttention):
    """TE core attention that switches to the QSA sparse attention when a selection is set.

    The layer sets ``qsa_selection`` from the indexer right before the attention call: None
    (every packed sequence within the exactly-dense bound) keeps the fused TE kernel, otherwise
    attention runs over the selected keys (``qsa_sparse_attention``, torch SDPA with a mask).
    """

    qsa_selection = None

    def forward(self, query, key, value, attention_mask, *args, **kwargs):
        selection = self.qsa_selection
        if selection is None:
            return super().forward(query, key, value, attention_mask, *args, **kwargs)
        scale = getattr(self, "softmax_scale", None) or query.shape[-1] ** -0.5
        return qsa_sparse_attention(query, key, value, selection, scale)


class Qwen4ExpTransformerLayer(TransformerLayer):
    """Decoder layer of Qwen4Exp. For each sub-layer F with n-stream residual x ([s, b, n*C]):

        y, w = hc(x)          # gated low-rank mix of the normed streams, injection weights
        x = x + y' ⊗ w        # y' = F(y); there is no separate pre-norm

    with ``x += PLE(x, input_ids)`` before the attention site on the PLE layers.

    Under sequence parallelism ``x`` holds this rank's tokens. PLE and the QSA indexer need whole
    sequences, so their inputs are all-gathered over TP; the attention / GDN and MoE handle SP
    themselves.
    """

    def __init__(
        self, config: TransformerConfig, submodules: TransformerLayerSubmodules, layer_number: int = 1, **kwargs
    ):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kwargs)
        if not isinstance(self.cross_attention, IdentityOp):
            raise ValueError("Qwen4ExpTransformerLayer does not support cross-attention.")

        global_idx = self.layer_number - 1  # layer_number is global and 1-based
        with torch.device(torch.cuda.current_device()):
            self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
            self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)
            ple_ids = list(config.qwen4_ple_layer_ids or ())
            # HF: layer ``i`` has PLE iff ``i + 1`` is in ``ple_layer_ids``.
            self.ple = Qwen4ExpPLE(config, ple_ids.index(global_idx + 1)) if global_idx + 1 in ple_ids else None
            if not self._qwen4_is_gdn:
                self.self_attention.indexer = Qwen4ExpQSAIndexer(config)
        indexer = None if self._qwen4_is_gdn else self.self_attention.indexer
        for module in (self.attn_hyper_connection, self.mlp_hyper_connection, self.ple, indexer):
            if module is not None and config.params_dtype is not None:
                module.to(config.params_dtype)
        self.expand_input = global_idx == 0
        self.sp_group = self.tp_group if config.sequence_parallel and self.tp_group.size() > 1 else None

    @property
    def _qwen4_is_gdn(self) -> bool:
        return isinstance(self.self_attention, Qwen4ExpGatedDeltaNet)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        if self.expand_input:
            hidden_states = hidden_states.repeat(1, 1, self.config.qwen4_hc_count)
        if self.ple is not None:
            hidden_states = hidden_states + sp_local(self.ple(sp_gather(hidden_states, self.sp_group)), self.sp_group)

        # Attention / GDN site.
        attn_in, inject = self.attn_hyper_connection(hidden_states)
        if not self._qwen4_is_gdn:
            core_attention = self.self_attention.core_attention
            # sp_gather under no_grad: the indexer is frozen and its selection non-differentiable.
            with torch.no_grad():
                index_in = sp_gather(attn_in, self.sp_group)
            core_attention.qsa_selection = self.self_attention.indexer.select(
                index_in, rotary_pos_emb, packed_seq_params
            )
        attn_out, attn_bias = self.self_attention(
            attn_in,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if not self._qwen4_is_gdn and not self.self_attention.checkpoint_core_attention:
            # Kept only when selective core-attention recompute replays the attention in backward.
            core_attention.qsa_selection = None
        if attn_bias is not None:
            attn_out = attn_out + attn_bias
        hidden_states = Qwen4ExpGatedResidual.inject(hidden_states, attn_out, inject)

        # MoE site.
        mlp_in, inject = self.mlp_hyper_connection(hidden_states)
        mlp_in, moe_padding_mask, _, moe_unflatten_mbs = self._maybe_unflatten_for_moe(
            mlp_in, padding_mask, None, packed_seq_params
        )
        mlp_out, mlp_bias = chunked_mlp(self.mlp, mlp_in, moe_padding_mask, self.config.mlp_chunks_for_training)
        if moe_unflatten_mbs is not None:
            mlp_out = self._maybe_reflatten_from_moe(mlp_out, packed_seq_params, moe_unflatten_mbs)
        if mlp_bias is not None:
            mlp_out = mlp_out + mlp_bias
        hidden_states = Qwen4ExpGatedResidual.inject(hidden_states, mlp_out, inject)

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output, None


def _without_input_layernorm(spec: ModuleSpec, linear_name: str) -> ModuleSpec:
    """Swap the fused layernorm+linear input projection for a plain column-parallel linear."""
    spec = copy.deepcopy(spec)
    setattr(spec.submodules, linear_name, TEColumnParallelLinear)
    spec.metainfo["fuse_input_layernorm"] = False
    return spec


def build_qwen4_exp_layer_spec(config: TransformerConfig, vp_stage: Optional[int] = None) -> TransformerBlockSubmodules:
    """Build the Qwen4Exp decoder block spec for this pipeline stage."""
    if config.transformer_impl != "transformer_engine":
        raise ValueError("The Qwen4Exp block spec requires transformer_impl='transformer_engine'.")
    backend = TESpecProvider()

    gdn_spec = _without_input_layernorm(get_gated_delta_net_module_spec(config, backend), "in_proj")
    gdn_spec.module = Qwen4ExpGatedDeltaNet
    attn_spec = _without_input_layernorm(_get_self_attention_module_spec(config, backend), "linear_qkv")
    attn_spec.submodules.core_attention = Qwen4ExpCoreAttention
    moe_spec = get_moe_module_spec_for_backend(
        backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )

    attention_pattern = get_linear_attention_pattern(config)
    layer_specs = [
        ModuleSpec(
            module=Qwen4ExpTransformerLayer,
            submodules=TransformerLayerSubmodules(
                input_layernorm=IdentityOp,
                self_attention=gdn_spec if attention_pattern[i] else attn_spec,
                self_attn_bda=get_bias_dropout_add,
                pre_mlp_layernorm=IdentityOp,
                mlp=moe_spec,
                mlp_bda=get_bias_dropout_add,
            ),
        )
        for i in range(config.num_layers)
    ]
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
    layer_specs = layer_specs[offset : offset + get_num_layers_to_build(config, vp_stage=vp_stage)]
    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=Qwen4ExpFinalMixer)
