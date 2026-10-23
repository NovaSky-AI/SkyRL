"""Transformer layer with Manifold-Constrained Hyper-Connections (mHC), MoE sub-layers included.

megatron-core's own ``HyperConnectionTransformerLayer`` rejects MoE MLP submodules (mHC + MoE is
only reachable there by wrapping MoE as a ``HybridStack`` layer). This one supports dense *and*
MoE MLPs, which GLM-5.3-Flash needs (mHC on every block, MoE in all but the first layer).

The n residual streams travel between layers as ``[s, b, n * hidden_size]``;
``TransformerBlock`` owns the block-boundary expand (replication) and contract (unweighted mean)
whenever ``enable_mhc_connections`` is set, so this layer only implements the per-sub-layer
residual update.

Parameter names (``self_attention_hyper_connection.*`` / ``mlp_hyper_connection.*`` with
``mapping_proj.weight``, ``bias``, ``alpha_pre``, ``alpha_post``, ``alpha_res``) match upstream so
checkpoints and HF bridges carry over unchanged.
"""

from typing import Optional

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor

from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.hyper_connection import (
    RMSNormInputHyperConnectionModule,
)


class HyperConnectionTransformerLayer(TransformerLayer):
    """``TransformerLayer`` whose residual paths are mHC n-stream hyper-connections.

    For each sub-layer F (self-attention, MLP/MoE) with n-stream input x ([s, b, n*C]):

        h_pre, h_post, h_res = mapping(x)           # per-token, fp32, Sinkhorn-projected h_res
        y = F(norm(h_pre @ x))                      # single-stream sub-layer
        x = h_res^T @ x + h_post ⊗ y                # residual mix + broadcast of the update

    Cross-attention is not supported.
    """

    supports_mhc_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            **kwargs,
        )
        if not getattr(config, "enable_mhc_connections", False):
            raise ValueError("HyperConnectionTransformerLayer requires enable_mhc_connections=True.")
        if not isinstance(self.cross_attention, IdentityOp):
            raise ValueError("HyperConnectionTransformerLayer does not support cross-attention.")
        if self.recompute_input_layernorm or self.recompute_pre_mlp_layernorm or self.recompute_mlp:
            raise NotImplementedError(
                "Selective 'layernorm'/'mlp' activation recompute is not supported by "
                "HyperConnectionTransformerLayer."
            )
        # megatron-core's mHC layer threads a CheckpointWithoutOutputManager through every mHC
        # site; this layer does not, and TransformerBlock builds those managers purely from the
        # config, so accepting 'mhc' here would silently drop the recompute rather than apply it.
        if config.recompute_granularity == "selective" and "mhc" in (config.recompute_modules or ()):
            raise NotImplementedError(
                "'mhc' in recompute_modules is not supported by this layer; drop it from "
                "recompute_modules (selective recompute of the other modules still applies)."
            )
        if config.fp32_residual_connection:
            raise NotImplementedError("fp32_residual_connection is not supported with mHC streams.")

        # megatron-core builds these from ``submodules.self_attention_hyper_connection`` /
        # ``mlp_hyper_connection`` specs; ``TransformerLayerSubmodules`` has no such slots, so
        # build them directly. Their parameters are fp32-marked, so the cast below only applies
        # to anything a future variant may add in the activation dtype.
        hyper_connection_cls = (
            RMSNormInputHyperConnectionModule
            if getattr(config, "mhc_norm_eps_inside_sqrt", False)
            else HyperConnectionModule
        )
        with torch.device(torch.cuda.current_device()):
            self.self_attention_hyper_connection = hyper_connection_cls(config=config, layer_number=self.layer_number)
            self.mlp_hyper_connection = hyper_connection_cls(config=config, layer_number=self.layer_number)
        if config.params_dtype is not None:
            convert_module_to_dtype_except_fp32_marked(self.self_attention_hyper_connection, config.params_dtype)
            convert_module_to_dtype_except_fp32_marked(self.mlp_hyper_connection, config.params_dtype)

    @staticmethod
    def _reject_residual_returning_norm(layernorm_output, norm_name: str):
        """mHC needs the n-stream residual captured before aggregation, so a layernorm that
        also returns a (single-stream) residual cannot be used."""
        if isinstance(layernorm_output, tuple):
            raise ValueError(
                f"HyperConnectionTransformerLayer does not support a {norm_name} that returns an "
                "(output, residual) tuple."
            )
        return layernorm_output

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
        input_ids: Optional[Tensor] = None,
        mhc_recompute_manager=None,
        **kwargs,
    ):
        """Run one mHC block over n-stream hidden states (``[s, b, n * hidden_size]``)."""
        if context is not None:
            raise ValueError("HyperConnectionTransformerLayer does not support cross-attention context.")
        if mhc_recompute_manager is not None:
            raise NotImplementedError(
                "HyperConnectionTransformerLayer does not implement mHC activation recompute; "
                "drop 'mhc' from recompute_modules."
            )

        # Self-attention site.
        residual = hidden_states
        aggregated, h_res, h_post = self.self_attention_hyper_connection(hidden_states)
        input_layernorm_output = self._reject_residual_returning_norm(
            self.input_layernorm(aggregated), "input_layernorm"
        )
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
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
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attention_hyper_connection.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                attention_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
            )

        # MLP / MoE site.
        residual = hidden_states
        aggregated, h_res, h_post = self.mlp_hyper_connection(hidden_states)
        pre_mlp_layernorm_output = self._reject_residual_returning_norm(
            self.pre_mlp_layernorm(aggregated), "pre_mlp_layernorm"
        )
        pre_mlp_layernorm_output, moe_padding_mask, moe_input_ids, moe_unflatten_mbs = self._maybe_unflatten_for_moe(
            pre_mlp_layernorm_output, padding_mask, input_ids, packed_seq_params
        )
        # Also in eval mode (megatron-core chunks only in training): eval and logprob forwards see the
        # same per-rank routing spikes, and without chunking they ran out of memory first.
        # Token ids are only consumed by hash-routed MoE layers, as in ``TransformerLayer._forward_mlp``.
        mlp_output_with_bias = chunked_mlp(
            self.mlp,
            pre_mlp_layernorm_output,
            moe_padding_mask,
            self.config.mlp_chunks_for_training,
            input_ids=moe_input_ids if self.is_moe_layer else None,
        )
        if moe_unflatten_mbs is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = self._maybe_reflatten_from_moe(mlp_output, packed_seq_params, moe_unflatten_mbs)
            mlp_output_with_bias = (mlp_output, mlp_bias)
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_hyper_connection.fused_h_res_h_post_bda(
                h_res,
                residual,
                h_post,
                mlp_output_with_bias,
                self.hidden_dropout,
                self.training,
                self.config.bias_dropout_fusion,
            )

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)
        return output, None


def chunked_mlp(
    mlp,
    hidden_states: Tensor,
    padding_mask: Optional[Tensor],
    num_chunks: int,
    input_ids: Optional[Tensor] = None,
):
    """Run ``mlp`` over ``num_chunks`` sequence chunks of ``hidden_states`` ([s, b, h]).

    megatron-core's ``TransformerLayer._forward_mlp`` does this for ``mlp_chunks_for_training``; this
    layer replaces that method, so it does it here. MLP and MoE outputs are per token, so the result
    is unchanged; with MoE, only one chunk's routed tokens are dispatched at a time, which bounds the
    dispatch / expert / combine buffers on ranks whose experts receive far more than their share
    under real routing.
    """
    if num_chunks <= 1:
        kwargs = {"input_ids": input_ids} if input_ids is not None else {}
        return mlp(hidden_states, padding_mask=padding_mask, **kwargs)
    # Every rank must make the same number of MoE calls: each one runs expert-parallel
    # collectives, and a rank that made fewer (a short micro-batch, e.g. the one-token padding
    # micro-batches that even out micro-batch counts across DP ranks) deadlocks the group. So
    # always split into exactly num_chunks non-empty chunks, padding the sequence with masked
    # dummy tokens when it is shorter than that; their outputs are dropped.
    seq, batch = hidden_states.shape[:2]
    pad = max(0, num_chunks - seq)
    if pad:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        # Mask the dummy tokens even when the caller passed no mask: the router counts unmasked
        # tokens in its aux / z-loss and expert-bias token counts.
        if padding_mask is None:
            padding_mask = hidden_states.new_zeros(batch, seq, dtype=torch.bool)
        padding_mask = torch.nn.functional.pad(padding_mask, (0, pad), value=True)
        if input_ids is not None:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad))
    chunks = hidden_states.tensor_split(num_chunks, dim=0)
    masks = padding_mask.tensor_split(num_chunks, dim=-1) if padding_mask is not None else [None] * num_chunks
    # input_ids is [b, s] while hidden_states is [s, b, h], so it is split along dim 1.
    id_chunks = input_ids.tensor_split(num_chunks, dim=1) if input_ids is not None else [None] * num_chunks
    replay = _ChunkedRouterReplay(mlp, [c.shape[0] * batch for c in chunks], pad * batch)
    outputs = []
    for i, (chunk, mask, ids) in enumerate(zip(chunks, masks, id_chunks)):
        replay.before_chunk(i)
        outputs.append(mlp(chunk, padding_mask=mask, **({"input_ids": ids} if ids is not None else {})))
        replay.after_chunk()
    replay.finish()
    return torch.cat([out for out, _ in outputs], dim=0)[:seq], outputs[0][1]


class _ChunkedRouterReplay:
    """Router replay (R3) for ``chunked_mlp``.

    ``RouterReplay`` assumes one router call per micro-batch:

    - Expected: on replay, ``get_replay_topk`` returns ``target_topk_idx`` as the routing and
      gathers the router's scores with it (``scores.gather(1, target_topk_idx)``), so the indices
      must cover exactly the tokens of this call: ``[call_tokens, topk]``.
    - Actual: ``target_topk_idx`` holds the whole micro-batch, ``[s * b, topk]``, while
      ``chunked_mlp`` calls the router once per chunk, with scores of ``[chunk_tokens, experts]``.
    - So the gather fails on the first chunk ("Size does not match at dimension 0").
      Recompute (``REPLAY_BACKWARD``) pops one ``replay_backward_list`` entry per call, so the
      second chunk would take the next micro-batch's entry. ``RECORD`` keeps only the last call's
      indices, so the earlier chunks' routing would be lost.

    So each chunk gets its slice in ``target_topk_idx`` before its call, and the full tensor is
    put back after the last one. The micro-batch's ``replay_backward_list`` entry is split into
    one per chunk. Recorded top-k is concatenated back into one tensor.
    """

    def __init__(self, mlp, chunk_tokens, pad_tokens=0):
        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        router = getattr(mlp, "router", None)
        self.replay = getattr(router, "router_replay", None)
        self.action = self.replay.router_replay_action if self.replay is not None else None
        self.bounds = [0]
        for n in chunk_tokens:
            self.bounds.append(self.bounds[-1] + n)
        self.recorded = []
        self.real_tokens = self.bounds[-1] - pad_tokens
        self.target = None
        if self.action == RouterReplayAction.REPLAY_FORWARD:
            self.saved_target = self.replay.target_topk_idx
            self.target = self._pad(self.saved_target, pad_tokens)
        elif self.action == RouterReplayAction.REPLAY_BACKWARD and self.replay.replay_backward_list:
            full = self._pad(self.replay.replay_backward_list.pop(0), pad_tokens)
            self.replay.replay_backward_list[0:0] = [full[lo:hi] for lo, hi in zip(self.bounds[:-1], self.bounds[1:])]
        self._record = self.action == RouterReplayAction.RECORD

    @staticmethod
    def _pad(indices, pad_tokens):
        """Dummy tokens replay the first real token's experts (any valid, distinct top-k works)."""
        if not pad_tokens or indices is None:
            return indices
        if indices.shape[0] == 0:  # no real token to copy: experts 0..topk-1
            dummy = torch.arange(indices.shape[1], dtype=indices.dtype, device=indices.device)
            return dummy.expand(pad_tokens, -1)
        return torch.cat([indices, indices[:1].expand(pad_tokens, -1)], dim=0)

    def before_chunk(self, i):
        if self.target is not None:
            self.replay.target_topk_idx = self.target[self.bounds[i] : self.bounds[i + 1]]

    def after_chunk(self):
        if self._record and self.replay.recorded_topk_idx is not None:
            self.recorded.append(self.replay.recorded_topk_idx.clone())

    def finish(self):
        if self.target is not None:
            self.replay.target_topk_idx = self.saved_target
        if self._record and self.recorded:
            self.replay.recorded_topk_idx = torch.cat(self.recorded, dim=0)[: self.real_tokens]
