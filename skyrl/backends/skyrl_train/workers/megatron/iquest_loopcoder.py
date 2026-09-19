"""Megatron-Core building blocks for LoRA-only IQuest LoopCoder training.

LoopCoder executes one physical decoder stack twice. The second pass mixes
full-context keys and values captured from the first pass with a sliding-window
attention over the second pass. The QKV, output, and MLP projections are the
same modules in both passes, so Megatron-Bridge LoRA naturally installs one
adapter that is reused by both passes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec
from megatron.core import tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import get_pg_size
from torch import nn


class LoopGateProjection(nn.Module):
    """Tensor-parallel form of LoopCoder's independent per-head gates."""

    def __init__(self, config, pg_collection, name: str | None = None):
        super().__init__()
        self.num_heads = config.num_attention_heads // get_pg_size(pg_collection.tp)
        self.head_dim = config.kv_channels
        self.gate_proj = tensor_parallel.ColumnParallelLinear(
            self.head_dim,
            config.num_attention_heads,
            config=config,
            init_method=config.init_method,
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            tp_group=pg_collection.tp,
            name=(name + ".gate_proj") if name is not None else None,
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        gate_logits, _ = self.gate_proj(query)
        if query.shape[-2] != self.num_heads or gate_logits.shape[-1] != self.num_heads:
            raise ValueError(
                "LoopCoder gate requires matching local query and output head counts; "
                f"got query={query.shape[-2]}, output={gate_logits.shape[-1]}, expected={self.num_heads}"
            )
        gate = torch.diagonal(gate_logits, dim1=-2, dim2=-1).unsqueeze(-1)
        gate = torch.sigmoid(gate.float()).to(query.dtype)
        return gate.expand(*gate.shape[:-1], self.head_dim).reshape(*query.shape[:-2], -1)


class LoopCoderCoreAttention(nn.Module):
    """Mix full loop-1 KV with loop-2 sliding-window attention."""

    def __init__(self, global_attention, local_attention, gate_projection):
        super().__init__()
        self.global_attention = global_attention
        self.local_attention = local_attention
        self.gate_projection = gate_projection
        self.loop_idx = 0
        self._shared_key: torch.Tensor | None = None
        self._shared_value: torch.Tensor | None = None

    @property
    def softmax_offset(self):
        return getattr(self.global_attention, "softmax_offset", None)

    def forward(self, query, key, value, attention_mask, **kwargs):
        if self.loop_idx == 0:
            self._shared_key = key
            self._shared_value = value
            return self.global_attention(query, key, value, attention_mask, **kwargs)

        if self.loop_idx != 1:
            raise ValueError(f"IQuest LoopCoder supports exactly two passes, got loop_idx={self.loop_idx}")
        if self._shared_key is None or self._shared_value is None:
            raise RuntimeError("LoopCoder's second pass ran before first-pass KV was captured")

        global_output = self.global_attention(
            query,
            self._shared_key,
            self._shared_value,
            attention_mask,
            **kwargs,
        )
        local_output = self.local_attention(query, key, value, attention_mask, **kwargs)
        gate = self.gate_projection(query)
        return local_output * (1.0 - gate) + global_output * gate


class LoopCoderSelfAttention(SelfAttention):
    """MCore self-attention with LoopCoder's two-pass core attention."""

    def __init__(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        cp_comm_type=None,
        pg_collection=None,
        pp_layer_offset=None,
        name=None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )
        local_config = copy.deepcopy(config)
        local_config.window_size = (config.loop_window_size - 1, 0)
        local_config.window_attn_skip_freq = None
        local_config.num_query_groups = max(local_config.num_query_groups, get_pg_size(self.pg_collection.tp))
        local_attention = submodules.core_attention(
            config=local_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            cp_comm_type=cp_comm_type,
            softmax_scale=self.config.softmax_scale,
            pg_collection=self.pg_collection,
        )
        self.core_attention = LoopCoderCoreAttention(
            global_attention=self.core_attention,
            local_attention=local_attention,
            gate_projection=LoopGateProjection(config, self.pg_collection, name=name),
        )


def loopcoder_layer_spec(config):
    """Return the ordinary dense GPT spec with LoopCoder self-attention."""
    spec = copy.deepcopy(default_layer_spec(config))
    spec.submodules.self_attention.module = LoopCoderSelfAttention
    return spec


class LoopCoderTransformerBlock(TransformerBlock):
    """Run one physical TransformerBlock twice without duplicating parameters."""

    def _set_loop_idx(self, loop_idx: int) -> None:
        for layer in self.layers:
            layer.self_attention.core_attention.loop_idx = loop_idx

    def forward(self, *args, **kwargs):
        if kwargs.get("extract_layer_indices"):
            raise NotImplementedError("LoopCoder does not support extracting intermediate layers")

        final_layernorm = self.final_layernorm
        self._set_loop_idx(0)
        self.final_layernorm = None
        try:
            hidden_states = super().forward(*args, **kwargs)
        finally:
            self.final_layernorm = final_layernorm

        self._set_loop_idx(1)
        if args:
            args = (hidden_states, *args[1:])
        else:
            kwargs["hidden_states"] = hidden_states
        return super().forward(*args, **kwargs)


@dataclass
class LoopCoderProvider(GPTModelProvider):
    """Provider for the two-pass, weight-shared LoopCoder GPT model."""

    loop_num: int = 2
    loop_window_size: int = 64

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        if self.loop_num != 2:
            raise ValueError(f"Only the released two-pass LoopCoder is supported, got loop_num={self.loop_num}")
        if self.pipeline_model_parallel_size != 1:
            raise ValueError("LoopCoder training currently requires pipeline_model_parallel_size=1")
        if self.recompute_granularity == "full":
            raise ValueError(
                "LoopCoder requires selective activation recomputation; full-block recomputation "
                "would sever gradients through the first-pass KV used by pass two"
            )
        if self.recompute_granularity == "selective" and "core_attn" in (self.recompute_modules or []):
            raise ValueError(
                "LoopCoder cannot recompute core attention because its checkpoint replay would run after "
                "the shared attention module has advanced to pass two; use recompute_modules=['mlp']"
            )

        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        model.decoder.__class__ = LoopCoderTransformerBlock
        return model
