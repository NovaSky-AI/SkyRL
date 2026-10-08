"""Kimi Delta Attention (KDA) linear attention for Megatron-Core.

KDA (Kimi Linear, arXiv:2510.26692) is a gated delta rule with a per-channel (rather than
per-head) forget gate. GLM-5.3-Flash uses it for 3 of every 4 layers. The layer computes, for
hidden states x:

    q, k, v = SiLU(conv1d(W_q x)), SiLU(conv1d(W_k x)), SiLU(conv1d(W_v x))     # depthwise, causal
    f       = W_fb (W_fa x)                                                       # low-rank forget gate
    g       = lower_bound * sigmoid(exp(A_log) * (f + dt_bias))                  # log-decay in [lb, 0)
    beta    = sigmoid(W_b x)                                                     # per-head write strength
    o       = chunk_kda(l2norm(q), l2norm(k), v, g, beta)                        # fla kernel
    y       = W_o (RMSNorm(o) * sigmoid(W_gb (W_ga x)))                          # gated output norm

Tensor parallelism shards heads: q/k/v/f_b/g_b/b projections are column-parallel, the
convolutions, ``A_log`` and ``dt_bias`` are split along the head dimension, the low-rank
``f_a``/``g_a`` down-projections are duplicated, and ``o_proj`` is row-parallel. Packed
sequences (``qkv_format == "thd"``) are supported through ``cu_seqlens``; inference caches are not.

Context parallelism is head-wise (Ulysses), as megatron-core's ``GatedDeltaNet`` does it: the
projections run on each rank's sequence shard, an all-to-all trades the sequence shard for a
1/cp slice of the local heads over the full sequence, the convolutions and ``chunk_kda`` run on
those heads (each head's recurrence is independent), and a second all-to-all returns the gated
norm output to sequence shards before ``o_proj``. The convolution weights, ``A_log`` and
``dt_bias`` stay whole on every CP rank and are sliced per rank in the forward pass, so their
gradients land in the matching slice and are summed by the data-parallel x CP all-reduce.

Config fields (shared with ``GatedDeltaNet``): ``linear_num_value_heads`` /
``linear_num_key_heads`` (equal), ``linear_key_head_dim`` / ``linear_value_head_dim`` (equal),
``linear_conv_kernel_dim``; plus ``kda_gate_lower_bound`` (``None`` selects the unbounded
``-exp(A_log) * softplus(f + dt_bias)`` gate of the original Kimi Linear).
"""

import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import (
    _build_thd_cp_a2a_perm,
    a2a_cp_to_hp,
    a2a_hp_to_cp,
    get_parameter_local_cp,
)
from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)
from torch import nn

# How KDA moves to head parallelism under CP: "a2a" (all-to-all the projected q/k/v/f/g/beta, as
# megatron-core's GatedDeltaNet does) or "allgather" (all-gather the sequence-parallel hidden-state
# shards over CP, then over TP, and project only this rank's head slice). The projected tensors are
# ~1.25x wider than the hidden state and the all-to-all runs on TP-gathered rows, so "allgather"
# moves ~5x fewer bytes across the CP group -- which matters when CP spans nodes.
_KDA_CP_EXCHANGE = os.environ.get("SKYRL_KDA_CP_EXCHANGE", "a2a").lower()

try:
    from fla.modules import FusedRMSNormGated
    from fla.modules.convolution import causal_conv1d
    from fla.ops.kda import chunk_kda

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


@dataclass
class KimiDeltaAttentionSubmodules:
    """Submodule specs for :class:`KimiDeltaAttention`.

    ``q_proj``/``k_proj``/``v_proj``/``f_b_proj``/``g_b_proj``/``b_proj`` are column-parallel
    linears, ``f_a_proj``/``g_a_proj`` are duplicated (non-parallel) linears and ``o_proj`` is
    row-parallel.
    """

    q_proj: Union[ModuleSpec, type] = None
    k_proj: Union[ModuleSpec, type] = None
    v_proj: Union[ModuleSpec, type] = None
    f_a_proj: Union[ModuleSpec, type] = None
    f_b_proj: Union[ModuleSpec, type] = None
    g_a_proj: Union[ModuleSpec, type] = None
    g_b_proj: Union[ModuleSpec, type] = None
    b_proj: Union[ModuleSpec, type] = None
    o_proj: Union[ModuleSpec, type] = None


class KimiDeltaAttention(MegatronModule):
    """KDA linear attention layer; drop-in for ``self_attention`` in a transformer layer spec."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: KimiDeltaAttentionSubmodules,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        cp_comm_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if not HAVE_FLA:
            raise ImportError(
                "flash-linear-attention is required for KimiDeltaAttention " "(`pip install flash-linear-attention`)."
            )
        super().__init__(config=config)
        del cp_comm_type, kwargs
        self.layer_number = layer_number
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size()
        self.cp_group = pg_collection.cp
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1

        if config.linear_num_key_heads != config.linear_num_value_heads:
            raise ValueError("KDA uses the same number of q/k and v heads.")
        if config.linear_key_head_dim != config.linear_value_head_dim:
            raise ValueError("KDA uses the same head dimension for q/k and v.")
        self.num_heads = config.linear_num_value_heads
        self.head_dim = config.linear_key_head_dim
        self.conv_kernel_dim = config.linear_conv_kernel_dim
        self.gate_lower_bound = getattr(config, "kda_gate_lower_bound", -5.0)
        if self.num_heads % self.tp_size != 0:
            raise ValueError(
                f"KDA heads ({self.num_heads}) must be divisible by tensor_model_parallel_size " f"({self.tp_size})."
            )
        self.local_num_heads = self.num_heads // self.tp_size
        if self.local_num_heads % self.cp_size != 0:
            raise ValueError(
                f"KDA heads per tensor-parallel rank ({self.local_num_heads}) must be divisible by "
                f"context_parallel_size ({self.cp_size})."
            )
        self.projection_size = self.num_heads * self.head_dim
        self.local_projection_size = self.local_num_heads * self.head_dim
        hidden_size = config.hidden_size
        device = torch.cuda.current_device()

        def column(spec, input_size, output_size, sub):
            return build_module(
                spec,
                input_size,
                output_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.tp_group,
                name=(name + "." + sub) if name is not None else None,
            )

        def duplicated(spec, input_size, output_size, sub):
            return build_module(
                spec,
                input_size,
                output_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
                name=(name + "." + sub) if name is not None else None,
            )

        self.q_proj = column(submodules.q_proj, hidden_size, self.projection_size, "q_proj")
        self.k_proj = column(submodules.k_proj, hidden_size, self.projection_size, "k_proj")
        self.v_proj = column(submodules.v_proj, hidden_size, self.projection_size, "v_proj")
        self.f_a_proj = duplicated(submodules.f_a_proj, hidden_size, self.head_dim, "f_a_proj")
        self.f_b_proj = column(submodules.f_b_proj, self.head_dim, self.projection_size, "f_b_proj")
        self.g_a_proj = duplicated(submodules.g_a_proj, hidden_size, self.head_dim, "g_a_proj")
        self.g_b_proj = column(submodules.g_b_proj, self.head_dim, self.projection_size, "g_b_proj")
        self.b_proj = column(submodules.b_proj, hidden_size, self.num_heads, "b_proj")

        # Depthwise causal convolutions on q, k, v (weights [local_projection_size, 1, kernel]).
        def conv():
            module = nn.Conv1d(
                in_channels=self.local_projection_size,
                out_channels=self.local_projection_size,
                bias=False,
                kernel_size=self.conv_kernel_dim,
                groups=self.local_projection_size,
                padding=self.conv_kernel_dim - 1,
                device=device,
                dtype=config.params_dtype,
            )
            setattr(module.weight, "tensor_model_parallel", True)
            setattr(module.weight, "partition_dim", 0)
            return module

        self.q_conv1d = conv()
        self.k_conv1d = conv()
        self.v_conv1d = conv()

        # Decay parameters, kept in fp32 like the reference implementation. ``dt_bias`` is laid
        # out head-major so the TP shard matches the local heads.
        self.A_log = nn.Parameter(torch.zeros(self.local_num_heads, dtype=torch.float32, device=device))
        self.dt_bias = nn.Parameter(torch.zeros(self.local_projection_size, dtype=torch.float32, device=device))
        for param in (self.A_log, self.dt_bias):
            setattr(param, "tensor_model_parallel", True)
            setattr(param, "partition_dim", 0)
            param.keep_in_fp32 = True

        # Gated RMSNorm over each head's channels: RMSNorm(o) * sigmoid(gate). The weight is
        # replicated across TP ranks while each rank only sees its local heads, so its gradient
        # must be summed across the TP group -- the ``sequence_parallel`` attribute is how
        # megatron-core's finalize_model_grads requests exactly that reduction.
        self.o_norm = FusedRMSNormGated(
            self.head_dim,
            eps=config.layernorm_epsilon,
            activation="sigmoid",
            device=device,
            dtype=config.params_dtype,
        )
        if config.sequence_parallel:
            setattr(self.o_norm.weight, "sequence_parallel", True)

        self.o_proj = build_module(
            submodules.o_proj,
            self.projection_size,
            hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.tp_group,
            name=(name + ".o_proj") if name is not None else None,
        )

    @staticmethod
    def _resolve_cu_seqlens(packed_seq_params: Optional[PackedSeqParams]) -> Optional[torch.Tensor]:
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            return None
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        if cu_seqlens is None:
            raise ValueError("Packed (thd) KDA input requires cu_seqlens_q.")
        # Global (all CP ranks) sequence boundaries, which is what the full-sequence kernels see.
        return cu_seqlens.to(dtype=torch.long)

    def _local_cp(self, param: torch.Tensor) -> torch.Tensor:
        """This CP rank's slice of a head-major per-channel/per-head parameter."""
        if self.cp_size == 1:
            return param
        return get_parameter_local_cp(param, dim=0, cp_group=self.cp_group)

    def _use_cp_allgather(self) -> bool:
        # The all-gather path replaces the column-parallel layers' own sequence-parallel gather,
        # so it needs the input sharded over TP (sequence parallel) or no TP at all.
        return _KDA_CP_EXCHANGE == "allgather" and (self.tp_size == 1 or self.config.sequence_parallel)

    def _project_cp_allgather(self, hidden_states, cu_seqlens, packed_seq_params):
        """Head-parallel q/k/v/f/gate/beta via an all-gather of the hidden states.

        ``hidden_states`` is this rank's sequence-parallel shard of its CP chunk. Gathering over CP
        first (the only cross-node hop when CP spans nodes) then over TP gives every rank the whole
        sequence in [tp][cp][rows] order; each rank then projects only its 1/cp slice of the local
        heads (the same channels the all-to-all would hand it), and one permutation of the narrow
        projected tensors puts tokens into natural order. Backward reduce-scatters in reverse.
        """
        x = gather_from_sequence_parallel_region(hidden_states, group=self.cp_group)
        if self.tp_size > 1:
            x = gather_from_sequence_parallel_region(x, group=self.tp_group)
        rows = hidden_states.size(0)  # per (tp, cp) rank
        total = x.size(0)

        def local(module):
            return self._local_cp(module.weight)

        f_low = self.f_a_proj(x)[0]
        g_low = self.g_a_proj(x)[0]
        projected = torch.cat(
            (
                torch.nn.functional.linear(x, local(self.q_proj)),
                torch.nn.functional.linear(x, local(self.k_proj)),
                torch.nn.functional.linear(x, local(self.v_proj)),
                torch.nn.functional.linear(f_low, local(self.f_b_proj)),
                torch.nn.functional.linear(g_low, local(self.g_b_proj)),
                torch.nn.functional.linear(x, local(self.b_proj)),
            ),
            dim=-1,
        )
        del x, f_low, g_low

        # natural token -> rank-major (cp-major, then tp) row, as a2a_cp_to_hp lays it out.
        thd_cp_a2a_inv = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            rank_major, thd_cp_a2a_inv = _build_thd_cp_a2a_perm(cu_seqlens, self.cp_size, total)
        else:
            positions = torch.arange(total, device=projected.device).view(total, 1, 1)
            rank_major = _undo_attention_load_balancing(positions, self.cp_size).view(-1)
        # rank-major row r sits in gathered chunk (tp, cp) = (chunk % tp, chunk // tp).
        chunk, offset = rank_major // rows, rank_major % rows
        gathered = ((chunk % self.tp_size) * self.cp_size + chunk // self.tp_size) * rows + offset
        projected = projected.index_select(0, gathered)

        heads = self.local_num_heads // self.cp_size
        width = heads * self.head_dim
        return torch.split(projected, [width] * 5 + [heads], dim=-1), thd_cp_a2a_inv

    def _conv(self, module: nn.Conv1d, x: torch.Tensor, cu_seqlens: Optional[torch.Tensor]) -> torch.Tensor:
        # fla expects [b, s, d] activations and a [d, kernel] weight.
        out, _ = causal_conv1d(
            x=x,
            weight=self._local_cp(module.weight).squeeze(1),
            bias=None,
            activation="silu",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
        )
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        **kwargs,
    ):
        """Run KDA on ``hidden_states`` ([s, b, hidden_size]; sequence-parallel shards allowed).

        Returns ``(output, bias)`` like the attention modules it replaces.
        """
        del attention_mask, kwargs
        if inference_context is not None:
            raise NotImplementedError("KimiDeltaAttention does not support inference caches.")
        cu_seqlens = self._resolve_cu_seqlens(packed_seq_params)

        cp_allgather = self.cp_size > 1 and self._use_cp_allgather()
        if not cp_allgather:
            # Column-parallel projections gather the sequence-parallel shard internally, so every
            # per-token/per-sequence op below sees this CP rank's whole shard ([s, b, local]).
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
            f, _ = self.f_b_proj(self.f_a_proj(hidden_states)[0])
            gate, _ = self.g_b_proj(self.g_a_proj(hidden_states)[0])
            beta, _ = self.b_proj(hidden_states)

        num_heads = self.local_num_heads
        thd_cp_a2a_inv = None
        if cp_allgather:
            (q, k, v, f, gate, beta), thd_cp_a2a_inv = self._project_cp_allgather(
                hidden_states, cu_seqlens, packed_seq_params
            )
            num_heads //= self.cp_size
        elif self.cp_size > 1:
            # CP -> head-parallel: [s/cp, b, heads] -> [s, b, heads/cp] in natural token order,
            # all six tensors in one all-to-all.
            sections = [self.local_projection_size] * 5 + [self.local_num_heads]
            mixed, thd_cp_a2a_inv = a2a_cp_to_hp(
                torch.cat((q, k, v, f, gate, beta), dim=-1),
                tuple(sections),
                self.cp_size,
                self.cp_group,
                cu_seqlens,
                q.size(0) * self.cp_size,
                packed_seq_params,
            )
            q, k, v, f, gate, beta = torch.split(mixed, [n // self.cp_size for n in sections], dim=-1)
            num_heads //= self.cp_size

        # [s, b, ·] -> [b, s, ·] (fla layout; b == 1 for packed sequences).
        q, k, v, f, gate, beta = (t.transpose(0, 1).contiguous() for t in (q, k, v, f, gate, beta))
        batch, seq_len, _ = q.shape
        if cu_seqlens is not None and batch != 1:
            raise ValueError("Packed KDA input expects batch dimension 1.")

        q = self._conv(self.q_conv1d, q, cu_seqlens)
        k = self._conv(self.k_conv1d, k, cu_seqlens)
        v = self._conv(self.v_conv1d, v, cu_seqlens)
        head_shape = (batch, seq_len, num_heads, self.head_dim)
        q, k, v, f = (t.view(head_shape) for t in (q, k, v, f))

        core_attn_out, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=f,
            beta=beta.float().sigmoid(),
            A_log=self._local_cp(self.A_log),
            dt_bias=self._local_cp(self.dt_bias),
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            safe_gate=self.gate_lower_bound is not None,
            lower_bound=self.gate_lower_bound,
            cu_seqlens=cu_seqlens,
        )

        out = self.o_norm(core_attn_out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim))
        out = out.view(batch, seq_len, num_heads * self.head_dim).transpose(0, 1)
        if self.cp_size > 1:
            # Head-parallel -> CP: back to this rank's sequence shard with all local heads.
            out = a2a_hp_to_cp(out.contiguous(), self.cp_size, self.cp_group, packed_seq_params, thd_cp_a2a_inv)
        return self.o_proj(out.to(hidden_states.dtype))

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        """Sharded state with the head-split convolution, ``A_log`` and ``dt_bias`` tensors."""
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        tp_group = tp_group if tp_group is not None else self.tp_group
        sharded_state_dict = {}
        self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            {"A_log": 0, "dt_bias": 0},
            sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )
        for name, module in self.named_children():
            if name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                module_sharded_sd = make_sharded_tensors_for_checkpoint(
                    module.state_dict(prefix="", keep_vars=True),
                    f"{prefix}{name}.",
                    {"weight": 0},
                    sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=metadata["dp_cp_group"],
                )
            else:
                module_sharded_sd = sharded_state_dict_default(
                    module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
                )
            sharded_state_dict.update(module_sharded_sd)
        return sharded_state_dict
