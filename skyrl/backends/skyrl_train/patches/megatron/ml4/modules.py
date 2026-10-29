"""Mistral Large 4 (ML4) modules on top of an unmodified Megatron-Core ``GPTModel``.

Megatron-Core covers ML4's decoder layer, MoE dispatch and grouped experts. ML4-specific pieces
are the subclasses / modules below, swapped in by :func:`.layer_specs.build_ml4_layer_spec`:

* :class:`ML4RMSNorm` -- vLLM's RMSNorm numerics (FP32 ``x * rsqrt(var + eps) * w``, one rounding);
* :class:`ML4Attention` -- MLA with per-layer RoPE/NoPE, cross-layer KV-latent sharing,
  SSMax, LeRoPE + YaRN rotary, the DSA lightning indexer with per-layer index sharing, and the
  FP8 latent-KV simulation;
* :class:`ML4Router` -- softmax top-k with vLLM's BF16 router logits, stable lowest-index
  tie-break and renormalization, through Megatron's ``RouterReplay`` (R3);
* :class:`ML4OutputLayer` -- FP32 final-logit tanh cap.

Numerics come from the FSDP trainer's own functions (vendored in ``ml4_train``), so both
trainers run the same ops; each function cites the vLLM code it mirrors.

Cross-layer state (the KV latent of a sharing source layer and the last indexer selection)
is carried on the forward's ``PackedSeqParams`` object, which every layer of one forward
receives. Requirements that follow, checked at build/forward time:

* inputs are THD-packed (``packed_seq_params.qkv_format == "thd"``, micro-batch dim 1);
* no full-layer activation recompute (a recomputed layer would read state from another
  forward); selective recompute of the attention core is done inside the module;
* a KV-sharing consumer and its source, and a shared-index layer and the preceding indexer
  layer, sit on the same pipeline stage.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    copy_to_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.layers import set_tensor_model_parallel_attributes
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplayAction
from megatron.core.transformer.spec_utils import build_module
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.ml4.experts import CALLS
from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train import ops
from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train.qat import ste

# --------------------------------------------------------------------------------------------
# Norms
# --------------------------------------------------------------------------------------------


class ML4RMSNorm(MegatronModule):
    """RMSNorm with vLLM's ``vllm_c`` numerics (``ops.rms_norm``)."""

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, **kwargs) -> None:
        super().__init__(config=config)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        setattr(self.weight, "sequence_parallel", config.sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(x, self.weight, self.eps)


class ML4LayerNorm(nn.Module):
    """vLLM ``LayerNorm`` (weight + bias), FP32 reduction (``ops.layer_norm_fp32``)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.layer_norm_fp32(x, self.weight, self.bias, self.eps)


class ML4SSMax(nn.Module):
    """``SSMaxScaling``: q *= s * log(pos + 1) + b per head (``ops.ssmax_scale``)."""

    def __init__(self, num_heads: int, tensor_parallel: bool) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.zeros(num_heads))
        self.b = nn.Parameter(torch.ones(num_heads))
        if tensor_parallel:
            for p in (self.s, self.b):
                set_tensor_model_parallel_attributes(p, True, 0, 1)

    def forward(self, q: torch.Tensor, log_pos: torch.Tensor) -> torch.Tensor:
        # q [1, T, H, D]; log_pos [1, T]
        return ops.ssmax_scale(q, log_pos, self.s, self.b)


# --------------------------------------------------------------------------------------------
# Per-forward context (carried on PackedSeqParams)
# --------------------------------------------------------------------------------------------


# (device, rotary_dim, rope parameters, LeRoPE alpha storage and version) -> inverse frequencies on
# the device. The frequencies are computed on the host (as vLLM does), which moves the alpha buffer
# to the host and the result back: two waits on the device per layer and micro-batch. The alpha
# buffers are frozen, so each layer's frequencies are computed once; an in-place update of an
# alpha bumps its version and recomputes.
_INV_FREQ: dict = {}


def _inv_freq_on(device, rope_parameters, rotary_dim, alpha):
    key = (
        str(device),
        rotary_dim,
        repr(sorted(rope_parameters.items())),
        None if alpha is None else (alpha.data_ptr(), alpha._version, alpha.dtype, tuple(alpha.shape)),
    )
    inv_freq = _INV_FREQ.get(key)
    if inv_freq is None:
        inv_freq = ops.rope_inv_freq(rope_parameters, rotary_dim, alpha).to(device)
        _INV_FREQ[key] = inv_freq
    return inv_freq


class ML4ForwardContext:
    """Positions, sequence boundaries and cross-layer tensors of one packed forward."""

    def __init__(self, packed_seq_params, total: int, device) -> None:
        cu = packed_seq_params.cu_seqlens_q_padded
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        cu = cu.tolist()
        self.total = total
        if cu[-1] > total:
            raise ValueError(f"cu_seqlens end {cu[-1]} > packed length {total}")
        if cu[-1] < total:  # trailing pad of the whole micro-batch: its own segment
            cu = cu + [total]
        self.segments = [(a, b) for a, b in zip(cu[:-1], cu[1:]) if b > a]
        pos = torch.empty(total, dtype=torch.long)
        for a, b in self.segments:
            pos[a:b] = torch.arange(b - a)
        self.positions = pos.to(device)
        # model.py:175 -- log(pos + 1) in fp32 for SSMax
        self.log_pos = torch.log(self.positions.float() + 1)[None]
        self.kv: dict[int, tuple] = {}
        self.sel: Optional[list] = None
        self.sel_layer: Optional[int] = None
        self._cos_sin: dict = {}

    @staticmethod
    def get(packed_seq_params, total: int, device) -> "ML4ForwardContext":
        if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
            raise ValueError("ML4 attention requires THD-packed inputs (packed_seq_params, qkv_format='thd')")
        ctx = getattr(packed_seq_params, "_ml4_ctx", None)
        if ctx is None:
            ctx = ML4ForwardContext(packed_seq_params, total, device)
            packed_seq_params._ml4_ctx = ctx
        return ctx

    def cos_sin(self, rope_parameters, rotary_dim, alpha, dtype):
        key = (dtype, None if alpha is None else alpha.data_ptr())
        if key not in self._cos_sin:
            inv_freq = _inv_freq_on(self.positions.device, rope_parameters, rotary_dim, alpha)
            cos, sin = ops.rope_cos_sin(self.positions[None], inv_freq, dtype)
            self._cos_sin[key] = (cos, sin)  # [1, T, rotary_dim / 2]
        return self._cos_sin[key]


# --------------------------------------------------------------------------------------------
# Lightning indexer
# --------------------------------------------------------------------------------------------


class ML4Indexer(nn.Module):
    """DSA lightning indexer (vllm/models/mistral/nvidia/indexer.py:80-224).

    Same math as the FSDP trainer's ``ML4Indexer``, computed per packed sequence: returns, per
    sequence, the top-k key indices [T, K] (local positions, -1 = unused slot). Runs under
    no_grad (the selection is discrete; no LM-loss gradient reaches the indexer). Replicated
    across tensor-parallel ranks; parameters are frozen.
    """

    def __init__(self, config, use_rope: bool) -> None:
        super().__init__()
        h = config.hidden_size
        self.n_head = config.ml4_index_n_heads
        self.head_dim = config.ml4_index_head_dim
        self.rope_dim = config.ml4_qk_rope_head_dim
        self.topk = config.ml4_index_topk
        self.use_rope = use_rope
        self.fp8 = config.ml4_indexer_fp8_simulation
        self.chunk = config.ml4_attn_query_chunk
        eps = config.ml4_indexer_norm_eps
        dt = config.params_dtype
        self.q_input_norm = ML4LayerNorm(h, eps)
        self.wq_b = nn.Linear(h, self.n_head * self.head_dim, bias=False, dtype=dt)
        # Stacked [wk (head_dim rows); weights_proj (n_head rows)].
        self.wk_weights_proj = nn.Linear(h, self.head_dim + self.n_head, bias=False, dtype=dt)
        self.k_norm = ML4LayerNorm(self.head_dim, eps)
        self.ssmax = ML4SSMax(self.n_head, tensor_parallel=False) if config.ml4_indexer_ssmax_scaling else None
        for p in self.parameters():
            p.requires_grad_(False)

    def _rope_first(self, x, cos, sin):
        # Indexer layout is [rope | nope] (indexer.py:214).
        r = ops.apply_rope_interleaved(x[..., : self.rope_dim], cos, sin)
        return torch.cat([r, x[..., self.rope_dim :]], -1)

    @torch.no_grad()
    def forward(self, x, ctx: ML4ForwardContext, cos, sin) -> list:
        """x: [T, h] (the attention input, i.e. the normed hidden states)."""
        T = x.shape[0]
        xb = x[None]
        q = self.wq_b(self.q_input_norm(xb)).view(1, T, self.n_head, self.head_dim)
        kw = self.wk_weights_proj(xb)
        k = self.k_norm(kw[..., : self.head_dim])
        w = kw[..., self.head_dim :].float()
        if self.use_rope:
            k = self._rope_first(k, cos, sin)
        if self.ssmax is not None:
            q = self.ssmax(q, ctx.log_pos)
        if self.use_rope:
            q = self._rope_first(q, cos, sin)
        w = w * (self.head_dim**-0.5) * (self.n_head**-0.5)
        if self.fp8:
            CALLS["indexer_fp8"] += 1
            q, q_scale = ops.ue8m0_quant(q, "q_rope" if self.use_rope else "q_nope")
            w = w * q_scale[..., 0]
            k8, k_scale = ops.ue8m0_quant(k, "k")
            k = k8 * k_scale
        q, k = q.float(), k.float()
        out = []
        for a, b in ctx.segments:
            n = b - a
            kk = min(self.topk, n)
            idx_all = torch.empty(n, kk, dtype=torch.long, device=x.device)
            causal = torch.ones(n, n, dtype=torch.bool, device=x.device).tril_()
            for t0 in range(0, n, self.chunk):
                t1 = min(n, t0 + self.chunk)
                s = ops.fp32_einsum("bthd,bsd->bths", q[:, a + t0 : a + t1], k[:, a:b]).relu_()
                s = (s * w[:, a + t0 : a + t1, :, None]).sum(2)[0]
                s = s.masked_fill(~causal[t0:t1], float("-inf"))
                # Ties go to the lower key index (stable sort), as in the FSDP trainer.
                vals, idx = s.sort(dim=-1, descending=True, stable=True)
                vals, idx = vals[..., :kk], idx[..., :kk]
                idx_all[t0:t1] = torch.where(vals > float("-inf"), idx, -1)
            out.append(idx_all)
        return out


# --------------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------------


class ML4Attention(MegatronModule):
    """ML4 MLA + DSA (vllm/models/mistral/nvidia/attention.py), Megatron TP layout.

    ``linear_q_proj`` / ``linear_kv_up_proj`` are column-parallel over heads, ``linear_proj``
    row-parallel, ``linear_kv_down_proj`` (+ ``kv_layernorm``) replicated: the latent is shared
    by all heads. KV-sharing layers have no ``linear_kv_down_proj`` and attend to the source
    layer's (kv_c_normed, k_pe), with their own ``linear_kv_up_proj``. NoPE layers: q is
    all-nope (qk_head_dim wide), no k_pe. Softmax scale is qk_head_dim**-0.5 (attention.py:136).

    The attention core is the FSDP trainer's: non-absorbed (k_nope/v from kv_up(kv_c)), FP32
    scores masked to the indexer's selection, FP32 softmax and PV, chunked over queries,
    per packed sequence.
    """

    def __init__(self, config, submodules, layer_number: int, attn_mask_type=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        if config.sequence_parallel and config.tensor_model_parallel_size > 1:
            raise NotImplementedError("ML4Attention: sequence parallelism is not supported yet")
        if (config.context_parallel_size or 1) > 1:
            raise NotImplementedError("ML4Attention: context parallelism is not supported")
        i = layer_number - 1
        self.layer_idx = i
        self.use_rope = not config.ml4_nope_layer_flags[i]
        self.kv_source = config.ml4_kv_source_layers[i]
        self.shares_kv = self.kv_source != i
        if config.ml4_nope_layer_flags[self.kv_source] != config.ml4_nope_layer_flags[i]:
            raise ValueError(f"layer {i}: RoPE/NoPE differs from its KV source layer {self.kv_source}")
        self.kv_needed_later = any(s == i and j != i for j, s in enumerate(config.ml4_kv_source_layers))
        self.use_sparse = config.ml4_index_topk is not None
        self.has_indexer = self.use_sparse and config.ml4_index_sharing[i] == "full"
        tp = parallel_state.get_tensor_model_parallel_world_size()
        H = config.num_attention_heads
        if H % tp:
            raise ValueError(f"num_attention_heads {H} not divisible by TP {tp}")
        self.H = H // tp
        self.kv_lora_rank = config.ml4_kv_lora_rank
        self.rope_dim = config.ml4_qk_rope_head_dim
        self.qk_head_dim = config.ml4_qk_nope_head_dim + config.ml4_qk_rope_head_dim
        self.nope_dim = config.ml4_qk_nope_head_dim if self.use_rope else self.qk_head_dim
        self.v_dim = config.ml4_v_head_dim
        self.scale = self.qk_head_dim**-0.5
        self.chunk = config.ml4_attn_query_chunk
        self.fp8_kv = bool(config.ml4_fp8_kv_cache)
        self.rope_parameters = config.ml4_rope_parameters
        self.lerope = config.ml4_lerope
        self.recompute_core = config.recompute_granularity == "selective" and "core_attn" in (
            config.recompute_modules or []
        )
        h = config.hidden_size
        tp_group = pg_collection.tp if pg_collection is not None else None
        self.linear_q_proj = build_module(
            submodules.linear_q_proj,
            h,
            H * self.qk_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_proj",
            tp_group=tp_group,
        )
        if not self.shares_kv:
            self.linear_kv_down_proj = build_module(
                submodules.linear_kv_down_proj,
                h,
                self.kv_lora_rank + (self.rope_dim if self.use_rope else 0),
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_down_proj",
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
            self.kv_layernorm = ML4RMSNorm(config, self.kv_lora_rank, config.layernorm_epsilon)
        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.kv_lora_rank,
            H * (self.nope_dim + self.v_dim),
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
            tp_group=tp_group,
        )
        self.linear_proj = build_module(
            submodules.linear_proj,
            H * self.v_dim,
            h,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=tp_group,
        )
        self.ssmax = ML4SSMax(self.H, tensor_parallel=True) if config.ml4_ssmax_scaling else None
        self.indexer = ML4Indexer(config, not config.ml4_indexer_nope_layer_flags[i]) if self.has_indexer else None
        # LeRoPE alpha (checkpoint ``lerope.alpha``, one per frequency, shared by all layers and
        # indexers; rope.py:70-136). Each layer holds a copy, loaded from the same tensor.
        self.register_buffer("lerope_alpha", torch.zeros(self.rope_dim // 2), persistent=True)

    # ---- helpers ------------------------------------------------------------------------

    def _latent(self, x, cos, sin):
        """This layer's KV-cache content: (kv_c_normed, roped k_pe or None); x [1, T, h]."""
        kv, _ = self.linear_kv_down_proj(x)
        kv_c = self.kv_layernorm(kv[..., : self.kv_lora_rank])
        if self.fp8_kv:  # straight-through: identity gradient, as the activation QAT
            CALLS["fp8_kv_latent"] += 1
            kv_c = ste(kv_c, ops.fp8_ds_mla_latent(kv_c).to(kv_c.dtype))
        k_pe = None
        if self.use_rope:
            pe = kv[..., self.kv_lora_rank :]
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                pe = copy_to_tensor_model_parallel_region(pe)
            k_pe = ops.apply_rope_interleaved(pe, cos, sin)
        return kv_c, k_pe

    def _core(self, q, kv_c, k_pe, sel, segments, dtype):
        """q [1, T, H, qk] (roped, SSMax-scaled), kv_c [1, T, 512], k_pe [1, T, 64] or None,
        sel: per-segment top-k indices or None (dense causal) -> [T, H * v_dim] in ``dtype``.

        Each (sequence, query chunk) piece is checkpointed when grad is on: its FP32 scores and
        probabilities are recomputed in backward instead of being kept for every chunk of every
        layer (O(chunk x T x H) per piece instead of O(T^2 x H) per layer). Same ops and values."""
        T = q.shape[1]
        kvb, _ = self.linear_kv_up_proj(kv_c)
        kvb = kvb.view(1, T, self.H, self.nope_dim + self.v_dim)
        k_nope, v = kvb[..., : self.nope_dim], kvb[..., self.nope_dim :]
        ckpt = torch.is_grad_enabled()
        outs = []
        for si, (a, b) in enumerate(segments):
            n = b - a
            causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril_()
            ks, vs = k_nope[:, a:b], v[:, a:b]
            pes = k_pe[:, a:b] if self.use_rope else None
            for t0 in range(0, n, self.chunk):
                t1 = min(n, t0 + self.chunk)
                if sel is None:
                    allowed = causal[t0:t1]
                else:
                    idx = sel[si][t0:t1]
                    allowed = torch.zeros(t1 - t0, n + 1, dtype=torch.bool, device=q.device)
                    allowed.scatter_(1, torch.where(idx >= 0, idx, n), True)
                    allowed = allowed[:, :n]
                qc = q[:, a + t0 : a + t1]
                if ckpt:
                    o = torch.utils.checkpoint.checkpoint(
                        self._chunk, qc, ks, vs, pes, allowed, dtype, use_reentrant=False
                    )
                else:
                    o = self._chunk(qc, ks, vs, pes, allowed, dtype)
                outs.append(o)
        return torch.cat(outs, 1).reshape(T, self.H * self.v_dim)

    def _chunk(self, qc, ks, vs, pes, allowed, dtype):
        """One query chunk of one sequence: FP32 scores, mask, FP32 softmax and PV (the FSDP
        trainer's attention core), output in ``dtype``."""
        qf = qc.float()
        s = ops.fp32_einsum("bthd,bshd->bhts", qf[..., : self.nope_dim], ks.float())
        if self.use_rope:
            s = s + ops.fp32_einsum("bthd,bsd->bhts", qf[..., self.nope_dim :], pes.float())
        s = (s * self.scale).masked_fill(~allowed[None, None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        return ops.fp32_einsum("bhts,bshd->bthd", p, vs.float()).to(dtype)

    # ---- forward ------------------------------------------------------------------------

    def forward(self, hidden_states, attention_mask=None, packed_seq_params=None, **kwargs):
        """hidden_states [T, 1, h] (THD-packed) -> (output [T, 1, h], None)."""
        T, B, h = hidden_states.shape
        if B != 1:
            raise ValueError(f"ML4Attention expects a THD-packed micro-batch (dim 1 == 1), got {B}")
        ctx = ML4ForwardContext.get(packed_seq_params, T, hidden_states.device)
        dtype = hidden_states.dtype
        x = hidden_states.transpose(0, 1)  # [1, T, h]
        alpha = self.lerope_alpha if self.lerope else None
        cos, sin = ctx.cos_sin(self.rope_parameters, self.rope_dim, alpha, dtype)

        q, _ = self.linear_q_proj(x)
        q = q.view(1, T, self.H, self.qk_head_dim)
        if self.shares_kv:
            if self.kv_source not in ctx.kv:
                raise RuntimeError(
                    f"layer {self.layer_idx}: KV source layer {self.kv_source} did not run in this forward "
                    "(it must be on the same pipeline stage)"
                )
            kv_c, k_pe = ctx.kv[self.kv_source]
        else:
            kv_c, k_pe = self._latent(x, cos, sin)
            if self.kv_needed_later:
                ctx.kv[self.layer_idx] = (kv_c, k_pe)
        if self.use_rope:
            q_pe = ops.apply_rope_interleaved(q[..., self.nope_dim :], cos, sin)
            q = torch.cat([q[..., : self.nope_dim], q_pe], -1)
        if self.ssmax is not None:
            q = self.ssmax(q, ctx.log_pos)
        sel = None
        if self.use_sparse:
            if self.indexer is not None:
                ctx.sel = self.indexer(x[0], ctx, cos, sin)
                ctx.sel_layer = self.layer_idx
            elif ctx.sel is None:
                raise RuntimeError(
                    f"layer {self.layer_idx}: shared-index layer without a preceding indexer layer in this "
                    "forward (it must be on the same pipeline stage)"
                )
            sel = ctx.sel
        if self.recompute_core and torch.is_grad_enabled():
            core = torch.utils.checkpoint.checkpoint(
                self._core, q, kv_c, k_pe, sel, ctx.segments, dtype, use_reentrant=False
            )
        else:
            core = self._core(q, kv_c, k_pe, sel, ctx.segments, dtype)
        out, bias = self.linear_proj(core[:, None])
        return out, bias


# --------------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------------


class ML4Router(TopKRouter):
    """``MistralMoE`` routing (moe.py) on Megatron's ``TopKRouter``.

    Logits: FP32 GEMM of the BF16 input and router weight; with ``ml4_router_logits_bf16`` they
    are rounded to BF16 (straight-through gradient), as vLLM's ``GateLinear`` returns them.
    Softmax over all experts in FP32, top-k by a stable descending sort (ties to the lower
    expert index, topk_softmax_kernels.cu:536-537), renormalized, times ``routed_scale``.
    The top-k selection goes through ``RouterReplay.get_replay_topk`` so R3 replays the
    rollout's experts (weights are this router's probs at the replayed ids).
    """

    def gating(self, input: torch.Tensor):
        with torch.autocast(input.device.type, enabled=False):
            logits = F.linear(input.float(), self.weight.float())
        if self.config.ml4_router_logits_bf16:
            logits = ste(logits, logits.to(torch.bfloat16).float())
        return logits

    @staticmethod
    def _stable_topk(scores, topk, num_groups=None, group_topk=None):
        vals, idx = scores.sort(dim=-1, descending=True, stable=True)
        return vals[..., :topk], idx[..., :topk]

    def routing(self, logits: torch.Tensor, padding_mask=None, input_ids=None):
        logits = logits.view(-1, self.config.num_moe_experts)
        probs = torch.softmax(logits.float(), dim=-1)
        w, ids = self._stable_topk(probs, self.topk)
        self.ml4_native_topk = ids.detach()
        rr = self.router_replay
        if rr is not None and rr.router_replay_action in (
            RouterReplayAction.REPLAY_FORWARD,
            RouterReplayAction.REPLAY_BACKWARD,
        ):
            # RouterReplay.get_replay_topk would gather at the replayed ids, which fails on rows of -1;
            # take the ids the same way it does (forward target / backward FIFO) and gather here.
            if rr.router_replay_action == RouterReplayAction.REPLAY_FORWARD:
                rids = rr.target_topk_idx
                CALLS["r3_replay_forward"] += 1
            else:
                rids = rr.replay_backward_list.pop(0)
                CALLS["r3_replay_backward"] += 1
            rids = rids.to(probs.device).long()
            # rows of -1 (no rollout record) keep this router's own top-k
            ids = torch.where(rids >= 0, rids, ids)
            w = probs.gather(1, ids)
        elif rr is not None:
            rr.get_replay_topk(probs, self.topk, None, None, self._stable_topk)  # RECORD / no action
        w = w / w.sum(-1, keepdim=True)
        scale = self.config.moe_router_topk_scaling_factor
        if scale:
            w = w * scale
        routing_probs = torch.zeros_like(probs).scatter(1, ids, w)
        routing_map = torch.zeros_like(probs, dtype=torch.bool).scatter(1, ids, True)
        return routing_probs, routing_map


# --------------------------------------------------------------------------------------------
# Output layer
# --------------------------------------------------------------------------------------------


def ml4_logit_cap(logits: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """``tanh(logits / cap) * cap`` in FP32 (LogitsProcessor soft_cap, model.py:359-361)."""
    logits = logits.float()
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


class ML4OutputLayer(ColumnParallelLinear):
    """Mixin applied with ``extend_instance``: the output layer returns FP32 capped logits."""

    def forward(self, *args, **kwargs):
        logits, bias = super().forward(*args, **kwargs)
        return ml4_logit_cap(logits, self.config.final_logit_softcapping), bias
