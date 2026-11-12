"""Megatron-Core modules for the Inkling2 language model.

Inkling2 replaces the residual stream with block attention residuals (attn-res): every sub-layer
reads a softmax-over-depth mixture of the completed block outputs and the running partial block
sum, and a new block starts every ``n_layers_per_attn_res_block`` layers. ``TransformerBlock`` passes
one tensor between layers, so the attn-res state travels as ``[s, b, (n_blocks + 1) * hidden]``:
the completed blocks followed by the partial sum (each ``[s, b, hidden]``). The embedding output
enters layer 0 as the partial sum of an empty state, and :class:`Inkling2FinalNorm` (the block's
``final_layernorm``) reads the output mixture.

Per layer (``Inkling2TransformerLayer``)::

    a_in  = norm(attn_res(state, attn_query))       # global layers: separate q/k/v mixtures + norms
    a     = attn_sconv(Inkling2Attention(a_in))
    m     = mlp_sconv(MoE(mlp_norm(attn_res(state + a, mlp_query))))
    state = (blocks + [partial], a + m) at a block boundary, else (blocks, partial + a + m)

The math follows the SGLang fork's ``sglang/tml`` model; ``reference.py`` in this package is the
plain-PyTorch transcription it is tested against.

Tensor parallelism shards attention by heads and requires sequence parallelism: between
sub-layers each rank holds a contiguous slice of the sequence, the column-parallel projections
gather the full sequence for attention (whose k / v convolutions are per channel, so they shard
with the heads), and the convolutions on attention / MoE outputs read the last ``W - 1`` tokens
of the previous rank's slice. Context and pipeline parallelism are not supported.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    set_tensor_model_parallel_attributes,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from torch import Tensor, nn

ATTN_RES_EPS = 1e-6
L2_CLIP_EPS = 1e-6
FLEX_BLOCK = 128
# Past this length the padded length grows in steps of FLEX_BUCKET instead of FLEX_BLOCK, which bounds
# the number of compiled shapes (each one compiles separately: dynamic=False).
FLEX_BUCKET = 1024
# FlexAttention compiles once per (padded length, window, grad mode): <= 8 lengths up to FLEX_BUCKET, then
# one per FLEX_BUCKET, for local and global layers. Past torch.compile's recompile limit (8 by default) it
# falls back to eager, which materializes the [s, s] scores per head and runs out of memory on long
# sequences (flex_core_attention re-asserts the limit before each call).
FLEX_RECOMPILE_LIMIT = 256

try:
    import torch._dynamo
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, FLEX_RECOMPILE_LIMIT)
    HAVE_FLEX = True
except ImportError:  # pragma: no cover
    HAVE_FLEX = False


def _assert_recompile_limits() -> None:
    """Re-assert dynamo's recompile limits: something in the training process resets them to their defaults (8)
    after this module is imported, and past them a compiled function silently runs eagerly."""
    import torch._dynamo

    cfg = torch._dynamo.config
    if cfg.recompile_limit < FLEX_RECOMPILE_LIMIT:
        cfg.recompile_limit = FLEX_RECOMPILE_LIMIT
    if cfg.accumulated_recompile_limit < 4 * FLEX_RECOMPILE_LIMIT:
        cfg.accumulated_recompile_limit = 4 * FLEX_RECOMPILE_LIMIT


def fused(fn):
    """``fn`` (a function of tensors, elementwise and small reductions) compiled into fused Triton kernels.

    The eager Inkling2 pieces are long chains of FP32 elementwise ops: one kernel and one full read and write per op,
    which made them ~45% of a training step's GPU time on the 8-layer slice and left the GPU idle on kernel launches.
    Compiled with dynamic shapes (microbatch lengths vary); FP32 math as in eager, so the outputs agree up to
    reassociation. The eager function stays available as ``fn.eager`` (``inkling2_fuse_elementwise=False``).
    """
    compiled = torch.compile(fn, dynamic=True)

    def call(*args, **kwargs):
        _assert_recompile_limits()
        return compiled(*args, **kwargs)

    call.eager = fn
    call.__name__ = fn.__name__
    call.__doc__ = fn.__doc__
    return call


def pick(config: TransformerConfig, fn):
    """The fused or the eager version of a :func:`fused` function, per ``config.inkling2_fuse_elementwise``."""
    return fn if getattr(config, "inkling2_fuse_elementwise", False) else fn.eager


def is_local_layer(config: TransformerConfig, layer_idx: int) -> bool:
    """Whether 0-based layer ``layer_idx`` uses sliding-window attention."""
    return layer_idx in config.local_layer_ids


def is_block_boundary(config: TransformerConfig, layer_idx: int) -> bool:
    return layer_idx % config.n_layers_per_attn_res_block == 0


# --------------------------------------------------------------------------------------------
# Sequence layout
# --------------------------------------------------------------------------------------------


class SequenceLayout:
    """Per-token sequence ids and positions for ``[s, b, ...]`` activations.

    Packed (THD) input has ``b == 1`` and sequence boundaries ``cu_seqlens`` (padded offsets when
    present); otherwise every batch row is one sequence starting at position 0. ``s`` is the full
    sequence length; under sequence parallelism rank ``sp_rank`` holds tokens
    ``[sp_rank * s / sp_size, (sp_rank + 1) * s / sp_size)`` between sub-layers.
    """

    def __init__(
        self,
        s: int,
        b: int,
        device: torch.device,
        packed_seq_params: Optional[PackedSeqParams],
        sp_size: int = 1,
        sp_rank: int = 0,
    ):
        self.s, self.b = s, b
        self.sp_size, self.sp_rank = sp_size, sp_rank
        cu = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cu = packed_seq_params.cu_seqlens_q_padded
            if cu is None:
                cu = packed_seq_params.cu_seqlens_q
        if cu is not None:
            if b != 1:
                raise ValueError("Packed Inkling2 input expects batch dimension 1.")
            cu = cu.to(device=device, dtype=torch.long)
            tokens = torch.arange(s, device=device)
            # Tokens past the last boundary (padding) form their own sequence.
            seq = torch.searchsorted(cu[1:], tokens, right=True)
            starts = torch.cat([cu, cu.new_tensor([s])])[seq]
            self.seq_ids = seq.view(s, 1)
            self.positions = (tokens - starts).view(s, 1)
        else:
            self.seq_ids = None
            self.positions = torch.arange(s, device=device).view(s, 1).expand(s, b)
        self._block_masks = {}

    @property
    def local_positions(self) -> Tensor:
        """Positions of this rank's sequence-parallel slice."""
        if self.sp_size == 1:
            return self.positions
        n = self.s // self.sp_size
        return self.positions[self.sp_rank * n : (self.sp_rank + 1) * n]

    @property
    def flex_len(self) -> int:
        """FlexAttention runs on the sequence padded to a multiple of ``FLEX_BLOCK`` (for lengths in
        (64, 128) torch's block mask gets a block size no kernel config divides: no valid choices
        at lowering), and past ``FLEX_BUCKET`` to a multiple of ``FLEX_BUCKET``, which bounds the
        number of compiled shapes; padding rows are masked out."""
        step = FLEX_BLOCK if self.s <= FLEX_BUCKET else FLEX_BUCKET
        return -(-self.s // step) * step

    def block_mask(self, window: Optional[int]):
        """FlexAttention mask over ``flex_len`` tokens: causal, same sequence, and
        ``q - kv < window`` when set. The padding is a sequence of its own."""
        if window not in self._block_masks:
            device = self.positions.device
            seq = torch.zeros(self.flex_len, dtype=torch.long, device=device)
            if self.seq_ids is not None:
                seq[: self.s] = self.seq_ids.view(-1)
            seq[self.s :] = -1

            def mask_mod(b, h, q_idx, kv_idx):
                keep = (q_idx >= kv_idx) & (seq[q_idx] == seq[kv_idx])
                if window is not None:
                    keep = keep & (q_idx - kv_idx < window)
                return keep

            self._block_masks[window] = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=self.flex_len, KV_LEN=self.flex_len, device=device
            )
        return self._block_masks[window]


def get_layout(
    hidden_states: Tensor, packed_seq_params: Optional[PackedSeqParams], config: TransformerConfig, tp_group
) -> SequenceLayout:
    """The layout shared by all layers of one forward pass (cached on ``packed_seq_params``).
    ``hidden_states`` is this rank's sequence-parallel slice when sequence parallelism is on."""
    sp_size = tp_group.size() if config.sequence_parallel else 1
    sp_rank = tp_group.rank() if sp_size > 1 else 0
    s, b = hidden_states.shape[0] * sp_size, hidden_states.shape[1]
    if packed_seq_params is not None:
        layout = getattr(packed_seq_params, "_inkling2_layout", None)
        if layout is None or (layout.s, layout.b) != (s, b):
            layout = SequenceLayout(s, b, hidden_states.device, packed_seq_params, sp_size, sp_rank)
            packed_seq_params._inkling2_layout = layout
        return layout
    return SequenceLayout(s, b, hidden_states.device, None, sp_size, sp_rank)


def mark_sequence_parallel(*params) -> None:
    """Parameters replicated across TP ranks whose gradients are partial per rank (each rank sees
    a slice of the sequence or of the heads): Megatron all-reduces them across TP."""
    for p in params:
        p.sequence_parallel = True


def mark_tensor_parallel(*params, dim: int = 0) -> None:
    """Parameters split across TP ranks along ``dim`` (excluded from TP-duplicated grad norms)."""
    for p in params:
        set_tensor_model_parallel_attributes(p, is_parallel=True, dim=dim, stride=1)


class TPShardedParamsCheckpointMixin:
    """Save this module's own TP-split params (``mark_tensor_parallel``) as TP-sharded checkpoint tensors.

    ``MegatronModule.sharded_state_dict`` wraps a module's direct params as TP-replicated, so only TP rank 0's
    slice of a split param was saved and every other TP rank got rank 0's slice back on resume.
    """

    def sharded_state_dict(self, prefix: str = "", sharded_offsets=(), metadata=None):
        from megatron.core import parallel_state
        from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

        sharded = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        tp_group = getattr(self, "tp_group", None) or parallel_state.get_tensor_model_parallel_group()
        dp_cp_group = (metadata or {}).get("dp_cp_group") or parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        for name, param in self.named_parameters(recurse=False):
            if getattr(param, "tensor_model_parallel", False):
                key = f"{prefix}{name}"
                sharded[key] = make_tp_sharded_tensor_for_checkpoint(
                    param,
                    key,
                    tp_axis=param.partition_dim,
                    prepend_offsets=sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
        return sharded


# --------------------------------------------------------------------------------------------
# Elementwise pieces
# --------------------------------------------------------------------------------------------


class Inkling2RMSNorm(MegatronModule):
    """RMSNorm computed in FP32 with a single rounding to ``out_dtype`` (BF16 by default).

    Accepts FP32 input (attn-res mixtures) as well as activation-dtype input.
    """

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(config=config)
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(
                torch.ones(hidden_size, dtype=config.params_dtype, device=torch.cuda.current_device())
            )
            if config.sequence_parallel:
                self.weight.sequence_parallel = True
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor, out_dtype: Optional[torch.dtype] = None) -> Tensor:
        return pick(self.config, rms_norm)(x, self.weight, self.eps, out_dtype or self.config.params_dtype)


@fused
def rms_norm(x: Tensor, weight: Optional[Tensor], eps: float, out_dtype: torch.dtype) -> Tensor:
    u = x.float()
    y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype)


@fused
def block_attn_res(sources: Sequence[Tensor], addend: Optional[Tensor], queries: Tensor, normalize_values: bool):
    """Softmax over depth of RMS-normalized ``sources`` (``[s, b, h]`` each), one mixture per row
    of ``queries`` (``[q, h]`` or ``[h]``, FP32). ``addend`` is added to the last source in FP32.
    Returns a tuple of FP32 mixtures ``[s, b, h]``.
    """
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    hidden = sources[0].shape[-1]
    values = torch.stack(list(sources)).float()  # [n, s, b, h]
    if addend is not None:
        values = torch.cat([values[:-1], (values[-1] + addend.float()).unsqueeze(0)])
    inv_rms = torch.rsqrt(values.square().mean(-1) + ATTN_RES_EPS)  # [n, s, b]
    logits = torch.einsum("nsbh,qh,nsb->qsbn", values, queries.float() * hidden**-0.5, inv_rms)
    probs = torch.softmax(logits, dim=-1)
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return tuple(torch.einsum("qsbn,nsbh->qsbh", probs, values).unbind(0))


@fused
def l2_clip(x: Tensor, threshold: float, scale: Optional[Tensor] = None, output_scale: Optional[Tensor] = None):
    """``x [..., heads, d]``: optional per-head scale, then clip each vector's L2 norm to
    ``threshold``; ``output_scale`` (per token, broadcast over heads) is applied after a rounding."""
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    out = (value * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        out = (out.float() * output_scale[..., None, None]).to(x.dtype)
    return out


def log_scaling_tau(positions: Tensor, n_floor: float, alpha: float) -> Tensor:
    """Attention length scaling of the global layers; exactly 1 for ``positions < n_floor``."""
    return 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))


def _shift(x: Tensor, lag: int, positions: Tensor, halo: Optional[Tensor] = None) -> Tensor:
    """``x[t - lag]`` along dim 0 of ``[s, b, c]``, zero where ``t - lag`` precedes the start of
    ``t``'s sequence. ``halo`` holds the tokens before ``x[0]`` (the previous sequence-parallel
    slice's last rows), zeros if None."""
    if lag == 0:
        return x
    if halo is None:
        halo = x.new_zeros(lag, *x.shape[1:])
    # The first ``s`` rows of [halo tail; x]: also right for a slice shorter than ``lag``.
    shifted = torch.cat([halo[halo.shape[0] - lag :], x])[: x.shape[0]]
    return shifted * (positions >= lag).unsqueeze(-1)


def previous_slice_tail(x: Tensor, n: int, tp_group) -> Tensor:
    """The last ``n`` rows of the previous rank's sequence-parallel slice (zeros on rank 0).
    Differentiable: the gather's backward reduce-scatters the gradient back to the owner. Rank 0
    keeps the gathered tensor in its graph too (times zero): otherwise autograd skips the
    gather's backward there and the other ranks' reduce-scatter waits forever."""
    # A slice shorter than ``n`` (a few-token microbatch split over TP) sends what it has; every
    # rank's slice has the same length, so the gather stays uniform. The missing older rows are
    # zeros: exact whenever slices hold at least ``n`` rows.
    k = min(n, x.shape[0])
    tails = gather_from_sequence_parallel_region(x[x.shape[0] - k :].contiguous(), group=tp_group)
    rank = tp_group.rank()
    tail = tails[:k] * 0 if rank == 0 else tails[(rank - 1) * k : rank * k]
    if k < n:
        tail = torch.cat([tail.new_zeros(n - k, *tail.shape[1:]), tail])
    return tail


class Inkling2ShortConv(TPShardedParamsCheckpointMixin, MegatronModule):
    """Causal depthwise convolution with identity residual (``tml/kernels/sconv.py``).

    Static: ``y = (x + sum_tap w[c, tap] * x[t - (W - 1 - tap)]) / sqrt(W + 1)`` with
    ``weight [C, 1, W]``. Dynamic (global-layer K/V): per-token, per-head weights
    ``static_w[c, lag] + weight_proj(source)[t, c // head_dim, lag]`` scaled by
    ``1 / sqrt(2 W + 1)`` (``static_w`` is channel-first here; the checkpoint stores it
    ``[W, C]``). FP32 accumulation, output in the input dtype.

    ``mode`` says how its input is distributed over TP ranks: ``"replicated"`` (full sequence,
    all channels), ``"heads"`` (full sequence, this rank's heads: attention k / v, weights split
    with the heads) or ``"sequence"`` (this rank's sequence-parallel slice, all channels: the
    convolutions on attention / MoE outputs, which read the previous slice's last rows).
    """

    def __init__(
        self,
        config: TransformerConfig,
        channels: int,
        tp_group,
        mode: str = "replicated",
        dynamic_heads: Optional[int] = None,
        weight_proj_spec=None,
    ):
        super().__init__(config=config)
        W = config.sconv_kernel_size
        self.width = W
        self.tp_group = tp_group
        self.dynamic = dynamic_heads is not None
        self.sequence_sharded = mode == "sequence" and config.sequence_parallel and tp_group.size() > 1
        device, dtype = torch.cuda.current_device(), config.params_dtype
        if self.dynamic:
            self.n_heads = dynamic_heads // tp_group.size()
            self.static_w = nn.Parameter(torch.zeros(channels, W, dtype=dtype, device=device))
            # Column-parallel over kv heads: rows are (head, lag) head-major.
            self.weight_proj = build_module(
                weight_proj_spec,
                config.hidden_size,
                dynamic_heads * W,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=tp_group,
            )
            mark_tensor_parallel(self.static_w)
        else:
            weight = torch.zeros(channels, 1, W, dtype=dtype, device=device)
            weight[:, 0, -1] = 1.0
            self.weight = nn.Parameter(weight)
            if mode == "heads":
                mark_tensor_parallel(self.weight)
            elif self.sequence_sharded:
                mark_sequence_parallel(self.weight)

    def forward(
        self, x: Tensor, positions: Tensor, source: Optional[Tensor] = None, dyn: Optional[Tensor] = None
    ) -> Tensor:
        """``dyn`` (dynamic convolutions): ``weight_proj(source)``, when the caller already projected it."""
        halo = previous_slice_tail(x.float(), self.width - 1, self.tp_group) if self.sequence_sharded else None
        if self.dynamic:
            if dyn is None:
                dyn, _ = self.weight_proj(source.to(x.dtype))  # BF16 GEMM, this rank's heads
            return pick(self.config, short_conv_dynamic)(x, self.static_w, dyn, positions, halo)
        return pick(self.config, short_conv_static)(x, self.weight, positions, halo)


@fused
def short_conv_static(x: Tensor, weight: Tensor, positions: Tensor, halo: Optional[Tensor]) -> Tensor:
    """``(x + sum_tap w[c, tap] x[t - (W - 1 - tap)]) / sqrt(W + 1)``, ``weight [C, 1, W]``."""
    W = weight.shape[-1]
    xf = x.float()
    halo = None if halo is None else halo.float()
    w = weight.squeeze(1).to(x.dtype).float()  # [C, W]
    acc = torch.zeros_like(xf)
    for tap in range(W):
        acc = acc + _shift(xf, W - 1 - tap, positions, halo) * w[:, tap]
    return ((acc + xf) * (W + 1) ** -0.5).to(x.dtype)


@fused
def short_conv_dynamic(x: Tensor, static_w: Tensor, dyn: Tensor, positions: Tensor, halo: Optional[Tensor]) -> Tensor:
    """Per-token, per-head taps ``static_w[c, lag] + dyn[t, c // head_dim, lag]`` over ``x [s, b, C]``;
    ``dyn [s, b, heads * W]`` (head-major), ``static_w [C, W]``."""
    s, b, C = x.shape
    W = static_w.shape[-1]
    n_heads = dyn.shape[-1] // W
    xf = x.float()
    halo = None if halo is None else halo.float()
    dyn = dyn.float().view(s, b, n_heads, 1, W)
    dyn = dyn.expand(s, b, n_heads, C // n_heads, W).reshape(s, b, C, W)
    acc = torch.zeros_like(xf)
    for lag in range(W):
        acc = acc + _shift(xf, lag, positions, halo) * (static_w[:, lag].float() + dyn[..., lag])
    return ((acc + xf) * (2 * W + 1) ** -0.5).to(x.dtype)


# --------------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------------


def dense_attention(q, k, v, rel, layout: SequenceLayout, window: Optional[int]):
    """FP32 reference attention for testing. ``q [b, hq, s, d]``, ``k/v [b, hkv, s, d]``,
    ``rel [b, hq, s, E]``; returns ``(out [b, hq, s, d] FP32, lse [b, hq, s] FP32)``."""
    b, hq, s, d = q.shape
    group = hq // k.shape[1]
    kf, vf = (t.float().repeat_interleave(group, dim=1) for t in (k, v))
    scores = (q.float() @ kf.transpose(-1, -2)) * d**-0.5  # [b, h, s, s]
    i = torch.arange(s, device=q.device)
    dist = i[:, None] - i[None, :]
    extent = rel.shape[-1]
    idx = dist.clamp(0, extent - 1).expand(b, hq, s, s)
    bias = rel.float().gather(-1, idx).masked_fill(~((dist >= 0) & (dist < extent)), 0.0)
    keep = dist >= 0
    if window is not None:
        keep = keep & (dist < window)
    if layout.seq_ids is not None:
        seq = layout.seq_ids.view(-1)
        keep = keep & (seq[:, None] == seq[None, :])
    scores = (scores + bias).masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return torch.exp(scores - lse.unsqueeze(-1)) @ vf, lse


def flex_core_attention(
    q, k, v, rel, layout: SequenceLayout, window: Optional[int], fp32: bool = False, padded: bool = False
):
    """FlexAttention with the sheared relative bias ``rel[b, h, q, q - kv]`` (zero outside
    ``[0, E)``) added to the scaled score, causal / same-sequence / sliding-window block mask.
    ``fp32`` runs the kernel on FP32 inputs, giving an FP32 output like the serving kernel.
    ``padded``: the inputs are already ``layout.flex_len`` long and in the kernel's dtype (:func:`flex_inputs`)."""
    extent = rel.shape[-1]

    def score_mod(score, b, h, q_idx, kv_idx):
        dist = q_idx - kv_idx
        bias = rel[b, h, q_idx, torch.clamp(dist, 0, extent - 1)]
        return score + torch.where((dist >= 0) & (dist < extent), bias, torch.zeros_like(bias))

    s = layout.s
    if not padded:
        if fp32:
            q, k, v = q.float(), k.float(), v.float()
        pad = layout.flex_len - s
        if pad:
            q, k, v, rel = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v, rel))
    # Re-asserted per call: something in the training process resets dynamo's limit to its default
    # (8) after this module is imported, and past it FlexAttention silently runs eagerly.
    if torch._dynamo.config.recompile_limit < FLEX_RECOMPILE_LIMIT:
        torch._dynamo.config.recompile_limit = FLEX_RECOMPILE_LIMIT
    out, lse = _compiled_flex_attention(
        q,
        k,
        v,
        score_mod=score_mod,
        block_mask=layout.block_mask(window),
        enable_gqa=True,
        return_lse=True,
    )
    return out[:, :, :s], lse[:, :, :s]


@dataclass
class Inkling2AttentionSubmodules:
    linear_column: Union[ModuleSpec, type]
    linear_row: Union[ModuleSpec, type]


class Inkling2Attention(TPShardedParamsCheckpointMixin, MegatronModule):
    """Inkling2 attention (``tml/layers/attn.py``), one layer.

    q, r (relative-position query), gate and sink are projected from the query input, k from the
    key input and v from the value input (one shared input in local layers). k / v pass a short
    convolution, q / k / r are L2-clipped (q and r with learned per-head scales; global layers
    scale q and r by the length factor tau), v is RMS-normalized per head without a gain. Scores
    are ``q.k / sqrt(d) + rel[q, h, q - kv]`` with ``rel = r @ rel_proj``; the output is gated
    elementwise by ``sigmoid(gate)`` and per head by ``sigmoid(lse - sink)``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Inkling2AttentionSubmodules,
        layer_number: int,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config=config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.tp_group = pg_collection.tp
        tp = self.tp_group.size()
        if tp > 1 and not config.sequence_parallel:
            raise NotImplementedError("Inkling2 with tensor parallelism requires sequence_parallel=True.")
        if config.num_query_groups % tp:
            raise ValueError(f"tensor_model_parallel_size={tp} must divide num_query_groups={config.num_query_groups}.")
        if pg_collection.cp is not None and pg_collection.cp.size() > 1:
            raise NotImplementedError("Inkling2 attention does not support context parallelism.")
        self.layer_idx = layer_number - 1
        self.is_local = is_local_layer(config, self.layer_idx)
        self.heads = config.num_attention_heads // tp  # this rank's heads
        self.kv_heads = config.num_query_groups // tp
        self.head_dim = config.kv_channels
        self.d_rel = config.d_rel
        self.window = config.sliding_window_size if self.is_local else None
        extent = config.sliding_window_size if self.is_local else config.rel_extent
        H = config.hidden_size
        device, dtype = torch.cuda.current_device(), config.params_dtype

        def column(out_size, sub):
            return build_module(
                submodules.linear_column,
                H,
                out_size,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_group=self.tp_group,
                name=(name + "." + sub) if name is not None else None,
            )

        heads, kv_heads = config.num_attention_heads, config.num_query_groups
        q_size, kv_size = heads * self.head_dim, kv_heads * self.head_dim
        self.linear_q = column(q_size, "linear_q")
        self.linear_k = column(kv_size, "linear_k")
        self.linear_v = column(kv_size, "linear_v")
        self.linear_r = column(heads * self.d_rel, "linear_r")
        self.linear_gate = column(q_size, "linear_gate")
        self.linear_sink = column(heads, "linear_sink")
        self.linear_proj = build_module(
            submodules.linear_row,
            q_size,
            H,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_group=self.tp_group,
            name=(name + ".linear_proj") if name is not None else None,
        )
        self.q_l2_clip_scale = nn.Parameter(torch.ones(self.heads, self.head_dim, dtype=dtype, device=device))
        self.rel_l2_clip_scale = nn.Parameter(torch.ones(self.heads, self.d_rel, dtype=dtype, device=device))
        self.rel_proj = nn.Parameter(torch.zeros(self.d_rel, extent, dtype=dtype, device=device))
        mark_tensor_parallel(self.q_l2_clip_scale, self.rel_l2_clip_scale)
        mark_sequence_parallel(self.rel_proj)
        dynamic = None if self.is_local else kv_heads
        local_kv_size = self.kv_heads * self.head_dim
        for name in ("k_sconv", "v_sconv"):
            conv = Inkling2ShortConv(config, local_kv_size, self.tp_group, "heads", dynamic, submodules.linear_column)
            setattr(self, name, conv)
        self.v_norm = Inkling2RMSNorm(config, self.head_dim, config.layernorm_epsilon, affine=False)

    def _project(self, x: Tensor, linears: Sequence[nn.Module]) -> List[Tensor]:
        """``[linear(x) for linear in linears]`` for column-parallel projections of one input.

        Under sequence parallelism each projection would all-gather ``x`` itself, in forward, in the activation
        recompute and again in backward (for its weight gradient): six gathers of the same tensor per local
        layer. Instead ``x`` is gathered once (whose backward reduce-scatters the summed input gradient) and
        each projection runs as a local GEMM on the gathered input. Only for plain Transformer Engine linears in
        BF16 (not LoRA-wrapped, not FP8 or userbuffer-overlapped), otherwise each projection gathers as before.
        """
        if not self._shared_gather(linears):
            return [linear(x)[0] for linear in linears]
        full = gather_from_sequence_parallel_region(x, group=self.tp_group)
        outs = []
        for linear in linears:
            mode, sp = linear.parallel_mode, linear.sequence_parallel
            linear.parallel_mode, linear.sequence_parallel = None, False
            try:
                outs.append(linear(full)[0])
            finally:
                linear.parallel_mode, linear.sequence_parallel = mode, sp
        return outs

    def _shared_gather(self, linears: Sequence[nn.Module]) -> bool:
        if len(linears) < 2 or not getattr(self.config, "inkling2_shared_sequence_gather", False):
            return False
        if not (self.config.sequence_parallel and self.tp_group.size() > 1) or self.config.fp8:
            return False
        import transformer_engine.pytorch as te

        return all(
            type(linear).__name__ == "TEColumnParallelLinear"
            and isinstance(linear, te.Linear)
            and linear.parallel_mode == "column"
            and linear.sequence_parallel
            and not getattr(linear, "ub_overlap_ag_fprop", False)
            and not getattr(linear, "ub_overlap_ag_dgrad", False)
            for linear in linears
        )

    def forward(self, xq: Tensor, xk: Tensor, xv: Tensor, layout: SequenceLayout) -> Tensor:
        """``x* [s / sp, b, hidden]`` (this rank's sequence slice under sequence parallelism);
        the column-parallel projections gather the full sequence."""
        cfg = self.config
        s, b = layout.s, layout.b
        if self.is_local:  # one input
            q, r, gate, sink, k, v = self._project(
                xq, [self.linear_q, self.linear_r, self.linear_gate, self.linear_sink, self.linear_k, self.linear_v]
            )
            k = self.k_sconv(k, layout.positions)
            v = self.v_sconv(v, layout.positions)
        else:
            q, r, gate, sink = self._project(xq, [self.linear_q, self.linear_r, self.linear_gate, self.linear_sink])
            k, k_dyn = self._project(xk, [self.linear_k, self.k_sconv.weight_proj])
            v, v_dyn = self._project(xv, [self.linear_v, self.v_sconv.weight_proj])
            k = self.k_sconv(k, layout.positions, dyn=k_dyn)
            v = self.v_sconv(v, layout.positions, dyn=v_dyn)
        tau = (
            None if self.is_local else log_scaling_tau(layout.positions, cfg.log_scaling_n_floor, cfg.log_scaling_alpha)
        )
        q, k, v, r = pick(cfg, attention_inputs)(
            q.view(s, b, self.heads, self.head_dim),
            k.view(s, b, self.kv_heads, self.head_dim),
            v.view(s, b, self.kv_heads, self.head_dim),
            r.view(s, b, self.heads, self.d_rel),
            self.q_l2_clip_scale,
            self.rel_l2_clip_scale,
            tau,
            cfg.qk_l2_clip_threshold,
            cfg.rel_l2_clip_threshold,
            self.v_norm.eps,
        )
        if cfg.inkling2_attention_backend in ("flex", "flex_fp32"):
            fp32 = cfg.inkling2_attention_backend == "flex_fp32"
            q, k, v, rel = pick(cfg, flex_inputs)(q, k, v, r, self.rel_proj, layout.flex_len - s, fp32)
            out, lse = flex_core_attention(q, k, v, rel, layout, self.window, fp32=fp32, padded=True)
        else:
            rel = torch.einsum("sbhd,de->sbhe", r, self.rel_proj)
            # [s, b, h, d] -> [b, h, s, d]
            q, k, v, rel = (t.permute(1, 2, 0, 3).contiguous() for t in (q, k, v, rel))
            out, lse = dense_attention(q, k, v, rel, layout, self.window)
        out, _ = self.linear_proj(pick(cfg, gated_attention_output)(out, lse, gate, sink, xq.dtype))
        return out


@fused
def attention_inputs(q, k, v, r, q_scale, r_scale, tau, qk_threshold: float, rel_threshold: float, v_eps: float):
    """L2-clipped q / k / r (q and r with their per-head scales and the length factor ``tau``), and v RMS-normalized
    per head without a gain; ``[s, b, heads, d]`` each, in their input dtype."""
    q = l2_clip.eager(q, qk_threshold, q_scale, tau)
    k = l2_clip.eager(k, qk_threshold)
    v = rms_norm.eager(v, None, v_eps, v.dtype)
    r = l2_clip.eager(r, rel_threshold, r_scale, tau)
    return q, k, v, r


@fused
def flex_inputs(q: Tensor, k: Tensor, v: Tensor, r: Tensor, rel_proj: Tensor, pad: int, fp32: bool):
    """FlexAttention's inputs from ``[s, b, heads, d]`` q / k / v and the relative-position query ``r``: the
    relative bias ``rel = r @ rel_proj``, each moved to ``[b, heads, s, d]`` and padded by ``pad`` rows (q / k / v
    in FP32 when ``fp32``), in one pass instead of an einsum, a contiguous copy and a padded copy each."""
    rel = torch.einsum("sbhd,de->sbhe", r, rel_proj)

    def layout(t: Tensor, upcast: bool) -> Tensor:
        t = t.permute(1, 2, 0, 3)
        if upcast:
            t = t.float()
        return F.pad(t, (0, 0, 0, pad))

    return layout(q, fp32), layout(k, fp32), layout(v, fp32), layout(rel, False)


@fused
def gated_attention_output(out: Tensor, lse: Tensor, gate: Tensor, sink: Tensor, dtype: torch.dtype) -> Tensor:
    """``out [b, h, s, d]`` gated elementwise by ``sigmoid(gate)`` and per head by ``sigmoid(lse - sink)``;
    returns ``[s, b, h * d]`` in ``dtype``."""
    b, h, s, d = out.shape
    out = out.permute(2, 0, 1, 3).float()  # [s, b, h, d]
    lse = lse.permute(2, 0, 1).float()  # [s, b, h]
    out = out * torch.sigmoid(gate.float().view(s, b, h, d))
    out = out * torch.sigmoid(lse - sink.float()).unsqueeze(-1)
    return out.reshape(s, b, h * d).to(dtype)


# --------------------------------------------------------------------------------------------
# Decoder layer and final norm
# --------------------------------------------------------------------------------------------


def split_state(hidden_states: Tensor, hidden_size: int) -> Tuple[List[Tensor], Tensor]:
    chunks = hidden_states.split(hidden_size, dim=-1)
    return list(chunks[:-1]), chunks[-1]


class Inkling2TransformerLayer(TransformerLayer):
    """One Inkling2 decoder layer over the packed attn-res state ``[s, b, (n + 1) * hidden]``.

    ``input_layernorm`` is the attention input norm (the query-input norm in global layers, which
    add ``k_input_layernorm`` / ``v_input_layernorm``); ``pre_mlp_layernorm`` is the MoE input
    norm. ``attn_query`` (absent in layer 0, ``[3, hidden]`` in global layers) and ``mlp_query`` are
    this layer's FP32 attn-res pseudo-queries.
    """

    def __init__(self, config: TransformerConfig, submodules: TransformerLayerSubmodules, layer_number: int = 1, **kw):
        super().__init__(config=config, submodules=submodules, layer_number=layer_number, **kw)
        self.layer_idx = self.layer_number - 1
        self.is_local = is_local_layer(config, self.layer_idx)
        self.is_boundary = is_block_boundary(config, self.layer_idx)
        H, eps = config.hidden_size, config.layernorm_epsilon
        device = torch.cuda.current_device()
        if not self.is_local:
            self.k_input_layernorm = Inkling2RMSNorm(config, H, eps)
            self.v_input_layernorm = Inkling2RMSNorm(config, H, eps)
        if self.layer_idx > 0:
            shape = (H,) if self.is_local else (3, H)
            self.attn_query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32, device=device))
        self.mlp_query = nn.Parameter(torch.zeros(H, dtype=torch.float32, device=device))
        self.attn_sconv = Inkling2ShortConv(config, H, self.tp_group, "sequence")
        self.mlp_sconv = Inkling2ShortConv(config, H, self.tp_group, "sequence")
        for p in (getattr(self, "attn_query", None), self.mlp_query):
            if p is not None:
                # Float16Module would otherwise cast these to the activation dtype.
                p.keep_in_fp32 = True
                if config.sequence_parallel:
                    mark_sequence_parallel(p)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        cfg = self.config
        H, normalize = cfg.hidden_size, cfg.attn_res_normalize_values
        layout = get_layout(hidden_states, packed_seq_params, cfg, self.tp_group)
        blocks, partial = split_state(hidden_states, H)

        n_streams = 1 if self.is_local else 3
        if not blocks:
            streams = (partial,) * n_streams
        else:
            streams = pick(cfg, block_attn_res)([*blocks, partial], None, self.attn_query, normalize)
        if self.is_local:
            xq = xk = xv = self.input_layernorm(streams[0])
        else:
            xq = self.input_layernorm(streams[0])
            xk = self.k_input_layernorm(streams[1])
            xv = self.v_input_layernorm(streams[2])
        attn = self.attn_sconv(self.self_attention(xq, xk, xv, layout), layout.local_positions)

        attn_res = pick(cfg, block_attn_res)
        if self.is_boundary:
            mix = attn_res([*blocks, partial, attn], None, self.mlp_query, normalize)[0]
        else:
            mix = attn_res([*blocks, partial], attn, self.mlp_query, normalize)[0]
        mlp_in = self.pre_mlp_layernorm(mix)
        mlp_in, moe_padding_mask, _, unflatten = self._maybe_unflatten_for_moe(
            mlp_in, padding_mask, None, packed_seq_params
        )
        mlp, _ = self.mlp(mlp_in, padding_mask=moe_padding_mask)
        if unflatten is not None:
            mlp = self._maybe_reflatten_from_moe(mlp, packed_seq_params, unflatten)
        mlp = self.mlp_sconv(mlp, layout.local_positions)

        output = pick(cfg, next_attn_res_state)(blocks, partial, attn, mlp, self.is_boundary)
        output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
        return output, None


@fused
def next_attn_res_state(blocks: List[Tensor], partial: Tensor, attn: Tensor, mlp: Tensor, boundary: bool) -> Tensor:
    """The packed state after a layer: at a block boundary the partial sum closes a block and ``attn + mlp`` starts
    the next one, otherwise ``attn + mlp`` is added to the partial sum (in FP32)."""
    if boundary:
        blocks = [*blocks, partial]
        partial = attn + mlp
    else:
        partial = (partial.float() + attn.float() + mlp.float()).to(attn.dtype)
    return torch.cat([*blocks, partial], dim=-1)


class Inkling2FinalNorm(MegatronModule):
    """The decoder's ``final_layernorm``: attn-res output mixture (``output_query``), then RMSNorm."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float = 1e-6):
        super().__init__(config=config)
        self.hidden_size = hidden_size
        self.output_query = nn.Parameter(
            torch.zeros(hidden_size, dtype=torch.float32, device=torch.cuda.current_device())
        )
        self.output_query.keep_in_fp32 = True
        if config.sequence_parallel:
            mark_sequence_parallel(self.output_query)
        self.norm = Inkling2RMSNorm(config, hidden_size, eps)

    def forward(self, hidden_states: Tensor) -> Tensor:
        blocks, partial = split_state(hidden_states, self.hidden_size)
        mix = pick(self.config, block_attn_res)(
            [*blocks, partial], None, self.output_query, self.config.attn_res_normalize_values
        )[0]
        return self.norm(mix)


class Inkling2EmbeddingNormMixin:
    """Applies ``embed_norm`` to the ``LanguageModelEmbedding`` output (mixed into the instance)."""

    def forward(self, *args, **kwargs):
        return self.embed_norm(super().forward(*args, **kwargs))


class Inkling2OutputLayerMixin:
    """Soft-caps the logits at ``final_logit_softcapping`` and masks the padded vocabulary
    (ids ``>= unpadded_vocab_size``) of the output layer (mixed into the instance)."""

    def forward(self, *args, **kwargs):
        logits, bias = super().forward(*args, **kwargs)
        vocab_start = self.tp_group.rank() * logits.shape[-1] if self.tp_group is not None else 0
        logits = pick(self.config, softcap_logits)(
            logits, self.config.final_logit_softcapping, vocab_start, self.config.unpadded_vocab_size
        )
        return logits, bias


@fused
def softcap_logits(logits: Tensor, cap: float, vocab_start: int, unpadded_vocab_size: int) -> Tensor:
    """``cap * tanh(logits / cap)``, with the padded vocabulary (ids ``>= unpadded_vocab_size``) masked."""
    logits = (cap * torch.tanh(logits.float() / cap)).to(logits.dtype)
    ids = torch.arange(vocab_start, vocab_start + logits.shape[-1], device=logits.device)
    # Finite: downstream entropy terms compute p * log p, which is NaN for p = 0, logp = -inf.
    return logits.masked_fill(ids >= unpadded_vocab_size, -1e30)


def extend_instance(obj, mixin) -> None:
    """Make ``mixin`` the first base of ``obj``'s class. The class keeps its name: Megatron-Bridge
    infers a weight's tensor-parallel layout from its module's class name."""
    base = type(obj)
    obj.__class__ = type(base.__name__, (mixin, base), {})


__all__ = [
    "Inkling2Attention",
    "Inkling2AttentionSubmodules",
    "Inkling2EmbeddingNormMixin",
    "Inkling2FinalNorm",
    "Inkling2OutputLayerMixin",
    "Inkling2RMSNorm",
    "Inkling2ShortConv",
    "Inkling2TransformerLayer",
    "block_attn_res",
    "extend_instance",
]
