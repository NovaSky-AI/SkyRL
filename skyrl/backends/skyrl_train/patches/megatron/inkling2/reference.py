"""Plain-PyTorch reference forward pass of the Inkling2 language model.

The release ships no reference implementation (only the SGLang fork's fused kernels), so this
module is the oracle the Megatron port is tested against. It is a transcription of the fork's
``sglang/tml`` model (``model.py``, ``layers/{attn,attn_res,moe,norm,sconv}.py`` and the torch
fallbacks / Triton kernels they call) for full-sequence prefill of a single sequence, keeping the
fork's dtypes at every rounding point:

- norms, attn-res, sconv, gate/sink and the MoE activation compute in FP32 and round to BF16
  where the kernels store BF16; attn-res outputs stay FP32 until the following norm;
- linears are BF16 GEMMs with BF16 output, except the router and unembedding (FP32 output);
- attention is computed densely in FP32 (the fork's FA4 kernel returns FP32 output and LSE).

Parameter names mirror the checkpoint (``model.llm.*``), so the state dict loads by name. Routed
experts are kept in their stored format (NVFP4 / MXFP8 / BF16) and dequantized one expert at a
time, so the full quantized release fits across a node's GPUs.

``act_quant=True`` additionally emulates the serving path's activation quantization of both
expert GEMM inputs (``hf_quant_config.json``: NVFP4 per-token "four over six" for NVFP4 experts,
MXFP8 for MXFP8 experts), with FP32 dequantized operands. ``act_quant=False`` is the BF16 model
that Megatron and a BF16 inference engine implement.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.inkling2.configuration import (
    Inkling2TextConfig,
)
from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (
    E2M1_VALUES,
    dequantize_expert_weight,
)

PREFIX = "model.llm."
ATTN_RES_EPS = 1e-6
L2_CLIP_EPS = 1e-6


def rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float, out_dtype=torch.bfloat16):
    """FP32 RMSNorm (``tml/kernels/norm.py``); ``weight`` is upcast, output rounded once."""
    u = x.float()
    y = u * torch.rsqrt(u.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(out_dtype)


def linear(x: torch.Tensor, weight: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    if out_dtype == torch.float32:
        return x.float() @ weight.float().t()
    return F.linear(x, weight).to(out_dtype)


# --------------------------------------------------------------------------------------------
# Block attention residuals (``tml/layers/attn_res.py``)
# --------------------------------------------------------------------------------------------


def block_attn_res(blocks, partial, queries, normalize_values: bool):
    """Softmax over depth of the (RMS-normalized) sources, one mixture per row of ``queries``.

    ``partial`` may be ``(p, addend)``: the last source is then ``p + addend`` in FP32.
    Returns a tuple of FP32 ``[T, H]`` tensors.
    """
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    addend = None
    if isinstance(partial, tuple):
        partial, addend = partial
    sources = [*blocks, partial]
    hidden = sources[0].shape[-1]
    values = torch.stack(sources).float()
    if addend is not None:
        values[-1] += addend.float()
    inv_rms = torch.rsqrt(values.square().mean(-1) + ATTN_RES_EPS)
    q_eff = queries.to(values.device).float() * hidden**-0.5
    logits = torch.einsum("ntd,qd,nt->qtn", values, q_eff, inv_rms)
    probs = torch.softmax(logits, dim=-1)
    if normalize_values:
        values = values * inv_rms.unsqueeze(-1)
    return tuple(torch.einsum("qtn,ntd->qtd", probs, values).unbind(0))


class AttnResState:
    def __init__(self, model: "Inkling2Reference", h0: torch.Tensor):
        self.model = model
        self.partial = h0
        self.blocks: List[torch.Tensor] = []

    def to(self, device: torch.device) -> None:
        """Move the carried state to the device of the next layer (layers may span GPUs)."""
        self.partial = self.partial.to(device)
        self.blocks = [b.to(device) for b in self.blocks]

    def _is_boundary(self, i: int) -> bool:
        return i % self.model.config.n_layers_per_attn_res_block == 0

    def attention_input(self, i: int, n_streams: int):
        if not self.blocks:
            return (self.partial,) * n_streams
        return block_attn_res(
            self.blocks, self.partial, self.model.attn_queries[i - 1], self.model.config.attn_res_normalize_values
        )

    def mlp_input(self, i: int, attn_out: torch.Tensor) -> torch.Tensor:
        if self._is_boundary(i):
            blocks, partial = [*self.blocks, self.partial], attn_out
        else:
            blocks, partial = self.blocks, (self.partial, attn_out)
        return block_attn_res(blocks, partial, self.model.mlp_queries[i], self.model.config.attn_res_normalize_values)[
            0
        ]

    def finalize_layer(self, i: int, attn_out: torch.Tensor, mlp_out: torch.Tensor) -> None:
        if self._is_boundary(i):
            self.blocks.append(self.partial)
            self.partial = attn_out + mlp_out  # BF16 add
        else:
            # ``tml/kernels/residual.py``: FP32 three-way sum, one rounding.
            self.partial = (self.partial.float() + attn_out.float() + mlp_out.float()).to(attn_out.dtype)

    def output(self) -> torch.Tensor:
        return block_attn_res(
            self.blocks, self.partial, self.model.output_query, self.model.config.attn_res_normalize_values
        )[0]


# --------------------------------------------------------------------------------------------
# Short convolutions (``tml/kernels/sconv.py``)
# --------------------------------------------------------------------------------------------


def _shift(x: torch.Tensor, lag: int) -> torch.Tensor:
    """``x[t - lag]`` with zeros before the sequence start (prefill without cache)."""
    if lag == 0:
        return x
    return torch.cat([x.new_zeros(lag, x.shape[-1]), x[:-lag]])


def static_sconv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal conv, ``weight [D, 1, W]`` (tap ``W - 1`` = current token), identity
    residual, scaled by ``(W + 1)^-1/2``."""
    w = weight.squeeze(1).to(x.dtype).float()  # [D, W]
    width = w.shape[1]
    xf = x.float()
    acc = torch.zeros_like(xf)
    for tap in range(width):
        acc = acc + _shift(xf, width - 1 - tap) * w[:, tap]
    return ((acc + xf) * (width + 1) ** -0.5).to(x.dtype)


def dynamic_sconv(x: torch.Tensor, static_w: torch.Tensor, weight_proj: torch.Tensor, source: torch.Tensor):
    """Global-layer K/V conv: per-token, per-head weights ``static_w[lag, c] + (source @
    weight_proj^T)[t, c // head_dim, lag]`` (``static_w [W, D]``, row 0 = current token), identity
    residual, scaled by ``(2W + 1)^-1/2``."""
    width, dim = static_w.shape
    n_heads = weight_proj.shape[0] // width
    dyn = linear(source.to(x.dtype), weight_proj.to(x.dtype)).view(-1, n_heads, width)  # BF16
    dyn = dyn.float().repeat_interleave(dim // n_heads, dim=1)  # [T, D, W]
    xf = x.float()
    acc = torch.zeros_like(xf)
    for lag in range(width):
        acc = acc + _shift(xf, lag) * (static_w[lag].float() + dyn[:, :, lag])
    return ((acc + xf) * (2 * width + 1) ** -0.5).to(x.dtype)


# --------------------------------------------------------------------------------------------
# Attention (``tml/layers/attn.py``)
# --------------------------------------------------------------------------------------------


def l2_clip(x: torch.Tensor, threshold: float, scale: Optional[torch.Tensor] = None, output_scale=None):
    """``x [T, heads, d]``: scale per head, then clip each vector's L2 norm to ``threshold``."""
    value = x.float()
    if scale is not None:
        value = value * scale.float()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    out = (value * (threshold / (norm + L2_CLIP_EPS)).clamp(max=1.0)).to(x.dtype)
    if output_scale is not None:
        out = (out.float() * output_scale.view(-1, 1, 1)).to(x.dtype)
    return out


def log_scaling_tau(positions: torch.Tensor, n_floor: float, alpha: float) -> torch.Tensor:
    return 1.0 + alpha * torch.log(torch.clamp((positions + 1).float() / n_floor, min=1.0))


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    window: Optional[int],
    query_chunk: int = 1024,
):
    """FP32 causal GQA attention with the sheared relative bias ``rel[i, h, i - j]``
    (0 outside ``[0, rel_extent)``) and an optional sliding window (keys ``i - j < window``).

    ``q [T, Hq, d]``, ``k / v [T, Hkv, d]``, ``rel [T, Hq, E]``. Returns ``(out [T, Hq, d] FP32,
    lse [T, Hq] FP32)``.
    """
    T, Hq, d = q.shape
    group = Hq // k.shape[1]
    extent = rel.shape[-1]
    kf = k.float().repeat_interleave(group, dim=1).transpose(0, 1)  # [Hq, T, d]
    vf = v.float().repeat_interleave(group, dim=1).transpose(0, 1)
    scale = d**-0.5
    outs, lses = [], []
    j = torch.arange(T, device=q.device)
    for start in range(0, T, query_chunk):
        i = torch.arange(start, min(start + query_chunk, T), device=q.device)
        qf = q[i].float().transpose(0, 1)  # [Hq, C, d]
        scores = (qf @ kf.transpose(1, 2)) * scale  # [Hq, C, T]
        dist = i[:, None] - j[None, :]  # [C, T]
        in_band = (dist >= 0) & (dist < extent)
        idx = dist.clamp(0, extent - 1).unsqueeze(1).expand(-1, Hq, -1)  # [C, Hq, T]
        bias = rel[i].float().gather(-1, idx).transpose(0, 1)  # [Hq, C, T]
        scores = scores + bias.masked_fill(~in_band, 0.0)
        visible = dist >= 0
        if window is not None:
            visible &= dist < window
        scores = scores.masked_fill(~visible, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)  # [Hq, C]
        probs = torch.exp(scores - lse.unsqueeze(-1))
        outs.append((probs @ vf).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs), torch.cat(lses)


class Attention(nn.Module):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.is_local = config.is_local_layer(layer_idx)
        heads = config.swa_num_attention_heads if self.is_local else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if self.is_local else config.num_key_value_heads
        head_dim = config.swa_head_dim if self.is_local else config.head_dim
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        H, R = config.hidden_size, config.d_rel
        extent = config.sliding_window_size if self.is_local else config.rel_extent

        def p(*shape, dtype=torch.bfloat16):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.wq_du = nn.Linear(H, heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.wk_dv = nn.Linear(H, kv_heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.wv_dv = nn.Linear(H, kv_heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.wr_du = nn.Linear(H, heads * R, bias=False, dtype=torch.bfloat16)
        self.gate_attn = nn.Linear(H, heads * head_dim, bias=False, dtype=torch.bfloat16)
        self.attn_sink_proj = nn.Linear(H, heads, bias=False, dtype=torch.bfloat16)
        self.wo_ud = nn.Linear(heads * head_dim, H, bias=False, dtype=torch.bfloat16)
        self.q_l2_clip = nn.Module()
        self.q_l2_clip.scale = p(heads, head_dim)
        self.rel_l2_clip = nn.Module()
        self.rel_l2_clip.scale = p(heads, R)
        self.rel_logits_proj = nn.Module()
        self.rel_logits_proj.proj = p(R, extent)
        W, kv_dim = config.sconv_kernel_size, kv_heads * head_dim
        for name in ("k_sconv", "v_sconv"):
            conv = nn.Module()
            if self.is_local:
                conv.weight = p(kv_dim, 1, W)
            else:
                conv.static_w = p(W, kv_dim)
                conv.weight_proj = p(kv_heads * W, H)
            setattr(self, name, conv)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor, positions: torch.Tensor):
        cfg, T = self.config, xq.shape[0]
        q = linear(xq, self.wq_du.weight)
        r = linear(xq, self.wr_du.weight)
        gate = linear(xq, self.gate_attn.weight)
        sink = linear(xq, self.attn_sink_proj.weight)
        k = linear(xk, self.wk_dv.weight)
        v = linear(xv, self.wv_dv.weight)
        if self.is_local:
            k = static_sconv(k, self.k_sconv.weight)
            v = static_sconv(v, self.v_sconv.weight)
            tau = None
        else:
            k = dynamic_sconv(k, self.k_sconv.static_w, self.k_sconv.weight_proj, xk)
            v = dynamic_sconv(v, self.v_sconv.static_w, self.v_sconv.weight_proj, xv)
            tau = log_scaling_tau(positions, cfg.log_scaling_n_floor, cfg.log_scaling_alpha)
        q = l2_clip(q.view(T, self.heads, self.head_dim), cfg.qk_l2_clip_threshold, self.q_l2_clip.scale, tau)
        k = l2_clip(k.view(T, self.kv_heads, self.head_dim), cfg.qk_l2_clip_threshold)
        v = v.view(T, self.kv_heads, self.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = rms_norm(v, None, cfg.rms_norm_eps, out_dtype=v.dtype)
        r = l2_clip(r.view(T, self.heads, cfg.d_rel), cfg.rel_l2_clip_threshold, self.rel_l2_clip.scale, tau)
        rel = torch.einsum("thd,de->the", r, self.rel_logits_proj.proj)  # BF16, as the fork
        window = cfg.sliding_window_size if self.is_local else None
        out, lse = dense_attention(q, k, v, rel, window)
        # ``tml/kernels/attention.py`` gate_sink: FP32, one rounding.
        out = out * torch.sigmoid(gate.float().view(T, self.heads, self.head_dim))
        out = out * torch.sigmoid(lse - sink.float()).unsqueeze(-1)
        out = out.reshape(T, -1).to(xq.dtype)
        return linear(out, self.wo_ud.weight)


# --------------------------------------------------------------------------------------------
# MoE (``tml/layers/moe.py``, ``tml/kernels/moe.py``)
# --------------------------------------------------------------------------------------------


def fake_quant_nvfp4(x: torch.Tensor) -> torch.Tensor:
    """NVFP4 activation quantization of ``tml/kernels/nvfp4.py`` (per-row amax, 1x16 blocks with
    the lower-MAE of the /6 and /4 block scales), returned dequantized in FP32."""
    x = x.float()
    fmax = torch.finfo(torch.float32).max
    amax = x.abs().amax(-1)
    encoder = (1536.0 / amax).clamp(max=fmax)
    decoder = 1.0 / encoder
    outer = amax * (1.0 / 1536.0)
    blocks = x.unflatten(-1, (-1, 16))
    block_max = blocks.abs().amax(-1)
    grid = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    boundaries = (grid[:-1] + grid[1:]) / 2
    best, best_err = None, None
    for inverse_max in (1.0 / 6.0, 0.25):
        scale = (block_max * (encoder[..., None] * inverse_max)).clamp(max=448.0)
        scale = scale.to(torch.float8_e4m3fn).float()
        inverse = (1.0 / (scale * decoder[..., None])).clamp(max=fmax)
        values = blocks * inverse[..., None]
        magnitude = values.abs()
        code = torch.bucketize(magnitude, boundaries)
        # cvt.rn: ties to the even code.
        ties = magnitude == boundaries[code.clamp(max=6)]
        code = code + (ties & (code % 2 == 1)).long()
        rounded = grid[code] * torch.where(values.signbit(), -1.0, 1.0)
        recon = rounded * scale[..., None] * outer[..., None, None]
        err = (recon - blocks).abs().sum(-1)
        if best is None:
            best, best_err = recon, err
        else:
            choose = err < best_err  # strictly better /4 wins, as the kernel
            best = torch.where(choose[..., None], recon, best)
    return best.flatten(-2)


def fake_quant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """MXFP8 activation quantization of ``tml/kernels/moe.py`` (``_mxfp8_block``), dequantized."""
    groups = x.float().unflatten(-1, (-1, 32))
    raw = groups.abs().amax(-1) * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, torch.zeros_like(exponent), exponent.clamp(max=254))
    scale = torch.exp2(exponent.float() - 127.0).clamp(min=2.0**-127)
    inverse = torch.where(exponent == 0, torch.full_like(scale, 2.0**127), 1.0 / scale)
    values = (groups * inverse[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (values * scale[..., None]).flatten(-2)


class Experts(nn.Module):
    """Routed experts kept in the stored format; ``weight(name, e)`` dequantizes one expert."""

    def __init__(self):
        super().__init__()
        self.stored: Dict[str, torch.Tensor] = {}

    def load(self, name: str, tensor: torch.Tensor) -> None:
        self.stored[name] = tensor

    def fmt(self, which: str) -> str:
        dtype = self.stored[f"{which}_weight"].dtype
        return {torch.uint8: "nvfp4", torch.float8_e4m3fn: "mxfp8"}.get(dtype, "bf16")

    def weight(self, which: str, e: int, dtype: torch.dtype) -> torch.Tensor:
        name = f"{which}_weight"
        sl = slice(e, e + 1)
        tensors = {name: self.stored[name][sl]}
        for suffix in ("_scale", "_amax"):
            if name + suffix in self.stored:
                tensors[name + suffix] = self.stored[name + suffix][sl]
        if name + "_outer_scale_denominator" in self.stored:
            tensors[name + "_outer_scale_denominator"] = self.stored[name + "_outer_scale_denominator"]
        return dequantize_expert_weight(name, tensors, dtype)[0]


class MoE(nn.Module):
    def __init__(self, config: Inkling2TextConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, dtype=torch.bfloat16), requires_grad=False
        )
        self.gate.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32), requires_grad=False)
        self.experts = Experts()

    def route(self, x: torch.Tensor):
        cfg = self.config
        logits = linear(x, self.gate.weight, out_dtype=torch.float32)
        selection = logits.sigmoid()
        if cfg.use_gate_bias:
            selection = selection + self.gate.bias.float()
        ids = selection.topk(cfg.num_experts_per_tok, dim=-1).indices
        chosen = logits.gather(-1, ids).sigmoid()
        weights = chosen / chosen.sum(-1, keepdim=True) * cfg.route_scale
        return weights, ids

    def _expert_matmul(self, x, which, e, act_quant):
        fmt = self.experts.fmt(which)
        if act_quant and fmt != "bf16":
            xq = fake_quant_nvfp4(x) if fmt == "nvfp4" else fake_quant_mxfp8(x)
            return (xq @ self.experts.weight(which, e, torch.float32).t()).to(x.dtype)
        return linear(x, self.experts.weight(which, e, x.dtype))

    def forward(self, x: torch.Tensor, act_quant: bool = False) -> torch.Tensor:
        cfg = self.config
        weights, ids = self.route(x)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in ids.unique().tolist():
            token, slot = (ids == e).nonzero(as_tuple=True)
            h = self._expert_matmul(x[token], "w13", e, act_quant)
            g, u = h[:, 0::2].float(), h[:, 1::2].float()
            act = cfg.mlp_gate_softcap * torch.tanh(g / cfg.mlp_gate_softcap) * torch.sigmoid(g)
            act = act * (cfg.mlp_up_softcap * torch.tanh(u / cfg.mlp_up_softcap))
            act = (act * weights[token, slot, None]).to(x.dtype)
            out.index_add_(0, token, self._expert_matmul(act, "w2", e, act_quant).float())
        return out.to(x.dtype)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_local = config.is_local_layer(layer_idx)
        H = config.hidden_size
        norms = ("attn_norm",) if self.is_local else ("q_input_norm", "k_input_norm", "v_input_norm")
        for name in (*norms, "mlp_norm"):
            norm = nn.Module()
            norm.weight = nn.Parameter(torch.empty(H, dtype=torch.bfloat16), requires_grad=False)
            setattr(self, name, norm)
        self.attn = Attention(config, layer_idx)
        self.mlp = MoE(config)
        for name in ("attn_sconv", "mlp_sconv"):
            conv = nn.Module()
            conv.weight = nn.Parameter(
                torch.empty(H, 1, config.sconv_kernel_size, dtype=torch.bfloat16), requires_grad=False
            )
            setattr(self, name, conv)
        self.eps = config.rms_norm_eps

    @property
    def device(self) -> torch.device:
        return self.mlp_norm.weight.device

    def forward(self, state: AttnResState, positions: torch.Tensor, act_quant: bool) -> None:
        i, eps, dev = self.layer_idx, self.eps, self.device
        state.to(dev)
        streams = state.attention_input(i, 1 if self.is_local else 3)
        if self.is_local:
            xq = xk = xv = rms_norm(streams[0], self.attn_norm.weight, eps)
        else:
            xq = rms_norm(streams[0], self.q_input_norm.weight, eps)
            xk = rms_norm(streams[1], self.k_input_norm.weight, eps)
            xv = rms_norm(streams[2], self.v_input_norm.weight, eps)
        attn = static_sconv(self.attn(xq, xk, xv, positions.to(dev)), self.attn_sconv.weight)
        mlp_in = rms_norm(state.mlp_input(i, attn), self.mlp_norm.weight, eps)
        mlp = static_sconv(self.mlp(mlp_in, act_quant), self.mlp_sconv.weight)
        state.finalize_layer(i, attn, mlp)


class Inkling2Reference(nn.Module):
    """The language model; ``forward(input_ids [T]) -> logits [T, V]`` (FP32, soft-capped, padded
    vocab masked to -inf)."""

    def __init__(self, config: Inkling2TextConfig):
        super().__init__()
        self.config = config
        H, V, L = config.hidden_size, config.vocab_size, config.num_hidden_layers

        def p(*shape, dtype=torch.bfloat16):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        self.embed = nn.Module()
        self.embed.weight = p(V, H)
        self.embed_norm = nn.Module()
        self.embed_norm.weight = p(H)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(L)])
        self.attn_res = nn.Module()
        self.attn_res.attn_queries = nn.ParameterList(
            [
                p(H, dtype=torch.float32) if config.is_local_layer(i) else p(3, H, dtype=torch.float32)
                for i in range(1, L)
            ]
        )
        self.attn_res.mlp_queries = nn.ParameterList([p(H, dtype=torch.float32) for _ in range(L)])
        self.attn_res.output_query = p(H, dtype=torch.float32)
        self.norm = nn.Module()
        self.norm.weight = p(H)
        self.unembed = nn.Module()
        self.unembed.weight = p(V, H)

    @property
    def attn_queries(self):
        return self.attn_res.attn_queries

    @property
    def mlp_queries(self):
        return self.attn_res.mlp_queries

    @property
    def output_query(self):
        return self.attn_res.output_query

    def hidden_states(self, input_ids: torch.Tensor, act_quant: bool = False) -> torch.Tensor:
        """Final-norm output ``[T, H]`` (BF16) for one sequence."""
        cfg = self.config
        dev0 = self.embed.weight.device
        input_ids = input_ids.to(dev0)
        positions = torch.arange(input_ids.shape[0], device=dev0)
        h0 = rms_norm(self.embed.weight[input_ids], self.embed_norm.weight, cfg.rms_norm_eps)
        state = AttnResState(self, h0)
        for layer in self.layers:
            layer(state, positions, act_quant)
        state.to(self.norm.weight.device)
        return rms_norm(state.output(), self.norm.weight, cfg.rms_norm_eps)

    def logits(self, hidden: torch.Tensor, lm_head_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``lm_head_dtype``: the release's quant config asks for FP32 unembedding output; without a
        quant config the fork's LM head returns BF16 logits."""
        cfg = self.config
        logits = linear(hidden.to(self.unembed.weight.device), self.unembed.weight, out_dtype=lm_head_dtype).float()
        cap = cfg.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)
        logits[..., cfg.unpadded_vocab_size :] = float("-inf")
        return logits

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, act_quant: bool = False) -> torch.Tensor:
        return self.logits(self.hidden_states(input_ids, act_quant))

    @torch.no_grad()
    def token_logprobs(
        self,
        input_ids: torch.Tensor,
        act_quant: bool = False,
        top_k: int = 0,
        chunk: int = 2048,
        lm_head_dtype: torch.dtype = torch.float32,
    ) -> Dict[str, torch.Tensor]:
        """Logprob of each next token ``[T - 1]`` and optionally the top-k logprobs per position."""
        hidden = self.hidden_states(input_ids, act_quant)
        target = input_ids[1:].to(self.unembed.weight.device)
        lps, topv, topi = [], [], []
        for s in range(0, hidden.shape[0] - 1, chunk):
            lp = torch.log_softmax(self.logits(hidden[s : min(s + chunk, hidden.shape[0] - 1)], lm_head_dtype), dim=-1)
            lps.append(lp.gather(-1, target[s : s + lp.shape[0], None]).squeeze(-1))
            if top_k:
                v, i = lp.topk(top_k, dim=-1)
                topv.append(v)
                topi.append(i)
        out = {"logprobs": torch.cat(lps).cpu()}
        if top_k:
            out["top_logprobs"], out["top_ids"] = torch.cat(topv).cpu(), torch.cat(topi).cpu()
        return out


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def _layer_of(name: str) -> Optional[int]:
    if name.startswith(PREFIX + "layers."):
        return int(name[len(PREFIX + "layers.") :].split(".", 1)[0])
    return None


def load_reference(path: str, devices: Optional[Sequence[torch.device]] = None) -> Inkling2Reference:
    """Build the reference from a checkpoint directory (release or dequantized), placing
    contiguous layer ranges on ``devices`` (default: all visible GPUs)."""
    path = Path(path)
    config = Inkling2TextConfig(**json.loads((path / "config.json").read_text())["text_config"])
    if devices is None:
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    L = config.num_hidden_layers
    per_device = math.ceil(L / len(devices))
    layer_device = [devices[min(i // per_device, len(devices) - 1)] for i in range(L)]

    with torch.device("meta"):
        model = Inkling2Reference(config)
    params = dict(model.named_parameters())
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    by_file = defaultdict(list)
    for name, file in weight_map.items():
        if name.startswith(PREFIX):
            by_file[file].append(name)

    loaded = set()
    for file, names in sorted(by_file.items()):
        with safe_open(path / file, framework="pt") as f:
            for name in names:
                key = name[len(PREFIX) :]
                layer = _layer_of(name)
                if key.startswith("unembed.") or key.startswith("norm."):
                    device = devices[-1]
                elif layer is not None:
                    device = layer_device[layer]
                else:
                    device = devices[0]
                tensor = f.get_tensor(name).to(device)
                if ".mlp.experts." in key:
                    module = model.layers[layer].mlp.experts
                    module.load(key.rsplit(".", 1)[1], tensor)
                    loaded.add(key)
                    continue
                if key not in params:
                    raise KeyError(f"unexpected checkpoint tensor {name}")
                parent, attr = key.rsplit(".", 1)
                module = model.get_submodule(parent)
                if tuple(tensor.shape) != tuple(params[key].shape):
                    raise ValueError(f"{name}: shape {tuple(tensor.shape)} != {tuple(params[key].shape)}")
                setattr(module, attr, nn.Parameter(tensor, requires_grad=False))
                loaded.add(key)
    missing = sorted(k for k in params if k not in loaded)
    if missing:
        raise KeyError(f"missing checkpoint tensors: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    for layer in model.layers:
        if "w13_weight" not in layer.mlp.experts.stored or "w2_weight" not in layer.mlp.experts.stored:
            raise KeyError(f"layer {layer.layer_idx}: missing expert weights")
    return model.eval()
