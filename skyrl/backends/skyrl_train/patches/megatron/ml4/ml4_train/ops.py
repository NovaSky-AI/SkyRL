"""Numerics shared by the ML4 training model, written to vLLM semantics.

Every function cites the vLLM code (Inferact/mav @ e9d8d15) it mirrors.
"""

import math

import torch
import torch.nn.functional as F

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM's ``vllm_c`` rms_norm (first in the IR priority on CUDA): fp32
    reduction, ``x * rsqrt(var + eps) * w`` in fp32, one rounding
    (csrc/libtorch_stable/layernorm_kernels.cu:88-93). The ``native`` IR path
    instead rounds before multiplying by the weight.
    """
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * weight.float()).to(x.dtype)


def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's ``vllm_c`` fused_add_rms_norm: the sum is rounded to x.dtype
    (the new residual) and that rounded sum is normalized
    (layernorm_kernels.cu:131-163; scalar path 199-212).
    """
    r = (x.float() + residual.float()).to(x.dtype)
    return rms_norm(r, weight, eps), r


def layer_norm_fp32(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    """LayerNorm reduced in fp32, result cast back to x.dtype.

    vLLM's ``LayerNorm.forward_native`` computes in the wider of x/param
    dtypes; the indexer ``k_norm`` has fp32 params (indexer.py:117) and CUDA
    ``F.layer_norm`` on bf16 also reduces in fp32, so fp32 is used for both.
    """
    y = F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps)
    return y.to(x.dtype)


def _yarn_correction_dim(rot: float, dim: int, base: float, max_pos: int) -> float:
    return (dim * math.log(max_pos / (rot * 2 * math.pi))) / (2 * math.log(base))


def rope_inv_freq(
    rope_parameters: dict, rotary_dim: int, lerope_alpha: torch.Tensor | None
) -> torch.Tensor:
    """fp32 inverse frequencies: YaRN + LeRoPE.

    vllm/models/mistral/common/rope.py:105-129 (``_compute_inv_freq``) and
    rotary_embedding/common.py yarn helpers. LeRoPE scales both the
    extrapolated and interpolated frequencies by exp(alpha); the YaRN ramp
    uses the unscaled ones. mscale is asserted to be 1 (rope.py:103).
    """
    base = float(rope_parameters["rope_theta"])
    pos_freqs = base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    extra = 1.0 / pos_freqs
    if rope_parameters.get("rope_type", "default") != "yarn":
        if lerope_alpha is not None:
            extra = extra * torch.exp(lerope_alpha.float().to(pos_freqs.device))
        return extra
    assert rope_parameters.get("attention_factor") == 1.0
    assert rope_parameters.get("mscale", 1) == rope_parameters.get("mscale_all_dim", 0)
    factor = float(rope_parameters["factor"])
    inter = 1.0 / (factor * pos_freqs)
    if lerope_alpha is not None:
        scale = torch.exp(lerope_alpha.float().to(pos_freqs.device))
        extra, inter = extra * scale, inter * scale
    max_pos = rope_parameters["original_max_position_embeddings"]
    low = math.floor(_yarn_correction_dim(rope_parameters.get("beta_fast", 32), rotary_dim, base, max_pos))
    high = math.ceil(_yarn_correction_dim(rope_parameters.get("beta_slow", 1), rotary_dim, base, max_pos))
    low, high = max(low, 0), min(high, rotary_dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(rotary_dim // 2, dtype=torch.float) - low) / (high - low)).clamp(0, 1)
    mask = 1 - ramp
    return inter * (1 - mask) + extra * mask


def fp32_einsum(eq: str, *operands: torch.Tensor) -> torch.Tensor:
    """``torch.einsum`` on FP32 operands with autocast off: under
    ``torch.autocast(bfloat16)`` (SkyRL's policy forward) einsum's bmm would
    otherwise run in BF16."""
    with torch.autocast(operands[0].device.type, enabled=False):
        return torch.einsum(eq, *operands)


def fp32_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """FP32 ``F.linear`` that autocast cannot downcast."""
    with torch.autocast(x.device.type, enabled=False):
        return torch.nn.functional.linear(x.float(), w.float())


def rope_cos_sin(
    positions: torch.Tensor, inv_freq: torch.Tensor, cache_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin rows of vLLM's cos_sin_cache for ``positions``.

    deepseek_scaling_rope.py:_compute_cos_sin_cache builds fp32
    ``t * inv_freq`` then base.py:61 casts the cache to the model dtype; the
    same fp32 product and the same cast are reproduced here per position.
    """
    with torch.autocast(positions.device.type, enabled=False):
        freqs = positions.float()[..., None] * inv_freq.to(positions.device).float()[None]
    return freqs.cos().to(cache_dtype), freqs.sin().to(cache_dtype)


def apply_rope_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """GPT-J (interleaved) rotation, fp32 math, result in x.dtype.

    x: [..., T, (H,) D] with D == 2 * cos.shape[-1]; cos/sin: [..., T, D/2].
    Matches forward_static with is_neox_style=False.
    """
    if x.dim() == cos.dim() + 1:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    xf = x.float()
    x0, x1 = xf[..., 0::2], xf[..., 1::2]
    c, s = cos.float(), sin.float()
    out = torch.stack((x0 * c - x1 * s, x1 * c + x0 * s), dim=-1).flatten(-2)
    return out.to(x.dtype)


def ssmax_scale(
    q: torch.Tensor, log_pos: torch.Tensor, s: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """q * (s * log(pos + 1) + b) per head, in q.dtype (ssmax.py:255-271).

    q: [B, T, H, D]; log_pos: [B, T] fp32.
    """
    dt = q.dtype
    scale = log_pos.to(dt)[..., None] * s.to(dt) + b.to(dt)
    return q * scale[..., None]


def fp8_round(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values to float8_e4m3fn (saturating) and back to fp32."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def ue8m0_quant(x: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last dim) FP8 quantization with a power-of-two scale.

    kind="q_rope": sparse_attn_indexer.py fused_indexer_q_rope_quant
                   scale = 2^ceil(log2(max(amax, 1e-10) / 448))
    kind="q_nope": per_token_group_quant_fp8(use_ue8m0) / ops/indexer.py:121
                   scale = 2^ceil(log2(max(amax / 448, 1e-10)))
    kind="k":      cache_kernels.cu indexer_k_quant_and_cache
                   scale = 2^ceil(log2(max(amax, 1e-4) / 448))
    Returns (fp8 values as fp32, fp32 scale with a trailing singleton dim).
    """
    xf = x.float()
    amax = xf.abs().amax(-1, keepdim=True)
    if kind == "q_rope":
        # Triton multiplies by the fp32 reciprocal, not a true division.
        raw = amax.clamp_min(1e-10) * torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
    elif kind == "q_nope":
        raw = (amax / FP8_MAX).clamp_min(1e-10)
    elif kind == "k":
        raw = amax.clamp_min(1e-4) / FP8_MAX
    else:
        raise ValueError(kind)
    scale = torch.exp2(torch.ceil(torch.log2(raw)))
    return fp8_round(xf / scale), scale


FP8_DS_MLA_TILE = 128


@torch.no_grad()
def fp8_ds_mla_latent(kv_c: torch.Tensor) -> torch.Tensor:
    """kv_c [..., 512] as vLLM's ``fp8_ds_mla`` MLA cache stores and attention reads it:
    E4M3 per 128-column tile with a power-of-two FP32 scale
    ``2^ceil(log2(max(amax / 448, 1e-4)))``, dequantized ``fp8 * scale`` (exact in BF16).
    The RoPE part (k_pe) is stored in BF16 unchanged, so only kv_c is quantized.

    CUDA BF16/FP16: vLLM's own cache-write op (``ops.concat_and_cache_mla`` with
    ``kv_cache_dtype="fp8_ds_mla"``, the op ``do_kv_cache_update`` runs) into a 656-byte
    per-token buffer, decoded (512 E4M3 bytes, 4 FP32 tile scales). Otherwise (CPU tests,
    FP32 inputs of the G2 validation runs): the kernel's rule in FP32 torch ops
    (cache_kernels.cu ``concat_and_cache_ds_mla_kernel``). Returns FP32 [..., 512].
    """
    *lead, D = kv_c.shape
    if D != 512:
        raise ValueError(f"fp8_ds_mla needs kv_lora_rank 512, got {D}")
    n = kv_c.numel() // D
    if not kv_c.is_cuda or kv_c.dtype not in (torch.bfloat16, torch.float16) or n == 0:
        xf = kv_c.float().reshape(*lead, D // FP8_DS_MLA_TILE, FP8_DS_MLA_TILE)
        amax = xf.abs().amax(-1, keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2((amax / FP8_MAX).clamp_min(1e-4))))
        return (fp8_round(xf / scale) * scale).reshape(*lead, D)
    from vllm import _custom_ops as vllm_ops

    x = kv_c.reshape(n, D).contiguous()
    cache = torch.empty(n, 1, 656, dtype=torch.uint8, device=x.device)
    k_pe = torch.zeros(n, 64, dtype=x.dtype, device=x.device)
    slots = torch.arange(n, dtype=torch.int64, device=x.device)
    one = torch.ones((), dtype=torch.float32, device=x.device)
    vllm_ops.concat_and_cache_mla(x, k_pe, cache, slots, "fp8_ds_mla", one)
    c = cache.view(n, 656)
    q = c[:, :D].view(torch.float8_e4m3fn).float().view(n, D // FP8_DS_MLA_TILE, FP8_DS_MLA_TILE)
    scale = c[:, D : D + 16].contiguous().view(torch.float32)
    return (q * scale[..., None]).reshape(*lead, D)
