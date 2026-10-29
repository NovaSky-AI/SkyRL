"""NVFP4 checkpoint modules and rollout-matching QAT (NVFP4_PLAN.md sections 2-3).

Checkpoint: ModelOpt MIXED_PRECISION (``params.json`` ``quantization_config``).

* ``layers.N.experts`` (NVFP4, group 16): routed experts are ``NVFP4Linear``
  modules holding the packed checkpoint tensors (``weight`` U8 [out, in/2], low
  nibble = even column; ``weight_scale`` E4M3 bits as U8 [out, in/16];
  ``weight_scale_2`` / ``input_scale`` FP32 scalars). The BF16 base
  ``E2M1(code) * E4M3(sf)`` is decoded on the fly (exact: at most 6 significant
  bits); ``weight_scale_2`` multiplies the FP32 GEMM output and is never folded
  into a BF16 weight. ``input_scale`` (the static per-tensor activation scale of
  vLLM's default NVFP4 MoE activation) is unused: with
  ``moe.activation=nvfp4_per_token`` vLLM overwrites it with 1.0
  (``ModelOptNvFp4FusedMoE.process_weights_after_loading``) and computes per-token
  scales, so QAT requires ``config.rollout_moe_activation == "nvfp4_per_token"``.
* FP8_PB_WO linears (``wq``, ``wo``, indexer ``wq_b``, dense and shared-expert
  ``w1``/``w2``/``w3``): BF16 weights dequantized by the loader. vLLM runs them as
  W8A8 (DeepGEMM, per-(token, 128-col) FP8 activations with power-of-two scales),
  so ``FP8ActQuantLinear`` fake-quantizes its input.

QAT switches (config attributes, both default on):

* ``nvfp4_act_qat``: routed-expert path of vLLM's ``TrtLlmNvFp4LoRAExperts``
  with ``moe.activation=nvfp4_per_token``:
    FC1 input per-token NVFP4 over the full row (FP32 scale amax * f32(1/2688),
    E4M3 block scales per 16; FlashInfer's own kernel on CUDA,
    ``nvfp4_per_token_quant``); FC1 out = acc * s * ws2(gate), FP32;
    + gate/up LoRA delta (BF16, from the unquantized input, unweighted) in FP32;
    SwiGLU in FP32, rounded to BF16;
    FC2 input per-token NVFP4 per rollout-TP shard of the intermediate dim
    (``rollout_moe_tp`` shards, must equal ``rollout_tp``), shard scale on that
    shard's FP32 partial, partials summed in FP32, FC2 output rounded to BF16;
    down LoRA delta from the BF16 FC1 activation, times the routing weight,
    rounded to BF16;
    finalize: FP32 sum of w_k * base_k + delta_k, one BF16 rounding.
  Off: the expert runs as a plain module (BF16 F.linear with the decoded weight,
  then x ws2), LoRA added by PEFT inside the expert.
* ``fp8_act_qat``: FP8_PB_WO input fake quant (vLLM's DeepGEMM input quant op on
  CUDA, ``fp8_block_act_quant``); LoRA (PEFT) is added on top of the base output
  and sees the unquantized input.

All activation quantizers use a straight-through estimator: the backward pass is
the identity through the quantizer (and through the per-token / per-shard
scales). On CUDA the quantizers are the serving stack's own kernels (FlashInfer
NVFP4, vLLM FP8 group quant); only their outputs are decoded here. The CPU
versions are test-only references.
"""

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from .loading import map_name
from .ops import FP8_MAX, fp8_round

E2M1_MAX = 6.0
NVFP4_BLOCK = 16
NVFP4_TOKEN_DIV = FP8_MAX * E2M1_MAX  # 2688: per-token scale = amax / 2688
FP8_ACT_BLOCK = 128

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


# --------------------------------------------------------------------------
# E2M1 / E4M3 primitives
# --------------------------------------------------------------------------


def e2m1_lut(device=None) -> torch.Tensor:
    """FP32 value of each 4-bit code (bit 3 = sign, bits 0-2 = grid index)."""
    g = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=device)
    return torch.cat([g, -g])


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> 4-bit codes (uint8), round-to-nearest-even, saturating at 6
    (``cvt.rn.satfinite.e2m1x2.f32``)."""
    a = x.float().abs()
    mid = torch.tensor(_E2M1_MID, dtype=torch.float32, device=x.device)
    lo = torch.searchsorted(mid, a, right=False)  # midpoints strictly below a
    hi = torch.searchsorted(mid, a, right=True)  # midpoints <= a
    # On a tie (lo != hi) pick the even grid index (even mantissa).
    idx = torch.where((lo != hi) & (lo % 2 == 1), hi, lo)
    sign = (x < 0) & (idx > 0)
    return (idx + 8 * sign.long()).to(torch.uint8)


def e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """FP32 -> nearest E2M1 value (RNE, saturating) as FP32."""
    return e2m1_lut(x.device)[e2m1_encode(x).long()]


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., 2k] -> bytes [..., k]; even column in the low nibble."""
    c = codes.to(torch.uint8)
    return c[..., 0::2] | (c[..., 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """bytes [..., k] -> uint8 codes [..., 2k] (low nibble first)."""
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack([lo, hi], -1).flatten(-2)


def e4m3_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    """E4M3 scale tensor stored as uint8 bits (or already float) -> FP32."""
    if bits.dtype == torch.uint8:
        return bits.view(torch.float8_e4m3fn).float()
    return bits.float()


_BYTE_LUTS: dict = {}


def _byte_lut(device, dtype) -> torch.Tensor:
    """[256, 2]: (low-nibble value, high-nibble value) of each packed byte."""
    key = (str(device), dtype)
    if key not in _BYTE_LUTS:
        b = torch.arange(256, dtype=torch.uint8)
        lut = e2m1_lut()[unpack_nibbles(b[:, None]).long()]
        _BYTE_LUTS[key] = lut.to(device=device, dtype=dtype)
    return _BYTE_LUTS[key]


_DECODE_KERNEL = None


def _triton_decode_kernel():
    global _DECODE_KERNEL
    if _DECODE_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _e2m1(code):
            idx = code & 7
            e, mant = idx >> 1, (idx & 1).to(tl.float32)
            mag = tl.where(e == 0, 0.5 * mant, (1.0 + 0.5 * mant) * tl.exp2((e - 1).to(tl.float32)))
            return mag * tl.where(code >= 8, -1.0, 1.0)  # a multiply keeps the sign of zero

        @triton.jit
        def _decode(p_ptr, s_ptr, o_ptr, half, p_row, s_row, BLOCK: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            m = j < half
            byte = tl.load(p_ptr + row * p_row + j, mask=m, other=0).to(tl.int32)
            sb = tl.load(s_ptr + row * s_row + j // 8, mask=m, other=0)
            sf = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            lo = (_e2m1(byte & 15) * sf).to(o_ptr.dtype.element_ty)
            hi = (_e2m1(byte >> 4) * sf).to(o_ptr.dtype.element_ty)
            out = tl.interleave(lo, hi)
            c = tl.program_id(1) * 2 * BLOCK + tl.arange(0, 2 * BLOCK)
            tl.store(o_ptr + row * 2 * half + c, out, mask=c < 2 * half)

        _DECODE_KERNEL = (_decode, triton.cdiv)
    return _DECODE_KERNEL


def _decode_nvfp4_fused(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``decode_nvfp4`` in one Triton pass: each output element reads its byte and its
    E4M3 scale and writes E2M1 * scale. The FP32 product is exact (<= 6 significant
    bits), so the ``dtype`` result is bitwise the LUT-gather path's."""
    *lead, half = packed.shape
    k = 2 * half
    p = packed.reshape(-1, half)
    s = scale.reshape(-1, scale.shape[-1])
    if p.stride(-1) != 1:
        p = p.contiguous()
    if s.stride(-1) != 1:
        s = s.contiguous()
    out = torch.empty((p.shape[0], k), dtype=dtype, device=packed.device)
    if out.numel():
        kern, cdiv = _triton_decode_kernel()
        kern[(p.shape[0], cdiv(half, 1024))](p, s, out, half, p.stride(0), s.stride(0), BLOCK=1024)
    return out.reshape(*lead, k)


def decode_nvfp4(packed: torch.Tensor, scale: torch.Tensor,
                 dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """W[o, i] = E2M1(code[o, i]) * E4M3(scale[o, i // 16]), exact in BF16.

    One byte-LUT gather (int32 index) and one broadcast multiply in ``dtype``:
    E2M1 x E4M3 has at most 6 significant bits, so the product is exact in
    BF16/FP32 and equals the FP32 product rounded to ``dtype``. CUDA uint8 inputs
    take the fused Triton kernel (same bits, one memory pass).
    """
    if (packed.is_cuda and packed.dtype == torch.uint8 and scale.dtype == torch.uint8
            and dtype in (torch.bfloat16, torch.float16, torch.float32)):
        return _decode_nvfp4_fused(packed, scale, dtype)
    return _decode_nvfp4_lut(packed, scale, dtype)


def _decode_nvfp4_lut(packed: torch.Tensor, scale: torch.Tensor,
                      dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    *lead, half = packed.shape
    k = 2 * half
    vals = F.embedding(packed.to(torch.int32), _byte_lut(packed.device, dtype))
    sf = e4m3_bits_to_float(scale)[..., : k // NVFP4_BLOCK].to(dtype)
    vals = vals.reshape(*lead, k // NVFP4_BLOCK, NVFP4_BLOCK) * sf[..., None]
    return vals.reshape(*lead, k)


# --------------------------------------------------------------------------
# Activation quantizers (forward values only; no autograd)
# --------------------------------------------------------------------------

# vLLM ``NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE`` (``flashinfer_utils.py``)
NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE = 1.0 / (448.0 * 6.0)


@torch.no_grad()
def nvfp4_per_token_quant(x: torch.Tensor, shards: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token NVFP4 quant-dequant of ``x`` [..., K], the last dim split into
    ``shards`` equal parts with one scale each (FC2: one per rollout TP rank; FC1: 1).

    CUDA: FlashInfer's own kernel, called as vLLM's ``quantize_nvfp4_per_token_input``
    (``nvfp4_quantize(x, 1/(448*6), sfLayout=layout_linear, per_token_activation=True)``,
    no swizzle; the FC1 path of ``TrtLlmNvFp4Experts`` with
    ``moe.activation=nvfp4_per_token``) on the [tokens * shards, K / shards] view
    (rows are independent, so this equals one call per shard; FC2 runs the same
    ``nvfp4QuantAndPerTokenScaleKernel`` inside the TRTLLM MoE on the BF16-staged FC1
    output, trtllm_fused_moe_runner.cu:754, unless FLASHINFER_NVFP4_4OVER6). The codes and E4M3
    block scales are decoded by lookup and multiply. The kernel takes BF16/FP16 only.
    CPU, or FP32 input (the FP32 validation runs of G2; rollout never quantizes
    FP32): ``_nvfp4_per_token_quant_reference``, the same rule in FP32 torch ops
    (on CUDA with the kernel's ``rcp.approx``: bitwise FlashInfer's codes and scales).

    Returns (q * sf as FP32 [..., K] -- exact in BF16 -- and the per-token scale s
    [..., shards]); the represented value is (q * sf) * s. Rows with amax = 0 give
    q = 0, s = 0.
    """
    *lead, K = x.shape
    if K % shards or (K // shards) % NVFP4_BLOCK:
        raise ValueError(f"K={K} not divisible into {shards} shards of 16-blocks")
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return _nvfp4_per_token_quant_reference(x, shards)
    width = K // shards
    rows = x.numel() // width
    if rows == 0:
        return (x.new_zeros(x.shape, dtype=torch.float32),
                x.new_zeros((*lead, shards), dtype=torch.float32))
    from flashinfer import SfLayout, nvfp4_quantize

    q, sf, s = nvfp4_quantize(
        x.reshape(rows, width).contiguous(),
        NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
    )
    sf = sf.view(torch.uint8).reshape(rows, -1)[:, : width // NVFP4_BLOCK]
    dq = decode_nvfp4(q.view(torch.uint8).reshape(rows, width // 2), sf, torch.float32)
    return dq.reshape(x.shape), s.float().reshape(*lead, shards)


_RCP_KERNEL = None


def _triton_rcp_kernel():
    """Triton kernel running PTX ``rcp.approx.ftz.f32`` (FlashInfer's
    ``reciprocal_approximate_ftz``)."""
    global _RCP_KERNEL
    if _RCP_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _rcp(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = off < n
            x = tl.load(x_ptr + off, mask=m, other=1.0)
            y = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x],
                                          dtype=tl.float32, is_pure=True, pack=1)
            tl.store(y_ptr + off, y, mask=m)

        _RCP_KERNEL = (_rcp, triton.cdiv)
    return _RCP_KERNEL


def rcp_approx(x: torch.Tensor) -> torch.Tensor:
    """FP32 reciprocal as FlashInfer's NVFP4 quantizer computes it: PTX
    ``rcp.approx.ftz.f32`` on CUDA (Triton), IEEE ``1 / x`` elsewhere (the
    approximate instruction is not reproducible off-GPU; at most 1 ulp apart)."""
    x = x.float()
    if not x.is_cuda:
        return 1.0 / x
    kern, cdiv = _triton_rcp_kernel()
    xc = x.contiguous()
    y = torch.empty_like(xc)
    n = xc.numel()
    if n:
        kern[(cdiv(n, 1024),)](xc, y, n, BLOCK=1024)
    return y


def _nvfp4_per_token_codes_reference(x: torch.Tensor, shards: int, rcp=rcp_approx):
    """FlashInfer's ``nvfp4QuantAndPerTokenScaleKernel`` in FP32 torch ops, fast-math
    order, per (token, shard):
        s   = amax * f32(1/2688)                 (per-token decode scale)
        enc = rcp(s)
        sf  = E4M3_rn_sat(enc * (vmax_16 * rcp(6)))       (per 16 columns)
        out = rcp(sf * rcp(enc))
        q   = E2M1_rne_sat(x * out)
    ``rcp`` = ``rcp.approx.ftz.f32`` on CUDA (bitwise the kernel's rule), IEEE 1/x on CPU.
    Returns FP32 (q [..., shards, K/shards/16, 16], sf [..., shards, K/shards/16, 1],
    s [..., shards, 1]).
    """
    *lead, K = x.shape
    xf = x.float().reshape(*lead, shards, K // shards)
    amax = xf.abs().amax(-1, keepdim=True)
    s = amax * (xf.new_tensor(1.0) / xf.new_tensor(FP8_MAX * E2M1_MAX))
    zero = torch.zeros_like(s)
    enc = torch.where(s != 0, rcp(s), zero)
    inv_enc = torch.where(enc != 0, rcp(enc), zero)[..., None]
    blk = xf.reshape(*lead, shards, K // shards // NVFP4_BLOCK, NVFP4_BLOCK)
    vmax = blk.abs().amax(-1, keepdim=True)
    sf = fp8_round(enc[..., None] * (vmax * rcp(xf.new_tensor([E2M1_MAX]))))
    out = rcp(sf * inv_enc)
    out = torch.where((vmax != 0) & out.isfinite(), out, torch.zeros_like(out))
    return e2m1_round(blk * out), sf, s


def _nvfp4_per_token_quant_reference(x: torch.Tensor, shards: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``nvfp4_per_token_quant`` without FlashInfer (CPU tests, and FP32 inputs on CUDA:
    the G2 FP32 validation runs; rollout never quantizes FP32): the kernel's rule
    (``_nvfp4_per_token_codes_reference``), with ``rcp.approx`` on CUDA."""
    *lead, K = x.shape
    q, sf, s = _nvfp4_per_token_codes_reference(x, shards)
    return (q * sf).reshape(*lead, K), s[..., 0]


@torch.no_grad()
def fp8_block_act_quant(x: torch.Tensor, block: int = FP8_ACT_BLOCK) -> torch.Tensor:
    """Per-(token, ``block`` columns) FP8 E4M3 quant-dequant with power-of-two
    (UE8M0) scales, as vLLM's W8A8 block linear on sm100 (DeepGEMM) quantizes
    its input. Returns FP32 values, exact in BF16.

    CUDA: vLLM's own op, called as ``QuantFP8.forward_cuda`` does for DeepGEMM with
    UE8M0 scales: ``per_token_group_quant_fp8_packed_for_deepgemm(x, 128,
    use_ue8m0=True)``; the packed int32 scales (4 exponent bytes each, group g in
    byte g % 4 of word g // 4) are decoded to 2^(e - 127). K must be a multiple
    of ``block``. CPU, or FP32 input (the vLLM op takes BF16/FP16 only; FP32 is used
    by the G2 validation runs): ``_fp8_block_act_quant_reference``.
    """
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return _fp8_block_act_quant_reference(x, block)
    *lead, K = x.shape
    if K % block:
        raise ValueError(f"K={K} not a multiple of the FP8 activation group {block}")
    G = K // block
    rows = x.numel() // K
    if rows == 0:
        return x.new_zeros(x.shape, dtype=torch.float32)
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8_packed_for_deepgemm,
    )

    xq, s_packed = per_token_group_quant_fp8_packed_for_deepgemm(x, block, use_ue8m0=True)
    e = s_packed.contiguous().view(torch.uint8).reshape(rows, -1)[:, :G]
    scale = (e.to(torch.int32) << 23).view(torch.float32)  # 2^(e - 127), e in [1, 254]
    out = xq.reshape(rows, G, block).float() * scale[..., None]
    return out.reshape(x.shape)


def _fp8_block_act_quant_reference(x: torch.Tensor, block: int) -> torch.Tensor:
    """Test-only CPU reference of vLLM's per-token-group UE8M0 FP8 quantizer:
    scale = 2^ceil(log2(max(amax, 1e-10) / 448)) per (token, ``block`` columns),
    q = E4M3_sat(x / scale). A last partial group uses its own amax."""
    *lead, K = x.shape
    nb = -(-K // block)
    xf = x.float()
    if nb * block != K:
        xf = F.pad(xf, (0, nb * block - K))
    g = xf.reshape(*lead, nb, block)
    amax = g.abs().amax(-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / FP8_MAX)))
    return (fp8_round(g / scale) * scale).reshape(*lead, nb * block)[..., :K]


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xq):
        return xq.to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None


def ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Forward value ``xq`` (cast to x.dtype), identity gradient to ``x``."""
    return _STE.apply(x, xq.detach())


# --------------------------------------------------------------------------
# NVFP4 GEMM with STE backward
# --------------------------------------------------------------------------


def _mm_f32(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a [n, k] @ w[o, k]^T with FP32 output. Inputs hold values exact in BF16;
    on CUDA use a BF16 GEMM with an FP32 output if the build supports it."""
    if a.is_cuda:
        try:
            return torch.mm(a.to(torch.bfloat16), w.to(torch.bfloat16).t(),
                            out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            pass
    return a.float() @ w.float().t()


class _NVFP4Matmul(torch.autograd.Function):
    """y = x @ W^T * ws2 with packed W.

    shards=None: plain (QAT off), BF16/FP32 F.linear in x.dtype, then x ws2,
    output x.dtype. shards=k: per-token NVFP4 of x over k shards of K, FP32
    output = sum_r (q_r*sf_r) @ W_r^T * s_r * ws2.
    Backward (both): grad_x = (g * ws2) @ W (STE through the quantizer).
    Packed tensors are re-decoded in backward instead of saving the BF16 weight.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, ws2, shards, xq=None):
        ctx.save_for_backward(packed, scale, ws2)
        ctx.x_dtype = x.dtype
        n_out = packed.shape[0]
        if x.shape[0] == 0:  # expert without tokens: no decode, no GEMM
            dt = x.dtype if shards is None else torch.float32
            return x.new_zeros((0, n_out), dtype=dt)
        w = decode_nvfp4(packed, scale, torch.bfloat16)
        if shards is None:
            return (F.linear(x, w.to(x.dtype)).float() * ws2).to(x.dtype)
        # xq: nvfp4_per_token_quant(x, shards) computed by the caller (row-wise,
        # so quantizing all tokens once and gathering rows is the same value).
        dq, s = nvfp4_per_token_quant(x, shards) if xq is None else xq
        K = x.shape[-1]
        width = K // shards
        out = None
        for r in range(shards):
            sl = slice(r * width, (r + 1) * width)
            part = _mm_f32(dq[:, sl], w[:, sl]) * s[:, r : r + 1]
            out = part if out is None else out + part
        return out * ws2

    @staticmethod
    def backward(ctx, g):
        packed, scale, ws2 = ctx.saved_tensors
        if g.shape[0] == 0:
            return g.new_zeros((0, 2 * packed.shape[1]), dtype=ctx.x_dtype), None, None, None, None, None
        w = decode_nvfp4(packed, scale, torch.float32)
        gx = (g.float() * ws2) @ w
        return gx.to(ctx.x_dtype), None, None, None, None, None


class NVFP4Linear(nn.Linear):
    """Frozen NVFP4 linear with the checkpoint's packed tensors.

    Subclasses ``nn.Linear`` (``in_features``/``out_features``) so PEFT wraps it
    as a LoRA ``Linear``; the packed weight and scales are frozen uint8
    parameters (FSDP2-shardable along dim 0), ``weight_scale_2`` and
    ``input_scale`` are FP32 scalar buffers kept FP32 under ``.to(dtype)``.
    """

    _fp32_buffers = ("weight_scale_2", "input_scale")

    def __init__(self, in_features: int, out_features: int, device=None) -> None:
        nn.Module.__init__(self)
        if in_features % NVFP4_BLOCK:
            raise ValueError(f"in_features {in_features} not a multiple of 16")
        self.in_features, self.out_features = in_features, out_features
        u8 = dict(dtype=torch.uint8, device=device)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features // 2, **u8),
                                   requires_grad=False)
        self.weight_scale = nn.Parameter(
            torch.zeros(out_features, in_features // NVFP4_BLOCK, **u8), requires_grad=False)
        self.register_parameter("bias", None)
        f32 = dict(dtype=torch.float32, device=device)
        self.register_buffer("weight_scale_2", torch.ones((), **f32))
        self.register_buffer("input_scale", torch.ones((), **f32))

    def reset_parameters(self) -> None:
        pass

    def _apply(self, fn, recurse=True):
        keep = {k: self._buffers[k] for k in self._fp32_buffers}
        super()._apply(fn, recurse)
        for k, old in keep.items():
            new = self._buffers[k]
            if new.dtype != torch.float32:
                self._buffers[k] = old.to(device=new.device)
        return self

    def decoded_weight(self, dtype=torch.bfloat16) -> torch.Tensor:
        return decode_nvfp4(self.weight, self.weight_scale, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NVFP4Matmul.apply(x, self.weight, self.weight_scale, self.weight_scale_2, None)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4"


class FP8ActQuantLinear(nn.Linear):
    """FP8_PB_WO linear (BF16 dequantized weight). With ``act_quant`` the input
    is fake-quantized as vLLM's W8A8 block GEMM does (STE backward)."""

    act_quant: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_quant:
            x = ste(x, fp8_block_act_quant(x))
        return F.linear(x, self.weight, self.bias)


# --------------------------------------------------------------------------
# Routed-expert forward (QAT on)
# --------------------------------------------------------------------------


def _split_lora(mod: nn.Module):
    """(base module, LoRA delta fn or None) for a plain or PEFT-wrapped linear.

    delta(x) = sum over active adapters of lora_B(lora_A(dropout(x))) * scaling,
    computed as PEFT does and returned in x.dtype.
    """
    if not (hasattr(mod, "base_layer") and hasattr(mod, "lora_A")):
        return mod, None
    base = mod.get_base_layer()
    if mod.merged:
        raise RuntimeError("merged LoRA on a packed NVFP4 weight is not supported")
    if mod.disable_adapters:
        return base, None
    active = [a for a in mod.active_adapters if a in mod.lora_A]
    for a in active:
        if a in getattr(mod, "lora_variant", {}):
            raise NotImplementedError("only vanilla LoRA on NVFP4 experts")
    if not active:
        return base, None

    def delta(x: torch.Tensor) -> torch.Tensor:
        out = None
        for a in active:
            A, B = mod.lora_A[a], mod.lora_B[a]
            d = B(A(mod.lora_dropout[a](x.to(A.weight.dtype)))) * mod.scaling[a]
            out = d if out is None else out + d
        return out.to(x.dtype)

    return base, delta


def nvfp4_expert(expert: nn.Module, x: torch.Tensor, route_w: torch.Tensor,
                 fc2_shards: int, xq=None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One routed expert on its tokens, vLLM TRTLLM NVFP4 LoRA order.

    x [n, H] (activation dtype), route_w [n] FP32. Returns (base FC2 output
    rounded to x.dtype, unweighted; down-LoRA delta x route_w rounded to
    x.dtype, or None). The caller adds w * base + delta in FP32.
    ``xq``: optional ``nvfp4_per_token_quant(x, 1)`` rows (FC1 input quant
    hoisted out of the expert loop by the caller).
    """
    g, g_delta = _split_lora(expert.gate_proj)
    u, u_delta = _split_lora(expert.up_proj)
    d, d_delta = _split_lora(expert.down_proj)
    for m in (g, u, d):
        if not isinstance(m, NVFP4Linear):
            raise TypeError(f"expected NVFP4Linear, got {type(m).__name__}")
    inter = g.out_features
    # w13 uses the gate's weight_scale_2 for up too (loader asserts equality).
    h = _NVFP4Matmul.apply(
        x,
        torch.cat([g.weight, u.weight]),
        torch.cat([g.weight_scale, u.weight_scale]),
        g.weight_scale_2,
        1,
        xq,
    )
    hg, hu = h[:, :inter], h[:, inter:]
    if g_delta is not None:
        hg = hg + g_delta(x).float()
    if u_delta is not None:
        hu = hu + u_delta(x).float()
    a = (F.silu(hg) * hu).to(x.dtype)
    y = _NVFP4Matmul.apply(a, d.weight, d.weight_scale, d.weight_scale_2, fc2_shards)
    y = y.to(x.dtype)
    dd = None
    if d_delta is not None:
        dd = (d_delta(a).float() * route_w[:, None]).to(x.dtype)
    return y, dd


# --------------------------------------------------------------------------
# Module installation from the checkpoint's quantization_config
# --------------------------------------------------------------------------


ROLLOUT_MOE_ACTIVATION = "nvfp4_per_token"


def _check_rollout_moe_activation(config) -> None:
    act = getattr(config, "rollout_moe_activation", None)
    if act != ROLLOUT_MOE_ACTIVATION:
        raise ValueError(
            f"nvfp4_act_qat models vLLM's per-token NVFP4 MoE activation and ignores the "
            f"checkpoint input_scale; set config.rollout_moe_activation={ROLLOUT_MOE_ACTIVATION!r} "
            f"and serve with quantization_config.moe.activation={ROLLOUT_MOE_ACTIVATION!r} "
            f"(got rollout_moe_activation={act!r})"
        )


def _rollout_shards(config, inter: int) -> int:
    tp = getattr(config, "rollout_tp", None)
    mtp = getattr(config, "rollout_moe_tp", None)
    if not isinstance(tp, int) or not isinstance(mtp, int) or tp < 1:
        raise ValueError(
            "nvfp4_act_qat needs integer config.rollout_tp and config.rollout_moe_tp "
            f"(the vLLM rollout TP size), got {tp!r}, {mtp!r}"
        )
    if mtp != tp:
        raise ValueError(f"rollout_moe_tp={mtp} must equal rollout_tp={tp} (no EP support)")
    if inter % mtp or (inter // mtp) % NVFP4_BLOCK:
        raise ValueError(f"moe_intermediate_size={inter} not splittable into {mtp} 16-blocks")
    return mtp


def install_quant_modules(model: nn.Module, config) -> None:
    """Swap in checkpoint-format modules for a ModelOpt MIXED_PRECISION
    checkpoint. ``model`` is the decoder (``ML4Model``; names ``layers.N...``).
    No-op for other checkpoints (the FP8 ``quantization`` path is unchanged).
    """
    q = getattr(config, "checkpoint_quantization", None) or {}
    if q.get("quant_algo") != "MIXED_PRECISION":
        return
    if q.get("quant_method") != "modelopt":
        raise ValueError(f"unsupported quant_method {q.get('quant_method')!r}")
    if q.get("ignore"):
        raise NotImplementedError(f"quantization_config.ignore={q['ignore']}")
    nvfp4_qat = bool(getattr(config, "nvfp4_act_qat", True))
    fp8_qat = bool(getattr(config, "fp8_act_qat", True))
    mods = dict(model.named_modules())
    shards = None
    for key, spec in q["quantized_layers"].items():
        algo = spec["quant_algo"]
        name = map_name(key)[0].removeprefix("language_model.model.")
        if name not in mods:
            raise ValueError(f"quantized layer {key} -> {name}: no such module")
        mod = mods[name]
        if algo == "NVFP4":
            if spec.get("group_size", q.get("group_size")) != NVFP4_BLOCK:
                raise ValueError(f"{key}: NVFP4 group_size must be 16")
            if not name.endswith(".mlp.experts"):
                raise ValueError(f"{key}: NVFP4 only supported on routed experts")
            dev = next(mod.parameters()).device
            for expert in mod:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    old = getattr(expert, proj)
                    setattr(expert, proj, NVFP4Linear(old.in_features, old.out_features, dev))
            moe = mods[name.removesuffix(".experts")]
            if nvfp4_qat:
                if shards is None:
                    _check_rollout_moe_activation(config)
                    shards = _rollout_shards(config, config.moe_intermediate_size)
                moe.nvfp4_fc2_shards = shards
        elif algo == "FP8_PB_WO":
            if type(mod) is not nn.Linear:
                raise ValueError(f"{key}: expected nn.Linear, got {type(mod).__name__}")
            mod.__class__ = FP8ActQuantLinear
            mod.act_quant = fp8_qat
        else:
            raise ValueError(f"{key}: unsupported quant_algo {algo}")


# --------------------------------------------------------------------------
# Routed experts as grouped GEMMs (QAT on)
# --------------------------------------------------------------------------


def grouped_mm(a: torch.Tensor, b: torch.Tensor, counts: list[int],
               offs: torch.Tensor) -> torch.Tensor:
    """Rows of ``a`` [M, K] in consecutive groups of ``counts``; group g times ``b[g]``
    [K, N]. Output in ``a.dtype`` (FP32 accumulation). CUDA: one ``torch._grouped_mm``
    (as torchtitan's grouped experts); elsewhere one ``mm`` per group."""
    if a.is_cuda:
        return torch._grouped_mm(a, b, offs=offs)
    outs, s = [], 0
    for g, n in enumerate(counts):
        outs.append(a[s : s + n] @ b[g])
        s += n
    return torch.cat(outs)


def _size_classes(counts: list[int]) -> list[tuple[list[int], int]]:
    """Groups (all counts > 0) by padded row count P = next power of two (>= 16):
    [(group indices, P)]. Padding stays below 2x the rows."""
    classes: dict[int, list[int]] = {}
    for g, n in enumerate(counts):
        classes.setdefault(max(16, 1 << (n - 1).bit_length()), []).append(g)
    return [(gs, P) for P, gs in sorted(classes.items())]


def _bmm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [G, P, K] @ b [G, K, N], FP32 output; inputs hold values exact in BF16
    (``_mm_f32`` batched: BF16 tensor-core GEMM with FP32 output on CUDA)."""
    if a.is_cuda:
        return torch.bmm(a.to(torch.bfloat16), b.to(torch.bfloat16), out_dtype=torch.float32)
    return a.float() @ b.float()


def _size_class_plan(counts: list[int], device: torch.device) -> list[tuple[torch.Tensor, ...]]:
    """Per size class of ``counts``: (group ids [G], padded row indices [G, P], positions of
    the real rows in the flattened [G * P] block, their output rows). Built on the host from
    the host-side ``counts`` and moved in one non-blocking copy from pinned memory: per-class
    ``torch.tensor(..., device=cuda)`` and boolean-mask indexing each wait for the device."""
    starts = list(itertools.accumulate([0] + counts[:-1]))
    shapes, parts = [], []
    for groups, P in _size_classes(counts):
        gi = torch.tensor(groups, dtype=torch.long)
        st = torch.tensor([starts[g] for g in groups], dtype=torch.long)[:, None]
        cn = torch.tensor([counts[g] for g in groups], dtype=torch.long)[:, None]
        ar = torch.arange(P, dtype=torch.long)[None]
        valid = ar < cn
        idx = torch.where(valid, st + ar, st)
        src = valid.flatten().nonzero().squeeze(1)
        dst = idx.flatten()[src]
        shapes.append((len(groups), P, src.numel()))
        parts += [gi, idx.flatten(), src, dst]
    flat = torch.cat(parts)
    if device.type == "cuda":
        flat = flat.pin_memory().to(device, non_blocking=True)
    else:
        flat = flat.to(device)
    plan, off = [], 0
    for G, P, n in shapes:
        sizes = (G, G * P, n, n)
        gi, idx, src, dst = flat[off : off + sum(sizes)].split(sizes)
        off += sum(sizes)
        plan.append((gi, idx.view(G, P), src, dst))
    return plan


def _by_size_class(counts: list[int], n_cols: int, like: torch.Tensor, fn) -> torch.Tensor:
    """FP32 [M, n_cols] for rows grouped by ``counts`` (all > 0), filled per size class:
    ``fn(gi, idx)`` gets the class's group ids [G] and padded row indices [G, P]
    (padding repeats a real row of the group) and returns FP32 [G, P, n_cols]; the
    padding rows are dropped."""
    out = like.new_empty((sum(counts), n_cols), dtype=torch.float32)
    for gi, idx, src, dst in _size_class_plan(counts, like.device):
        res = fn(gi, idx)
        out[dst] = res.reshape(-1, res.shape[-1])[src]
    return out


class _GroupedNVFP4Matmul(torch.autograd.Function):
    """``_NVFP4Matmul`` (QAT on) for a group of experts with batched GEMMs.

    x [M, K] rows sorted by expert (``counts`` > 0 per expert), packed [G, N, K/2], scale
    [G, N, K/16], ws2_rows [M] (each row's expert ``weight_scale_2``), xq =
    ``nvfp4_per_token_quant(x, shards)`` of the rows. Forward, FP32 output as the loop:
        sum_r (q_r*sf_r @ W_r^T in FP32) * s_r, times ws2.
    torch's grouped GEMM only writes its input dtype (BF16), and the FC1 output must stay
    FP32: the FC2 input quantizer follows it, and BF16-rounded FC1 outputs change the
    layer output by ~2% (relative norm) at real dims through flipped FC2 codes. So
    experts of similar row count are padded to a common size (``_size_classes``) and
    run as one ``bmm`` with FP32 output.
    Backward (STE): grad_x = (g * ws2) @ W, FP32 as the loop, with g * ws2 split into
    BF16 hi + lo parts (two tensor-core GEMMs, ~16-bit mantissa), rounded to x.dtype.
    """

    @staticmethod
    def forward(ctx, x, packed, scale, ws2_rows, counts, shards, xq):
        ctx.save_for_backward(packed, scale, ws2_rows)
        ctx.counts, ctx.x_dtype = counts, x.dtype
        dq, s = xq
        if dq.is_cuda:
            dq = dq.to(torch.bfloat16)  # exact
        width = x.shape[-1] // shards

        def fn(gi, idx):
            w = decode_nvfp4(packed[gi], scale[gi], torch.bfloat16)
            a, s_b = dq[idx], s[idx]
            acc = None
            for r in range(shards):
                sl = slice(r * width, (r + 1) * width)
                part = _bmm_f32(a[:, :, sl], w[:, :, sl].transpose(-2, -1)) * s_b[:, :, r : r + 1]
                acc = part if acc is None else acc + part
            return acc * ws2_rows[idx][..., None]

        return _by_size_class(counts, packed.shape[1], x, fn)

    @staticmethod
    def backward(ctx, g):
        packed, scale, ws2_rows = ctx.saved_tensors
        gs = g.float() * ws2_rows[:, None]

        def fn(gi, idx):
            w = decode_nvfp4(packed[gi], scale[gi], torch.bfloat16)
            gb = gs[idx]
            if not gb.is_cuda:
                return gb @ w.float()
            hi = gb.to(torch.bfloat16)
            lo = (gb - hi.float()).to(torch.bfloat16)
            return _bmm_f32(hi, w) + _bmm_f32(lo, w)

        gx = _by_size_class(ctx.counts, 2 * packed.shape[2], g, fn)
        return gx.to(ctx.x_dtype), None, None, None, None, None, None


def _stacked_lora(mods: list):
    """(A [G, r, in], B [G, out, r], scaling) of the single vanilla LoRA adapter on each
    PEFT linear: stacked copies of the per-expert parameters (names unchanged, gradients
    flow back through the stack), or None without an active adapter. Raises on setups
    the per-expert loop handles but this path does not (several adapters, dropout)."""
    parts = [_split_lora(m)[1] for m in mods]
    if all(d is None for d in parts):
        return None
    As, Bs, scales = [], [], set()
    for m, d in zip(mods, parts):
        if d is None:
            raise NotImplementedError("grouped experts: LoRA active on some experts only")
        active = [a for a in m.active_adapters if a in m.lora_A]
        if len(active) != 1:
            raise NotImplementedError("grouped experts: exactly one active LoRA adapter")
        a = active[0]
        if not isinstance(m.lora_dropout[a], nn.Identity):
            raise NotImplementedError("grouped experts: LoRA dropout must be 0")
        As.append(m.lora_A[a].weight)
        Bs.append(m.lora_B[a].weight)
        scales.add(float(m.scaling[a]))
    if len(scales) != 1:
        raise NotImplementedError("grouped experts: different LoRA scaling per expert")
    return torch.stack(As), torch.stack(Bs), scales.pop()


def _grouped_lora(lora, x, counts, offs):
    """PEFT's B(A(x)) * scaling per expert group, in x.dtype (``_split_lora`` delta)."""
    A, B, scaling = lora
    h = grouped_mm(x.to(A.dtype), A.transpose(-2, -1), counts, offs)
    return (grouped_mm(h, B.transpose(-2, -1), counts, offs) * scaling).to(x.dtype)


def nvfp4_experts_grouped(experts, x: torch.Tensor, xq, route_w: torch.Tensor,
                          counts: list[int], fc2_shards: int, chunk: int = 16) -> torch.Tensor:
    """``nvfp4_expert`` for all experts on expert-sorted rows, with batched GEMMs.

    x [M, H] activation rows sorted by expert (``counts[e]`` rows of expert e), xq =
    ``nvfp4_per_token_quant`` of those rows (shards=1), route_w [M] FP32. Returns the
    FP32 contributions ``w * base + delta`` [M, H] in the same row order; the caller
    sums them per token. ``chunk`` experts at a time (a chunk's decoded weights are
    transient), experts without rows skipped. Same operations, dtypes and order as the
    loop: base NVFP4 GEMMs with FP32 output (``_GroupedNVFP4Matmul``), LoRA GEMMs as
    ``torch._grouped_mm`` in the LoRA dtype (PEFT's F.linear rounding). Differences:
    GEMM kernels / reduction order, and grad_x of the base GEMMs takes g * ws2 rounded
    to x.dtype.
    """
    outs = []
    start = 0
    for c0 in range(0, len(experts), chunk):
        sel = [e for e in range(c0, min(c0 + chunk, len(experts))) if counts[e]]
        if not sel:
            continue
        cnt = [counts[e] for e in sel]
        m = sum(cnt)
        rows = slice(start, start + m)
        start += m
        offs = torch.tensor(list(itertools.accumulate(cnt)), dtype=torch.int32, device=x.device)
        cnt_t = torch.tensor(cnt, device=x.device)
        exs = [experts[e] for e in sel]
        g = [_split_lora(ex.gate_proj)[0] for ex in exs]
        u = [_split_lora(ex.up_proj)[0] for ex in exs]
        d = [_split_lora(ex.down_proj)[0] for ex in exs]
        for mod in (*g, *u, *d):
            if not isinstance(mod, NVFP4Linear):
                raise TypeError(f"expected NVFP4Linear, got {type(mod).__name__}")
        inter = g[0].out_features
        xr, rw = x[rows], route_w[rows]
        # w13 uses the gate's weight_scale_2 for up too (loader asserts equality).
        ws13 = torch.stack([m_.weight_scale_2 for m_ in g]).repeat_interleave(cnt_t, output_size=m)
        ws2 = torch.stack([m_.weight_scale_2 for m_ in d]).repeat_interleave(cnt_t, output_size=m)
        h = _GroupedNVFP4Matmul.apply(
            xr,
            torch.stack([torch.cat([a.weight, b.weight]) for a, b in zip(g, u)]),
            torch.stack([torch.cat([a.weight_scale, b.weight_scale]) for a, b in zip(g, u)]),
            ws13, cnt, 1, (xq[0][rows], xq[1][rows]),
        )
        hg, hu = h[:, :inter], h[:, inter:]
        lg = _stacked_lora([ex.gate_proj for ex in exs])
        lu = _stacked_lora([ex.up_proj for ex in exs])
        if lg is not None:
            hg = hg + _grouped_lora(lg, xr, cnt, offs).float()
        if lu is not None:
            hu = hu + _grouped_lora(lu, xr, cnt, offs).float()
        a = (F.silu(hg) * hu).to(x.dtype)
        y = _GroupedNVFP4Matmul.apply(
            a,
            torch.stack([m_.weight for m_ in d]),
            torch.stack([m_.weight_scale for m_ in d]),
            ws2, cnt, fc2_shards, nvfp4_per_token_quant(a, fc2_shards),
        ).to(x.dtype)
        contrib = y.float() * rw[:, None]
        ld = _stacked_lora([ex.down_proj for ex in exs])
        if ld is not None:
            dd = (_grouped_lora(ld, a, cnt, offs).float() * rw[:, None]).to(x.dtype)
            contrib = contrib + dd.float()
        outs.append(contrib)
    return torch.cat(outs) if outs else x.new_zeros(x.shape, dtype=torch.float32)
