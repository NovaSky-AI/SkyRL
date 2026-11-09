"""NVFP4 routed-expert weights for expert_nvfp4, on FlashInfer's ``fp4_quantize`` grid, all experts of a grouped
GEMM in one call.

The grid is the rollouts' (``expert_nvfp4``'s ModelOpt weight sync, FlashInfer ``fp4_quantize`` with vLLM's online
global scale):

- per expert projection one fp32 global scale ``gs = 2688 / max(amax, 1e-8)``, ``amax`` the expert's max|w| from
  ``aminmax``, as ``vllm...online.nvfp4._quantize_moe_weight_to_nvfp4`` computes it (fc1's gate and up share one,
  as vLLM's fused w13); decode scale ``alpha = 1 / gs``;
- ``fp4_quantize(w.reshape(E * N, K), gs per row, sf_vec_size=16)``: E4M3 scales per 16 along K, E2M1 values,
  round-to-nearest; absmax, or 4/6 (FlashInfer's 4over6, MSE error on its fast path; set per call). A per-row
  global scale is bit-identical to one call per expert.

The round trip a bf16 fprop / dgrad multiplies is the decode of those bytes, ``bf16((e2m1 * sf) * alpha)``
(:func:`.decode.dequantize_rows`). The packed operand of the NVFP4 tensor-core GEMM is the same call with 128x4
swizzled scales.

The experts are read as one ``(E, N, K)`` tensor (:func:`stack_experts`). Megatron's DDP maps every parameter into
one contiguous buffer when it owns the params (bf16 params with the distributed optimizer), laid out in reverse
registration order (``params[::-1]``) with each start aligned to 64 elements, so a grouped linear's equally sized
expert weights usually sit back to back, highest expert first, and the stack is a zero-copy view. Otherwise (other
layouts, bucket padding between them) they are stacked into a transient copy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (
    BLOCK_SCALE_FOUR_OVER_SIX,
    check_block_scale,
    flashinfer_block_scale_env,
    nvfp4_global_scale,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.decode import dequantize_rows

# How stack_experts read the experts, for tests and logs.
STACK_STATS = {"views": 0, "copies": 0}


@dataclass
class ExpertStack:
    """``data (E, N, K)``: the experts in memory order, i.e. expert ``E - 1 - i`` at index ``i`` when ``reversed``
    (Megatron's buffer order). ``copied``: a transient stack rather than a view of the parameters."""

    data: torch.Tensor
    reversed: bool
    copied: bool

    def in_expert_order(self, items: Sequence) -> list:
        """A per-index sequence of :attr:`data` (e.g. its ``unbind(0)``) in expert order."""
        return list(items)[::-1] if self.reversed else list(items)


def _uniform_view(weights: Sequence[torch.Tensor]) -> ExpertStack | None:
    w0 = weights[0]
    if len(weights) == 1:
        return ExpertStack(w0.unsqueeze(0), False, False)
    step = w0.numel() * w0.element_size()
    ptrs = [w.data_ptr() for w in weights]
    if ptrs[1] - ptrs[0] == step:
        first, rev = w0, False
    elif ptrs[0] - ptrs[1] == step:
        first, rev = weights[-1], True
    else:
        return None
    d = step if not rev else -step
    if any(b - a != d for a, b in zip(ptrs, ptrs[1:])):
        return None
    storage = w0.untyped_storage()
    if any(w.untyped_storage().data_ptr() != storage.data_ptr() for w in weights):
        return None
    n, k = w0.shape
    view = torch.as_strided(first, (len(weights), n, k), (n * k, k, 1))
    return ExpertStack(view, rev, False)


def stack_experts(weights: Sequence[torch.Tensor]) -> ExpertStack:
    """The ``(N, K)`` bf16 expert weights of one grouped GEMM as one ``(E, N, K)`` tensor: a view when they are laid
    out back to back in one storage (ascending or descending), else a stacked copy."""
    w0 = weights[0]
    for w in weights:
        if w.shape != w0.shape or w.dtype != torch.bfloat16 or not w.is_contiguous() or w.device != w0.device:
            raise ValueError(
                "NVFP4 expert weights must be contiguous bf16 tensors of one shape and device, got "
                f"{[(tuple(w.shape), w.dtype, w.is_contiguous(), str(w.device)) for w in weights]}"
            )
    if w0.dim() != 2 or w0.shape[1] % 16:
        raise ValueError(f"expert weights must be [N, K] with K % 16 == 0, got {tuple(w0.shape)}")
    stack = _uniform_view(weights)
    if stack is None:
        STACK_STATS["copies"] += 1
        return ExpertStack(torch.stack(list(weights)), False, True)
    STACK_STATS["views"] += 1
    return stack


def global_scales(w3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(gs, alpha = 1 / gs)``, fp32 ``(E,)``, of ``(E, N, K)`` expert weights: vLLM's online NVFP4 arithmetic."""
    lo, hi = w3.flatten(1).aminmax(dim=-1)
    return nvfp4_global_scale(torch.maximum(lo.abs(), hi.abs()).to(torch.float32).clamp_min(1e-8))


def quantize(
    w3: torch.Tensor, gs: torch.Tensor, block_scale: str, swizzled: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer ``fp4_quantize`` of ``(E, N, K)`` bf16 weights with per-expert global scales ``gs (E,)``, in one
    call: ``(codes (E, N, K/2) uint8, E4M3 scales)``, the scales ``(E, N, K/16)`` uint8 linear or, ``swizzled``,
    ``(E, N * K/16)`` uint8 128x4-swizzled per expert (``N % 128 == 0``, ``K % 64 == 0``)."""
    from flashinfer import fp4_quantize

    check_block_scale(block_scale)
    e, n, k = w3.shape
    if swizzled and (n % 128 or k % 64):
        raise ValueError(f"swizzled NVFP4 expert scales need N % 128 == 0 and K % 64 == 0, got {(n, k)}")
    with flashinfer_block_scale_env(block_scale == BLOCK_SCALE_FOUR_OVER_SIX):
        q, sf = fp4_quantize(
            w3.reshape(e * n, k), gs.repeat_interleave(n), sf_vec_size=16, sf_use_ue8m0=False,
            is_sf_swizzled_layout=swizzled,
        )  # fmt: skip
    q = q.view(torch.uint8).view(e, n, k // 2)
    sf = sf.view(torch.uint8).reshape(e, -1) if swizzled else sf.view(torch.uint8).view(e, n, k // 16)
    return q, sf


@dataclass
class PackedExperts:
    """One grouped GEMM's experts packed for a block-scaled NVFP4 GEMM: ``q (E, N, K/2)`` codes, ``sf (E, N*K/16)``
    128x4-swizzled E4M3 scales and ``alpha (E,)`` decode scales, indexed in memory order (:class:`ExpertStack`;
    expert ``E - 1 - i`` at ``i`` when ``reversed``). ``sf_linear (E, N, K/16)`` when kept for a dequant."""

    q: torch.Tensor
    sf: torch.Tensor
    alpha: torch.Tensor
    reversed: bool
    sf_linear: torch.Tensor | None = None

    @property
    def num_experts(self) -> int:
        return self.q.shape[0]

    def gemm_expert_index(self, expert: torch.Tensor) -> torch.Tensor:
        """Index into the packed tensors of each entry of ``expert`` (expert ids 0..E-1)."""
        return (self.num_experts - 1) - expert if self.reversed else expert

    def select(self, e0: int, e1: int) -> PackedExperts:
        """Experts ``e0 .. e1 - 1`` (expert order), as their own pack."""
        i0, i1 = (self.num_experts - e1, self.num_experts - e0) if self.reversed else (e0, e1)
        lin = None if self.sf_linear is None else self.sf_linear[i0:i1]
        return PackedExperts(self.q[i0:i1], self.sf[i0:i1], self.alpha[i0:i1], self.reversed, lin)

    def dequantize(self) -> list[torch.Tensor]:
        """The bf16 ``(N, K)`` round trip of each expert, in expert order (bit-identical to
        :func:`fake_quantize_expert_weights` of the weights packed)."""
        if self.sf_linear is None:
            raise ValueError("PackedExperts.dequantize needs the linear scales (pack_expert_weights(keep_linear=True))")
        e, n, k2 = self.q.shape
        y = dequantize_rows(
            self.q.view(e * n, k2), self.sf_linear.view(e * n, -1), self.alpha.repeat_interleave(n)
        ).view(e, n, 2 * k2)
        items = list(y.unbind(0))
        return items[::-1] if self.reversed else items

    @property
    def nbytes(self) -> int:
        ts = (self.q, self.sf, self.alpha) + (() if self.sf_linear is None else (self.sf_linear,))
        return sum(t.numel() * t.element_size() for t in ts)


def fake_quantize_expert_weights(weights: Sequence[torch.Tensor], block_scale: str) -> list[torch.Tensor]:
    """``dequant(quant(w))`` of each ``(N, K)`` bf16 expert weight on the rollouts' grid (see the module docstring):
    one FlashInfer quantize and one decode for all experts. CUDA only."""
    stack = stack_experts(weights)
    w3 = stack.data
    e, n, k = w3.shape
    gs, alpha = global_scales(w3)
    q, sf = quantize(w3, gs, block_scale)
    stack.data = w3 = None  # a stacked copy is freed before the round trip is allocated
    y = dequantize_rows(q.view(e * n, k // 2), sf.view(e * n, k // 16), alpha.repeat_interleave(n)).view(e, n, k)
    return stack.in_expert_order(y.unbind(0))


def pack_expert_weights(weights: Sequence[torch.Tensor], block_scale: str, keep_linear: bool = False) -> PackedExperts:
    """The quantization of :func:`fake_quantize_expert_weights`, packed for the NVFP4 grouped GEMM (128x4-swizzled
    scales; ``N % 128 == 0``, ``K % 64 == 0``). ``keep_linear`` also keeps the linear scales for
    :meth:`PackedExperts.dequantize`."""
    from flashinfer import block_scale_interleave

    stack = stack_experts(weights)
    w3 = stack.data
    e, n, k = w3.shape
    gs, alpha = global_scales(w3)
    if not keep_linear:
        q, sf = quantize(w3, gs, block_scale, swizzled=True)
        return PackedExperts(q, sf, alpha, stack.reversed)
    q, lin = quantize(w3, gs, block_scale)
    sf = block_scale_interleave(lin).view(torch.uint8).reshape(e, -1)
    return PackedExperts(q, sf, alpha, stack.reversed, lin)


def reference_round_trip(w3: torch.Tensor, block_scale: str) -> torch.Tensor:
    """Test reference: the ``(E, N, K)`` round trip from FlashInfer's bytes, decoded in torch."""
    e, n, k = w3.shape
    gs, alpha = global_scales(w3)
    q, sf = quantize(w3, gs, block_scale)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=w3.device)
    v = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).reshape(e, n, k)
    s = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return ((v * s) * alpha[:, None, None]).bfloat16()
