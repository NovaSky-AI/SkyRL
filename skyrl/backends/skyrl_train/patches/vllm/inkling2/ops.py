# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels of SkyRL's BF16 Inkling2 experts, vendored from an early revision of the
Inkling2 vLLM branch: the k-group router GEMM
(``ops/linear.py``), top-k gate and combine (``ops/moe.py``), softcapped SwiGLU
(``ops/activation.py``) and BF16 grouped expert GEMM (``ops/grouped_gemm.py``).

The fork's release branch replaced these with fused NVFP4 / MXFP8 routing, which has no
BF16 mode, so the BF16 experts carry their own. Unchanged from the fork apart from this header;
checked against Megatron by the Inkling2 rows of ``test_megatron_models.py``.
"""

import torch
from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def _k_group_partial_kernel(X, W, Partials, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr):
    # One 1024-wide K group per program. 16-row tiles keep tl.dot on the same
    # mma.sync instruction sequence as the reference's 16x32x32 tiling.
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.program_id(2)
    ks = tl.arange(0, 128)
    partial = tl.zeros((16, BN), tl.float32)
    for tile in range(8):
        k = group * 1024 + tile * 128 + ks
        x = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        w = tl.load(
            W + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        partial = tl.dot(x, w, partial)
    tl.store(
        Partials + (group * M + rows[:, None]) * N + cols[None, :],
        partial,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _k_group_sum_kernel(Partials, Y, MN, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    total = tl.load(Partials + offs, mask)
    for group in range(1, GROUPS):
        total = total + tl.load(Partials + group * MN + offs, mask)
    tl.store(Y + offs, total, mask)


def linear_k_groups(x: torch.Tensor, weight: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``x @ weight.T`` for bf16 ``x`` [T, K] and ``weight`` [N, K].

    Each 1024-wide K group is accumulated in fp32 and the partials are summed in
    order, as in the reference ``linear``.
    """
    assert x.is_contiguous() and weight.is_contiguous()
    tokens, hidden = x.shape
    experts = weight.shape[0]
    groups = triton.cdiv(hidden, 1024)
    partials = torch.empty((groups, tokens, experts), dtype=torch.float32, device=x.device)
    out = torch.empty((tokens, experts), dtype=out_dtype, device=x.device)
    block_n = 64
    grid = (triton.cdiv(tokens, 16), triton.cdiv(experts, block_n), groups)
    _k_group_partial_kernel[grid](
        x,
        weight,
        partials,
        tokens,
        experts,
        hidden,
        block_n,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    _k_group_sum_kernel[(triton.cdiv(out.numel(), 1024),)](partials, out, out.numel(), groups, 1024)
    return out


@triton.jit
def _sum8(x):
    # ((x0 + x2) + (x1 + x3)) + ((x4 + x6) + (x5 + x7)): the summation order of
    # the reference renormalization kernel, so routing weights match bitwise.
    low, high = tl.split(tl.permute(tl.reshape(x, (2, 2, 2)), (0, 2, 1)))
    low, high = tl.split(low + high)
    low, high = tl.split(low + high)
    return low + high


@triton.jit
def _gate_kernel(
    Logits,
    Bias,
    Ids,
    Weights,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    TOPK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, E_POW2)
    mask = offs < E
    logits = tl.load(Logits + token * E + offs, mask, other=0.0)
    sigmoid = tl.div_rn(1.0, 1.0 + tldevice.exp(-logits))
    selection = sigmoid + tl.load(Bias + offs, mask, other=0.0)
    # Sort keys: order-preserving float bits in the high word and the reversed
    # expert index in the low bits, so ties prefer the lower expert index.
    bits = selection.to(tl.uint32, bitcast=True)
    bits = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000)
    keys = (bits.to(tl.uint64) << 16) | (E - offs).to(tl.uint64)
    keys = tl.where(mask, keys, 0)

    slots = tl.arange(0, TOPK)
    ids = tl.zeros((TOPK,), tl.int32)
    selected = tl.zeros((TOPK,), tl.float32)
    for k in tl.static_range(TOPK):
        best = tl.max(keys, 0)
        expert = E - (best & 0xFFFF).to(tl.int32)
        ids = tl.where(slots == k, expert, ids)
        logit = tl.max(tl.where(offs == expert, logits, float("-inf")), 0)
        selected = tl.where(slots == k, logit, selected)
        keys = tl.where(keys == best, 0, keys)

    log_probs = tl.minimum(selected, 0.0) - tldevice.log1p(tldevice.exp(-tl.abs(selected)))
    max_log_probs = tl.max(log_probs, 0)
    shifted = tldevice.exp(log_probs - max_log_probs)
    total = _sum8(shifted) if TOPK == 8 else tl.sum(shifted, 0)
    logsumexp = max_log_probs + tldevice.log(total)
    weights = tldevice.exp(log_probs - logsumexp) * route_scale
    tl.store(Ids + token * TOPK + slots, ids)
    tl.store(Weights + token * TOPK + slots, weights)


def select_experts(
    logits: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns fp32 routing weights and int32 expert ids, both [T, topk]."""
    tokens, experts = logits.shape
    assert logits.is_contiguous() and experts < 1 << 16
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=logits.device)
    weights = torch.empty((tokens, topk), dtype=torch.float32, device=logits.device)
    _gate_kernel[(tokens,)](
        logits,
        bias,
        ids,
        weights,
        route_scale,
        experts,
        triton.next_power_of_2(experts),
        topk,
    )
    return weights, ids


@triton.jit
def _combine_kernel(
    Down,
    Src2Dst,
    RowLo,
    RowHi,
    Out,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < H
    lo = tl.load(RowLo)
    hi = tl.load(RowHi)
    acc = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(TOPK):
        row = tl.load(Src2Dst + token * TOPK + slot).to(tl.int64)
        local = (row >= lo) & (row < hi)
        acc += tl.load(Down + row * H + cols, mask & local, other=0).to(tl.float32)
    tl.store(Out + token * H + cols, acc, mask)


def combine(
    down: torch.Tensor,
    src2dst: torch.Tensor,
    offsets: torch.Tensor,
    topk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Sum each token's rows of ``down`` (rows outside ``offsets`` count 0)."""
    hidden = down.shape[1]
    tokens = src2dst.numel() // topk
    out = torch.empty((tokens, hidden), dtype=out_dtype, device=down.device)
    _combine_kernel[(tokens, triton.cdiv(hidden, 1024))](
        down, src2dst, offsets[:1], offsets[-1:], out, hidden, topk, 1024
    )
    return out


@triton.jit
def _sigmoid_approx(x):
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _softcap(x, CAP: tl.constexpr):
    if CAP > 0:
        x = CAP * tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            [tl.div_rn(x, CAP)],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    return x


@triton.jit
def _swiglu_kernel(
    X,
    Gammas,
    Y,
    RowLo,
    RowHi,
    N: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row < tl.load(RowLo) or row >= tl.load(RowHi):
        return
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    pairs = 2 * cols[:, None] + tl.arange(0, 2)[None, :]
    gate, up = tl.split(tl.load(X + row * (2 * N) + pairs, mask[:, None], other=0).to(tl.float32))
    sigmoid = _sigmoid_approx(gate)
    gate = _softcap(gate, GATE_CAP)
    up = _softcap(up, UP_CAP)
    y = ((gate * sigmoid) * up) * tl.load(Gammas + row)
    tl.store(Y + row * N + cols, y, mask)


def swiglu(
    x: torch.Tensor,
    gammas: torch.Tensor,
    gate_cap: float,
    up_cap: float,
    row_bounds: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """``softcap(gate) * sigmoid(gate) * softcap(up) * gamma`` in fp32 -> bf16.

    ``x`` is [M, 2N] with interleaved (gate, up) columns; rows outside
    ``row_bounds`` are skipped (left uninitialized).
    """
    rows, n = x.shape[0], x.shape[1] // 2
    out = x.new_empty(rows, n)
    _swiglu_kernel[(rows, triton.cdiv(n, 1024))](
        x,
        gammas,
        out,
        *row_bounds,
        N=n,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    return out


@triton.jit
def _grouped_gemm_kernel(
    X,
    W,
    Y,
    Perm,
    Offsets,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
    TOPK: tl.constexpr,
    GATHER: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # Program (n tile, row block): the row block is found from the per-expert
    # block counts, so the grid is a static upper bound and nothing is read
    # back to the host (CUDA graph capture).
    pid = tl.program_id(1)
    experts = tl.arange(0, EXPERTS_POW2)
    starts = tl.load(Offsets + experts, experts < NUM_EXPERTS, other=0)
    ends = tl.load(Offsets + experts + 1, experts < NUM_EXPERTS, other=0)
    blocks = tl.cumsum(tl.cdiv(ends - starts, BM), 0)
    expert = tl.sum((blocks <= pid).to(tl.int32), 0)
    if expert >= NUM_EXPERTS:
        return
    block = pid - tl.max(tl.where(blocks <= pid, blocks, 0), 0)
    rows = (tl.load(Offsets + expert) + block * BM).to(tl.int64) + tl.arange(0, BM)
    row_mask = rows < tl.load(Offsets + expert + 1)
    x_rows = tl.load(Perm + rows, row_mask, other=0) // TOPK if GATHER else rows
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    col_mask = cols < N
    ks = tl.arange(0, BK)
    x_ptrs = X + x_rows[:, None] * K + ks[None, :]
    w_ptrs = W + (expert.to(tl.int64) * N + cols[None, :]) * K + ks[:, None]
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        k_mask = (k + ks) < K
        x = tl.load(x_ptrs, row_mask[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BK
        w_ptrs += BK
    y_ptrs = Y + rows[:, None] * N + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    if ACCUMULATE:
        acc += tl.load(y_ptrs, mask, other=0.0).to(tl.float32)
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask)


def grouped_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    perm: torch.Tensor,
    offsets: torch.Tensor,
    num_rows: int,
    topk: int,
    gather: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out[r] (+)= x[r] @ weight[e].T`` for this rank's expert-sorted rows.

    Same row contract as ``ops.moe.expert_gemm``: rows ``[offsets[0],
    offsets[-1])`` are this rank's, expert ``e``'s rows are ``[offsets[e],
    offsets[e + 1])``; with ``gather`` row ``r`` reads ``x[perm[r] // topk]``.
    ``weight`` is BF16 ``[E, N, K]``. Without ``out``, returns a new
    ``[num_rows, N]`` tensor whose rows outside this rank's range are
    uninitialized; with ``out``, adds into it (FP32 accumulation either way).
    """
    num_experts, n, k = weight.shape
    accumulate = out is not None
    if out is None:
        out = x.new_empty(num_rows, n)
    assert out.is_contiguous() and out.shape == (num_rows, n)
    block_m, block_n, block_k = 64, min(128, max(16, triton.next_power_of_2(n))), 64
    # Every non-empty expert holds at least one row.
    max_blocks = triton.cdiv(num_rows, block_m) + min(num_experts, num_rows) - 1
    if max_blocks > 0:
        _grouped_gemm_kernel[(triton.cdiv(n, block_n), max_blocks)](
            x,
            weight,
            out,
            perm,
            offsets,
            n,
            k,
            num_experts,
            triton.next_power_of_2(num_experts),
            topk,
            gather,
            accumulate,
            block_m,
            block_n,
            block_k,
            num_warps=4,
            num_stages=3,
        )
    return out
