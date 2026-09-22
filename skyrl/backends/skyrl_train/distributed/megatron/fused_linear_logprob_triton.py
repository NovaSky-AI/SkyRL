# ==============================================================================
# Vendored Triton fused linear-cross-entropy / log-prob kernel.
#
# Triton kernels are vendored from:
#
#     volcengine/verl  @ commit 29ffe75
#       verl/utils/kernel/kernels.py
#
# Original Apache-2.0 attribution is reproduced below.
#
# Changes from upstream:
#   * Replaced ``verl.utils.device`` with local torch.cuda helpers.
#   * Made triton optional at import time via ``TRITON_AVAILABLE``.
#   * Added SkyRL's TP-aware ``FusedLinearLogprobTriton`` adapter.
#   * Autotune all reachable kernels with cached, bucketed token-shape keys;
#     exact token counts remain runtime values to avoid recompilation churn.
#   * Compile out entropy-only work in SkyRL's log-prob-only adapter path.
#   * Made epilogue outputs distinct from inputs so repeated tuning runs cannot
#     corrupt reductions.
#   * Use the true row maximum as the epilogue log-sum-exp shift, including
#     ``-inf`` padding, so strongly negative logits remain finite.
#   * Release the oversized d-logits staging buffer before final partial-vocab
#     backward projections.
#   * Removed unused VERL wrappers, reductions, and backward strategies.
#
# Preserve verl's #2656 OOB-vocab masking lines (search ``vocab_bound`` /
# ``logits_for_lse``); they are required for TP correctness.
# ==============================================================================
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Triton fused LM-head log-prob backend for SkyRL.

The module is import-safe without triton; applying the adapter still requires
CUDA + triton.
"""

import typing
from typing import Any

import torch
import torch.distributed as dist

# Local replacements for verl.utils.device.
is_cuda_available = torch.cuda.is_available()


def get_device_name() -> str:
    return "cuda" if is_cuda_available else "cpu"


def get_torch_device():
    return torch.cuda if is_cuda_available else torch.cpu


def get_device_capability(device_id: int = 0):
    if is_cuda_available:
        return torch.cuda.get_device_capability(device_id)
    return (None, None)


# Keep imports working without triton; decorated kernels become no-ops until called.
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
    HAVE_TRITON = True
    SUPPORT_CUDA_TMA = (
        is_cuda_available
        and get_device_capability()[0] is not None
        and get_device_capability()[0] >= 9
        and hasattr(tl, "make_tensor_descriptor")
    )
except ImportError:
    TRITON_AVAILABLE = False
    HAVE_TRITON = False
    SUPPORT_CUDA_TMA = False

if not HAVE_TRITON:
    from unittest.mock import MagicMock

    def null_decorator(*args, **kwargs):
        if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
            return args[0]
        else:

            def inner(func):
                return func

            return inner

    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()

elif SUPPORT_CUDA_TMA:
    # TMA descriptors require a global memory allocation.
    def alloc_fn(size: int, alignment: int, stream: typing.Optional[int]):
        return torch.empty(size, device=get_device_name(), dtype=torch.int8)

    # https://github.com/triton-lang/triton/commit/43625fc968b693ab51884ca95adbcf3e43483fd0
    # Triton 3.5.0 stores allocators in ContextVar; values do not propagate to new
    # threads by default. Set a ContextVar *default* to avoid falling back to
    # NullAllocator in worker threads.
    try:
        import contextvars

        import triton.runtime._allocation as _triton_allocation

        if isinstance(getattr(_triton_allocation, "_allocator", None), contextvars.ContextVar):
            _triton_allocation._allocator = contextvars.ContextVar(
                _triton_allocation._allocator.name,
                default=alloc_fn,
            )
    except (ImportError, AttributeError):
        pass

    triton.set_allocator(alloc_fn)


# ======================================================================================
# Below: adapted from verl/utils/kernel/kernels.py @ 29ffe75 (Apache-2.0).
# ======================================================================================


_AUTOTUNE_MIN_TOKEN_BUCKET = 128
# This saturates only the autotune cache key. Kernels still receive and launch
# over the exact num_tokens, so larger batches are neither truncated nor padded.
_AUTOTUNE_MAX_TOKEN_BUCKET = 1048576

# Matmul specs adapt Triton 3.6's CUDA matmul set (tile sizes, stages, warps):
# https://github.com/triton-lang/triton/blob/v3.6.0/python/tutorials/03-matrix-multiplication.py#L164-L199
_FORWARD_MAINLOOP_CONFIG_SPECS = (
    (128, 256, 32, 2, 8),
    (128, 256, 32, 3, 8),
    (128, 256, 32, 4, 8),
    (128, 256, 32, 5, 8),
    (128, 256, 64, 3, 8),
    (64, 256, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 64, 32, 4, 4),
    (64, 128, 32, 4, 4),
    (64, 64, 32, 3, 4),
    (128, 32, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (32, 64, 32, 5, 2),
    (32, 32, 32, 2, 4),
)

# Reduction tiles follow PyTorch Inductor's max-autotune reduction candidates:
# https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_inductor/runtime/triton_heuristics.py#L3240-L3253
# Specs are (BLOCK_M, BLOCK_N, warps); update specs are (BLOCK_M, warps).
_EPILOGUE_CONFIG_SPECS = (
    (16, 32, 2),
    (16, 64, 4),
    (32, 32, 4),
    (32, 64, 4),
    (32, 128, 4),
    (64, 32, 4),
    (64, 64, 4),
    (64, 128, 8),
    (128, 32, 4),
    (128, 64, 8),
)
_EPILOGUE_UPDATE_CONFIG_SPECS = ((16, 4), (32, 4), (64, 4), (128, 4), (256, 8), (512, 8))

# Backward reuses the Triton matmul ranges above and tunes its grouped ordering;
# specs are (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, stages, warps).
_BACKWARD_CONFIG_SPECS = (
    (128, 128, 32, 16, 2, 8),
    (128, 128, 32, 16, 3, 8),
    (128, 128, 32, 16, 4, 8),
    (128, 128, 32, 16, 5, 8),
    (128, 256, 32, 16, 3, 8),
    (128, 128, 64, 8, 3, 8),
    (64, 256, 32, 8, 4, 4),
    (64, 128, 32, 8, 4, 4),
    (128, 64, 32, 8, 4, 4),
    (64, 64, 32, 8, 3, 4),
    (32, 64, 32, 8, 5, 2),
)


def _autotune_token_bucket(num_tokens: int) -> int:
    """Bucket dynamic counts; shapes above 1M share a tuning decision."""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    power_of_two = 1 << (num_tokens - 1).bit_length()
    return min(_AUTOTUNE_MAX_TOKEN_BUCKET, max(_AUTOTUNE_MIN_TOKEN_BUCKET, power_of_two))


def _matmul_autotune_configs(specs):
    return [
        triton.Config(
            {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m, block_n, block_k, num_stages, num_warps in specs
    ]


def _epilogue_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}, num_warps=num_warps)
        for block_m, block_n, num_warps in _EPILOGUE_CONFIG_SPECS
    ]


def _epilogue_update_autotune_configs():
    return [
        triton.Config({"BLOCK_SIZE_M": block_m}, num_warps=num_warps)
        for block_m, num_warps in _EPILOGUE_UPDATE_CONFIG_SPECS
    ]


def _backward_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": group_m,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for block_m, block_n, block_k, group_m, num_stages, num_warps in _BACKWARD_CONFIG_SPECS
    ]


@triton.autotune(
    configs=_matmul_autotune_configs(_FORWARD_MAINLOOP_CONFIG_SPECS),
    key=["num_tokens_bucket", "hidden_size", "vocab_size", "COMPUTE_ENTROPY"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["num_tokens", "num_tokens_bucket"])
def efficient_entropy_kernel_general_mainloop(
    rank,
    hidden_ptr,
    weight_ptr,
    labels_ptr,
    num_tokens,
    num_tokens_bucket,
    hidden_size,
    vocab_size,
    vocab_per_split,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    rcp_temperature: tl.float32,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_TMA: tl.constexpr,
    COMPUTE_ENTROPY: tl.constexpr,
    # Tests can force IEEE fp32; production uses Triton's fast default precision.
    INPUT_PRECISION: tl.constexpr,
):
    """forward mainloop"""
    pid = tl.program_id(axis=0)
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # create pointers for the first blocks of hidden
    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )

        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)

    # load labels for this block
    labels = tl.load(labels_ptr + offs_am, mask=offs_am < num_tokens)

    # traverse over N dimension
    # _max = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    _accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    if COMPUTE_ENTROPY:
        _entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    _logprobs = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    vocab_bound = min((pid_n + 1) * vocab_per_split, vocab_size)
    for n in range(0, num_pid_n):
        start_offs_bn = pid_n * vocab_per_split + n * BLOCK_SIZE_N
        offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)

        logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if not USE_TMA:
            # weight_ptrs = weight_ptr + (offs_k[:, None] * stride_weight_k + offs_bn[None, :] * stride_weight_n)
            weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)

        # iterate over K dimension
        for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
            if USE_TMA:
                # load the next block of hidden and weight
                start_offs_k = k * BLOCK_SIZE_K
                _hidden = hidden_desc.load([start_offs_am, start_offs_k])
                _weight = weight_desc.load([start_offs_bn, start_offs_k])
            else:
                # load the next block of hidden and weight
                _hidden = tl.load(
                    hidden_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                    other=0.0,
                )

                _weight = tl.load(
                    weight_ptrs,
                    mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K)
                    & (offs_bn[:, None] < (min((pid_n + 1) * vocab_per_split, vocab_size))),
                    other=0.0,
                )

                # advance the ptrs to the next K block
                hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
                weight_ptrs += BLOCK_SIZE_K * stride_weight_k

            # GEMM
            logits = tl.dot(_hidden, _weight.trans(), logits, input_precision=INPUT_PRECISION)

        if not USE_TMA:
            # reset hidden_ptrs for next iteration
            hidden_ptrs -= hidden_size * stride_hidden_k

        # scale logits by temperature
        logits *= rcp_temperature

        # #2656 OOB-vocab fix: mask out-of-bounds vocab columns to -inf for the LSE
        # (max / exp / denominator) only. The unmasked ``logits`` are retained below for
        # the entropy accumulator and the label-logit gather.
        logits_for_lse = tl.where(offs_bn[None, :] < vocab_bound, logits, float("-inf"))

        # update global maximum
        _max_old = _max
        m_pid_n = tl.max(logits_for_lse, axis=1)
        _max = tl.maximum(_max_old, m_pid_n)

        exp_logits = tl.exp(logits_for_lse - _max[:, None])
        coeff = tl.exp(_max_old - _max)
        _accu = coeff * _accu + tl.sum(exp_logits, axis=1)

        if COMPUTE_ENTROPY:
            _entropy_b = _entropy_b * coeff + tl.sum(logits * exp_logits, axis=1)

        label_mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
        _logprobs += tl.sum(logits * label_mask, axis=1)

    # store maximum
    offs_max_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_max_n = pid_n
    maximum_ptrs = max_ptr + offs_max_n * stride_max_n + offs_max_m * stride_max_m
    tl.store(maximum_ptrs, _max, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))

    # store entropy
    accu_ptrs = accu_ptr + offs_max_n * stride_accu_n + offs_max_m * stride_accu_m
    tl.store(accu_ptrs, _accu, mask=(offs_max_m < num_tokens) & (offs_max_n[None] < num_splits))
    if COMPUTE_ENTROPY:
        entropy_b_ptrs = entropy_b_ptr + offs_max_n * stride_entropy_b_n + offs_max_m * stride_entropy_b_m
        tl.store(entropy_b_ptrs, _entropy_b, mask=(offs_max_m < num_tokens) & (offs_max_n < num_splits))
    # store logprobs
    vocab_left_idx = pid_n * vocab_per_split + rank * vocab_size
    vocab_right_idx = min((pid_n + 1) * vocab_per_split, vocab_size) + rank * vocab_size
    mask = (labels >= vocab_left_idx) & (labels < vocab_right_idx)
    mask &= offs_am < num_tokens
    global_logprobs_ptrs = global_logprobs_ptr + offs_am * stride_global_logprobs
    # tl.atomic_add(global_logprobs_ptrs, _logprobs, mask=mask)
    tl.store(global_logprobs_ptrs, _logprobs, mask=mask)


@triton.autotune(
    configs=_epilogue_autotune_configs(),
    key=["num_tokens_bucket", "num_splits", "COMPUTE_ENTROPY"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["num_tokens", "num_tokens_bucket"])
def efficient_entropy_triton_kernel_epilogue(
    max_ptr,
    stride_max_m: tl.int64,
    stride_max_n: tl.int64,
    num_tokens,
    num_tokens_bucket,
    num_splits,
    global_max_ptr,
    stride_global_max: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    global_entropy_ptr,
    stride_global_entropy: tl.int64,
    global_logprobs_ptr,
    stride_global_logprobs: tl.int64,
    result_logprobs_ptr,
    stride_result_logprobs: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMPUTE_ENTROPY: tl.constexpr,
):
    """foward epilogue"""
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # -inf, not 0: a 0 init clamps the softmax shift to max(0, true_max), so a token
    # whose logits are all strongly negative underflows exp() and yields log(0).
    global_max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    if COMPUTE_ENTROPY:
        global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        max_ptrs = max_ptr + offs_m[:, None] * stride_max_m + offs_n[None, :] * stride_max_n

        # -inf padding: with a negative global_max, a 0.0 pad would make
        # exp(0 - global_max) overflow and 0*inf produce NaN.
        _max = tl.load(
            max_ptrs,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=-float("inf"),
        )

        accu_ptrs = accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n
        _accu = tl.load(accu_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0)

        if COMPUTE_ENTROPY:
            entropy_b_ptrs = entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n
            _entropy_b = tl.load(
                entropy_b_ptrs, mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits), other=0.0
            )

        # local reduction
        _max_old = global_max
        _local_max = tl.max(_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        _scale = tl.exp(_max - global_max[:, None])
        _coeff = tl.exp(_max_old - global_max)
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)
        if COMPUTE_ENTROPY:
            global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

    # store
    maximum_ptrs = global_max_ptr + offs_m * stride_global_max
    tl.store(maximum_ptrs, global_max, mask=offs_m < num_tokens)

    global_accu_ptrs = global_accu_ptr + offs_m * stride_global_accu
    tl.store(global_accu_ptrs, global_accu, mask=offs_m < num_tokens)
    if COMPUTE_ENTROPY:
        global_entropy_b = tl.fdiv(global_entropy_b, global_accu)
        tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, global_entropy_b, mask=offs_m < num_tokens)
        global_entropy = tl.log(global_accu) + global_max - global_entropy_b
        global_entropy_ptrs = global_entropy_ptr + offs_m * stride_global_entropy
        tl.store(global_entropy_ptrs, global_entropy, mask=offs_m < num_tokens)
    # update logprobs
    global_logprobs_ptrs = global_logprobs_ptr + offs_m * stride_global_logprobs
    global_logprobs = tl.load(global_logprobs_ptrs, mask=offs_m < num_tokens)
    global_logprobs = global_max + tl.log(global_accu) - global_logprobs

    global_logprobs = -1 * global_logprobs
    tl.store(result_logprobs_ptr + offs_m * stride_result_logprobs, global_logprobs, mask=offs_m < num_tokens)


@triton.autotune(
    configs=_epilogue_autotune_configs(),
    key=["num_tokens_bucket", "num_splits", "COMPUTE_ENTROPY"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["num_tokens", "num_tokens_bucket"])
def efficient_entropy_triton_kernel_epilogue_tp(
    num_tokens,
    num_tokens_bucket,
    num_splits,
    reduced_max_ptr,
    stride_reduced_max_m: tl.int64,
    stride_reduced_max_n: tl.int64,
    original_max_ptr,
    stride_original_max_m: tl.int64,
    stride_original_max_n: tl.int64,
    accu_ptr,
    stride_accu_m: tl.int64,
    stride_accu_n: tl.int64,
    entropy_b_ptr,
    stride_entropy_b_m: tl.int64,
    stride_entropy_b_n: tl.int64,
    global_max_ptr,
    stride_global_max: tl.int64,
    global_accu_ptr,
    stride_global_accu: tl.int64,
    global_entropy_b_ptr,
    stride_global_entropy_b: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    COMPUTE_ENTROPY: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # See the local epilogue: -inf init, and -inf padding on both max operands.
    global_max = tl.full((BLOCK_SIZE_M,), -float("inf"), dtype=tl.float32)
    global_accu = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    if COMPUTE_ENTROPY:
        global_entropy_b = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for pid_n in range(0, tl.cdiv(num_splits, BLOCK_SIZE_N)):
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        _reduced_max = tl.load(
            reduced_max_ptr + offs_m[:, None] * stride_reduced_max_m + offs_n[None, :] * stride_reduced_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=-float("inf"),
        )
        _original_max = tl.load(
            original_max_ptr + offs_m[:, None] * stride_original_max_m + offs_n[None, :] * stride_original_max_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=-float("inf"),
        )
        _accu = tl.load(
            accu_ptr + offs_m[:, None] * stride_accu_m + offs_n[None, :] * stride_accu_n,
            mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
            other=0.0,
        )

        # local reduce-max
        _max_old = global_max
        _local_max = tl.max(_reduced_max, axis=1)
        global_max = tl.maximum(global_max, _local_max)

        # update accumulate
        _coeff = tl.exp(_max_old - global_max)
        _scale = tl.exp(_original_max - global_max[:, None])
        global_accu = _coeff * global_accu + tl.sum(_scale * _accu, axis=1)

        if COMPUTE_ENTROPY:
            _entropy_b = tl.load(
                entropy_b_ptr + offs_m[:, None] * stride_entropy_b_m + offs_n[None, :] * stride_entropy_b_n,
                mask=(offs_m[:, None] < num_tokens) & (offs_n[None, :] < num_splits),
                other=0.0,
            )
            global_entropy_b = _coeff * global_entropy_b + tl.sum(_scale * _entropy_b, axis=1)

    # store
    tl.store(global_max_ptr + offs_m * stride_global_max, global_max, mask=offs_m < num_tokens)
    tl.store(global_accu_ptr + offs_m * stride_global_accu, global_accu, mask=offs_m < num_tokens)
    if COMPUTE_ENTROPY:
        tl.store(global_entropy_b_ptr + offs_m * stride_global_entropy_b, global_entropy_b, mask=offs_m < num_tokens)


@triton.autotune(
    configs=_epilogue_update_autotune_configs(),
    key=["num_tokens_bucket", "COMPUTE_ENTROPY"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["num_tokens", "num_tokens_bucket"])
def efficient_entropy_triton_epilogue_tp_update(
    num_tokens,
    num_tokens_bucket,
    logprobs_ptr,
    stride_logprobs: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accumulate_ptr,
    stride_accumulate: tl.int64,
    entropy_b_ptr,
    stride_entropy_b: tl.int64,
    result_entropy_b_ptr,
    stride_result_entropy_b: tl.int64,
    entropy_ptr,
    stride_entropy: tl.int64,
    result_logprobs_ptr,
    stride_result_logprobs: tl.int64,
    BLOCK_SIZE_M: tl.constexpr,
    COMPUTE_ENTROPY: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    maximum = tl.load(maximum_ptr + offs_m * stride_maximum, mask=offs_m < num_tokens)
    accumulate = tl.load(accumulate_ptr + offs_m * stride_accumulate, mask=offs_m < num_tokens)

    if COMPUTE_ENTROPY:
        entropy_b = tl.load(entropy_b_ptr + offs_m * stride_entropy_b, mask=offs_m < num_tokens)
        entropy_b = tl.fdiv(entropy_b, accumulate)
        tl.store(result_entropy_b_ptr + offs_m * stride_result_entropy_b, entropy_b, mask=offs_m < num_tokens)

        entropy = tl.log(accumulate) + maximum - entropy_b
        tl.store(entropy_ptr + offs_m * stride_entropy, entropy, mask=offs_m < num_tokens)

    logprobs = tl.load(logprobs_ptr + offs_m * stride_logprobs, mask=offs_m < num_tokens)
    logprobs = maximum + tl.log(accumulate) - logprobs

    logprobs = -1 * logprobs
    tl.store(result_logprobs_ptr + offs_m * stride_result_logprobs, logprobs, mask=offs_m < num_tokens)


_dedicated_stream, _dedicated_events = None, None


# Tests set this for tight fp32 parity; production keeps the fast path.
FORCE_FP32_IEEE_PRECISION = False


def _dot_input_precision(hidden: torch.Tensor):
    """Choose ``tl.dot`` input precision; accumulation remains fp32."""
    if FORCE_FP32_IEEE_PRECISION and hidden.dtype == torch.float32:
        return "ieee"
    return "tf32"


def efficient_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    compute_entropy: bool = True,
) -> list:
    """Forward host function; ``compute_entropy=False`` skips entropy-only work."""
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()

    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    _rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    _world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if dist_process_group is not None and not hasattr(efficient_entropy_forward, "_initialized"):
        global _dedicated_stream, _dedicated_events
        _dedicated_stream = get_torch_device().Stream(hidden.device)
        _dedicated_events = [get_torch_device().Event() for _ in range(2)]
        efficient_entropy_forward._initialized = True

    num_tokens, hidden_size = hidden.shape
    num_tokens = labels.shape[0]
    vocab_size, hidden_size = weight.shape
    assert hidden_size % 128 == 0

    logprobs = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)

    maximum = torch.empty((num_tokens,), device=hidden.device, dtype=torch.float32)
    entropy = torch.empty_like(maximum) if compute_entropy else None
    reduction_storage = torch.empty(
        (num_tokens * (2 if compute_entropy else 1),),
        device=hidden.device,
        dtype=torch.float32,
    )
    reduction_storage_view = reduction_storage.view(-1, num_tokens)
    accumulate = reduction_storage_view[0, :]
    reduced_entropy_b = reduction_storage_view[1, :] if compute_entropy else accumulate
    entropy_b = torch.empty_like(maximum) if compute_entropy else None
    # Entropy arguments remain valid pointers even when their uses compile out.
    entropy_output = entropy if entropy is not None else accumulate
    entropy_b_output = entropy_b if entropy_b is not None else accumulate
    assert logprobs.is_contiguous() and maximum.is_contiguous()
    assert accumulate.is_contiguous() and reduced_entropy_b.is_contiguous()

    vocab_per_split = 1024
    assert vocab_per_split % 128 == 0
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split

    _max = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _accu = torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32)
    _entropy_b = (
        torch.empty((num_tokens, num_splits), device=hidden.device, dtype=torch.float32) if compute_entropy else _accu
    )

    # Keep raw target logits separate: autotuned epilogues execute repeatedly.
    _logprobs = torch.zeros((num_tokens,), device=hidden.device, dtype=torch.float32)

    assert _accu.is_contiguous() and _entropy_b.is_contiguous() and _max.is_contiguous()
    assert _accu.is_cuda and _entropy_b.is_cuda and _max.is_cuda

    if TRITON_AVAILABLE:
        # 1D kernel launch, then split the tile
        def mainloop_grid(meta):
            return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * num_splits,)

        efficient_entropy_kernel_general_mainloop[mainloop_grid](
            _rank,
            hidden,
            weight,
            labels,
            num_tokens,
            _autotune_token_bucket(num_tokens),
            hidden_size,
            vocab_size,
            vocab_per_split,
            hidden.stride(0),
            hidden.stride(1),
            weight.stride(0),
            weight.stride(1),
            _max,
            _max.stride(0),
            _max.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            _logprobs,
            _logprobs.stride(0),
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            COMPUTE_ENTROPY=compute_entropy,
            INPUT_PRECISION=_dot_input_precision(hidden),
        )
    else:
        raise AssertionError("Triton is required for efficient entropy kernel")

    # reduction on maximum and maximum_indices
    def epilogue_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]),)

    if dist_process_group is None:
        efficient_entropy_triton_kernel_epilogue[epilogue_grid](
            _max,
            _max.stride(0),
            _max.stride(1),
            num_tokens,
            _autotune_token_bucket(num_tokens),
            num_splits,
            maximum,
            maximum.stride(0),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            accumulate,
            accumulate.stride(0),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            entropy_b_output,
            entropy_b_output.stride(0),
            entropy_output,
            entropy_output.stride(0),
            _logprobs,
            _logprobs.stride(0),
            logprobs,
            logprobs.stride(0),
            COMPUTE_ENTROPY=compute_entropy,
        )
    else:
        # tensor-parallel
        _max_backup = _max.clone()
        dist.all_reduce(_max, op=dist.ReduceOp.MAX, group=dist_process_group)

        get_torch_device().current_stream().record_event(_dedicated_events[0])
        with get_torch_device().stream(_dedicated_stream):
            _dedicated_stream.wait_event(_dedicated_events[0])
            dist.all_reduce(_logprobs, op=dist.ReduceOp.SUM, group=dist_process_group)
            _dedicated_stream.record_event(_dedicated_events[1])

        efficient_entropy_triton_kernel_epilogue_tp[epilogue_grid](
            num_tokens,
            _autotune_token_bucket(num_tokens),
            num_splits,
            _max,
            _max.stride(0),
            _max.stride(1),
            _max_backup,
            _max_backup.stride(0),
            _max_backup.stride(1),
            _accu,
            _accu.stride(0),
            _accu.stride(1),
            _entropy_b,
            _entropy_b.stride(0),
            _entropy_b.stride(1),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            reduced_entropy_b,
            reduced_entropy_b.stride(0),
            COMPUTE_ENTROPY=compute_entropy,
        )
        get_torch_device().current_stream().wait_event(_dedicated_events[1])

        dist.all_reduce(reduction_storage, op=dist.ReduceOp.SUM, group=dist_process_group)

        # update logprobs & entropy
        efficient_entropy_triton_epilogue_tp_update[epilogue_grid](
            num_tokens,
            _autotune_token_bucket(num_tokens),
            _logprobs,
            _logprobs.stride(0),
            maximum,
            maximum.stride(0),
            accumulate,
            accumulate.stride(0),
            reduced_entropy_b,
            reduced_entropy_b.stride(0),
            entropy_b_output,
            entropy_b_output.stride(0),
            entropy_output,
            entropy_output.stride(0),
            logprobs,
            logprobs.stride(0),
            COMPUTE_ENTROPY=compute_entropy,
        )

    return (logprobs, entropy, maximum, accumulate, entropy_b)


@triton.autotune(
    configs=_backward_autotune_configs(),
    key=["num_tokens_bucket", "hidden_size", "vocab_size", "COMPUTE_ENTROPY"],
    cache_results=True,
)
@triton.jit(do_not_specialize=["split_idx", "num_tokens", "num_tokens_bucket"])
def efficient_entropy_backward_kernel_general_d_logits_split_N(
    split_idx: int,
    num_tokens: int,
    num_tokens_bucket: int,
    hidden_size: int,
    vocab_size: int,
    vocab_per_split: int,
    rank: int,
    hidden_ptr,
    stride_hidden_m: tl.int64,
    stride_hidden_k: tl.int64,
    weight_ptr,
    stride_weight_n: tl.int64,
    stride_weight_k: tl.int64,
    labels_ptr,
    stride_labels: tl.int64,
    maximum_ptr,
    stride_maximum: tl.int64,
    accu_ptr,
    stride_accu: tl.int64,
    d_entropy_ptr,
    stride_d_entropy: tl.int64,
    d_logprobs_ptr,
    stride_d_logprobs: tl.int64,
    entropy_b_ptr,
    stride_entropy_b,
    d_logits_ptr,
    stride_d_logits_m: tl.int64,
    stride_d_logits_n: tl.int64,
    rcp_temperature: tl.float32,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_TMA: tl.constexpr,
    COMPUTE_ENTROPY: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(vocab_per_split, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_am = pid_m * BLOCK_SIZE_M
    offs_am = start_offs_am + tl.arange(0, BLOCK_SIZE_M)
    start_offs_bn = split_idx * vocab_per_split + pid_n * BLOCK_SIZE_N
    offs_bn = start_offs_bn + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    maximum = tl.load(maximum_ptr + offs_am * stride_maximum, mask=offs_am < num_tokens, other=0.0)
    accu = tl.load(accu_ptr + offs_am * stride_accu, mask=offs_am < num_tokens, other=1e-6)
    accu_rcp = tl.fdiv(1.0, accu)
    if COMPUTE_ENTROPY:
        d_entropy = tl.load(d_entropy_ptr + offs_am * stride_d_entropy, mask=offs_am < num_tokens, other=0.0)
    d_logprobs = tl.load(d_logprobs_ptr + offs_am * stride_d_logprobs, mask=offs_am < num_tokens, other=0.0)
    d_logprobs = -1 * d_logprobs
    if COMPUTE_ENTROPY:
        entropy_b = tl.load(entropy_b_ptr + offs_am * stride_entropy_b, mask=offs_am < num_tokens, other=0.0)
    labels = tl.load(labels_ptr + offs_am * stride_labels, mask=offs_am < num_tokens, other=0)

    logits = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    if USE_TMA:
        # using TMA and device-side descriptor creation
        hidden_desc = tl.make_tensor_descriptor(
            hidden_ptr,
            shape=[num_tokens, hidden_size],
            strides=[stride_hidden_m, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
        weight_desc = tl.make_tensor_descriptor(
            weight_ptr,
            shape=[vocab_size, hidden_size],
            strides=[stride_weight_n, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    else:
        hidden_ptrs = hidden_ptr + (offs_am[:, None] * stride_hidden_m + offs_k[None, :] * stride_hidden_k)
        weight_ptrs = weight_ptr + (offs_bn[:, None] * stride_weight_n + offs_k[None, :] * stride_weight_k)
        vocab_right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)

    for k in range(0, tl.cdiv(hidden_size, BLOCK_SIZE_K)):
        if USE_TMA:
            start_offs_k = k * BLOCK_SIZE_K
            _hidden = hidden_desc.load([start_offs_am, start_offs_k])
            _weight = weight_desc.load([start_offs_bn, start_offs_k])
        else:
            _hidden = tl.load(
                hidden_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_am[:, None] < num_tokens),
                other=0.0,
            )
            _weight = tl.load(
                weight_ptrs,
                mask=(offs_k[None, :] < hidden_size - k * BLOCK_SIZE_K) & (offs_bn[:, None] < vocab_right_bound),
                other=0.0,
            )
            hidden_ptrs += BLOCK_SIZE_K * stride_hidden_k
            weight_ptrs += BLOCK_SIZE_K * stride_weight_k
        logits = tl.dot(_hidden, _weight.T, logits, input_precision=INPUT_PRECISION)

    logits *= rcp_temperature
    exp_logits = tl.exp(logits - maximum[:, None])

    mask = (offs_bn + rank * vocab_size)[None, :] == labels[:, None]
    d_logits = d_logprobs[:, None] * (exp_logits * accu_rcp[:, None] - mask)
    if COMPUTE_ENTROPY:
        d_logits += d_entropy[:, None] * (-exp_logits * accu_rcp[:, None]) * (logits - entropy_b[:, None])

    d_logits *= rcp_temperature

    # filter d_logits with mask
    result_offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_am[:, None] < num_tokens) & (result_offs_n[None, :] < vocab_per_split)

    tl.store(
        d_logits_ptr + offs_am[:, None] * stride_d_logits_m + result_offs_n[None, :] * stride_d_logits_n, d_logits, mask
    )


def efficient_entropy_backward(
    dlogprobs: torch.Tensor,
    dentropy: typing.Optional[torch.Tensor],
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    maximum: torch.Tensor,
    acc: torch.Tensor,
    entropy_b: typing.Optional[torch.Tensor],
    should_return_fp32_grad: bool = False,
    temperature: typing.Optional[float] = 1.0,
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    compute_entropy: bool = True,
) -> list:
    """Backward host function; returns shard-local ``d_hidden`` without TP all-reduce."""
    assert hidden.is_cuda and weight.is_cuda and labels.is_cuda
    assert weight.device == hidden.device and labels.device == hidden.device
    assert hidden.dim() == 2 and weight.dim() == 2 and labels.dim() == 1
    assert hidden.is_contiguous() and weight.is_contiguous() and labels.is_contiguous()
    assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    num_tokens, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    assert hidden_size % 128 == 0

    assert dlogprobs.shape == (num_tokens,)
    assert dlogprobs.is_contiguous() and dlogprobs.is_cuda
    assert dlogprobs.device == hidden.device
    if compute_entropy:
        assert dentropy is not None and entropy_b is not None
        assert dentropy.is_contiguous() and dentropy.is_cuda
        assert dlogprobs.device == dentropy.device
        assert dentropy.shape == (num_tokens,)

    grad_dtype = torch.float32 if should_return_fp32_grad else hidden.dtype
    d_hidden = torch.empty_like(hidden, dtype=grad_dtype)
    d_weight = torch.empty_like(weight, dtype=grad_dtype)
    assert maximum.shape == labels.shape == acc.shape
    assert maximum.is_contiguous() and acc.is_contiguous()
    if compute_entropy:
        assert entropy_b is not None and entropy_b.shape == labels.shape and entropy_b.is_contiguous()
    d_entropy_input = dentropy if dentropy is not None else dlogprobs
    entropy_b_input = entropy_b if entropy_b is not None else acc

    vocab_per_split = 9504
    num_splits = (vocab_size + vocab_per_split - 1) // vocab_per_split
    d_logits = torch.empty((num_tokens, vocab_per_split), device=hidden.device, dtype=hidden.dtype).contiguous()

    def d_logits_grid(meta):
        return (triton.cdiv(num_tokens, meta["BLOCK_SIZE_M"]) * triton.cdiv(vocab_per_split, meta["BLOCK_SIZE_N"]),)

    for split_idx in range(num_splits):
        efficient_entropy_backward_kernel_general_d_logits_split_N[d_logits_grid](
            split_idx,
            num_tokens,
            _autotune_token_bucket(num_tokens),
            hidden_size,
            vocab_size,
            vocab_per_split,
            rank,
            hidden,
            hidden.stride(0),
            hidden.stride(1),
            weight,
            weight.stride(0),
            weight.stride(1),
            labels,
            labels.stride(0),
            maximum,
            maximum.stride(0),
            acc,
            acc.stride(0),
            d_entropy_input,
            d_entropy_input.stride(0),
            dlogprobs,
            dlogprobs.stride(0),
            entropy_b_input,
            entropy_b_input.stride(0),
            d_logits,
            d_logits.stride(0),
            d_logits.stride(1),
            1.0 / temperature,
            USE_TMA=SUPPORT_CUDA_TMA and hidden.stride(1) == 1 and weight.stride(1) == 1,
            COMPUTE_ENTROPY=compute_entropy,
            INPUT_PRECISION=_dot_input_precision(hidden),
        )

        right_bound = min((split_idx + 1) * vocab_per_split, vocab_size)
        split_width = right_bound - split_idx * vocab_per_split
        if split_width == vocab_per_split:
            # Whole staging buffer is live vocabulary; slicing would copy nothing.
            split_d_logits = d_logits
        else:
            # Only the final split can be narrower than the staging buffer, so
            # nothing launches into ``d_logits`` after this. Pack the live columns
            # and release the [M, vocab_per_split] buffer before the projections,
            # otherwise it stays resident across the [M, H] temporary that
            # ``d_hidden += torch.matmul(...)`` allocates below.
            assert split_idx == num_splits - 1
            split_d_logits = d_logits[:, :split_width].contiguous()
            d_logits = None
        split_weight = weight[split_idx * vocab_per_split : right_bound]
        if split_idx == 0:
            torch.matmul(split_d_logits, split_weight, out=d_hidden)
        else:
            d_hidden += torch.matmul(split_d_logits, split_weight)
        torch.matmul(
            split_d_logits.T,
            hidden,
            out=d_weight[split_idx * vocab_per_split : right_bound],
        )

    return d_hidden, d_weight


# ======================================================================================
# SkyRL adapter. Matches FusedLinearChunkedDistributedLogprob's contract.
# ======================================================================================

# Guaranteed not to match ``local_col + rank * local_vocab``.
_OOV_LABEL_SENTINEL = torch.iinfo(torch.int64).max


def _verl_logprob_kernel_available() -> bool:
    return TRITON_AVAILABLE and is_cuda_available


class FusedLinearLogprobTriton(torch.autograd.Function):
    """Triton fused LM-head + vocab-parallel log-prob of the target token.

    Uses the same ``apply`` signature and grad contract as
    ``FusedLinearChunkedDistributedLogprob``. Targets are already shifted by the
    caller; outputs are TP-combined fp32 ``[B, S]`` log-probs.

    SkyRL re-keys shard-local targets into verl's ``rank * local_vocab`` frame.
    Targets outside this shard get a sentinel that matches no column. Fully OOV
    targets are forced to log-prob 0 in forward, but their backward path is left
    unchanged to match the stock and torch fused references.
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        if not _verl_logprob_kernel_available():
            raise RuntimeError(
                "FusedLinearLogprobTriton requires Triton (and a CUDA device), but triton is not "
                "importable. Use the Linux Megatron/FSDP dependency stack, or use the default pure-torch "
                "'torch' backend (FusedLinearChunkedDistributedLogprob) instead."
            )

        B, S, H = hidden.shape
        local_vocab = int(weight.shape[0])
        rank = 0 if tp_group is None else dist.get_rank(tp_group)

        # Targets not owned by this shard receive a no-match sentinel.
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        labels_verl = (target - vocab_start_index) + rank * local_vocab
        labels_verl = torch.where(target_mask, _OOV_LABEL_SENTINEL, labels_verl).to(torch.int64).contiguous()

        # The kernel needs 2D contiguous operands with matching dtypes; restore
        # the original hidden dtype on the returned grad.
        ctx_hidden_dtype = hidden.dtype
        hidden_2d = hidden.reshape(B * S, H).to(weight.dtype).contiguous()
        weight_c = weight.contiguous()

        # The caller folds temperature in before dispatch.
        logprobs_flat, _entropy, _maximum, _accumulate, _entropy_b = efficient_entropy_forward(
            hidden_2d,
            weight_c,
            labels_verl.reshape(-1),
            1.0,
            tp_group,
            compute_entropy=False,
        )
        assert _entropy is None and _entropy_b is None
        log_probs = logprobs_flat.reshape(B, S).to(torch.float32)

        # Fully-OOV targets have no label-logit contribution, so verl returns
        # LSE; match the stock path by forcing those log-probs to 0.
        owned_here = (~target_mask).to(torch.int32)
        owned_anywhere = owned_here.clone()
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(owned_anywhere, op=dist.ReduceOp.SUM, group=tp_group)
        fully_oov = owned_anywhere == 0  # [B, S] bool, identical on all ranks
        log_probs = log_probs.masked_fill(fully_oov, 0.0)

        if not inference_only:
            ctx.save_for_backward(
                hidden_2d,
                weight_c,
                labels_verl.reshape(-1),
                _maximum,
                _accumulate,
            )
            ctx.B, ctx.S, ctx.H = B, S, H
            ctx.tp_group = tp_group
            ctx.hidden_dtype = ctx_hidden_dtype

        return log_probs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple:
        grad_output = grad_outputs[0]  # [B, S], grad of full-vocab logprob
        hidden_2d, weight_c, labels_verl, _maximum, _accumulate = ctx.saved_tensors
        B, S, H = ctx.B, ctx.S, ctx.H
        tp_group = ctx.tp_group

        # Do not zero fully-OOV rows in backward: the reference still emits
        # -softmax * grad_output because no shard owns the one-hot column.
        # If log-probs are unused downstream, treat a missing grad as zero.
        if grad_output is None:
            dlogprobs = torch.zeros(B * S, dtype=torch.float32, device=hidden_2d.device)
        else:
            dlogprobs = grad_output.reshape(-1).to(torch.float32).contiguous()

        d_hidden_2d, d_weight = efficient_entropy_backward(
            dlogprobs,
            None,
            hidden_2d,
            weight_c,
            labels_verl,
            _maximum,
            _accumulate,
            None,
            False,  # should_return_fp32_grad
            1.0,  # temperature (pre-baked into hidden by the caller)
            tp_group,
            compute_entropy=False,
        )

        # d_hidden is this rank's partial; SP gather backward performs TP reduction.
        d_hidden = d_hidden_2d.reshape(B, S, H).to(ctx.hidden_dtype)

        # (hidden, weight, target, vstart, vend, chunk, tp_group, inference_only)
        return d_hidden, d_weight, None, None, None, None, None, None
