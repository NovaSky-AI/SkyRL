# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from functools import partial
from typing import NamedTuple, Optional

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

CompilerParams = getattr(plgpu, "CompilerParams", getattr(plgpu, "TritonCompilerParams", None))
if CompilerParams is None:
    raise RuntimeError(
        "Could not find CompilerParams in `jax.experimental.pallas.triton`. Upgrade to an newer JAX version please."
    )

# kernel ###############################################################################################################

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64

cdiv = lambda a, b: pl.cdiv(a, jnp.array(b, a.dtype) if isinstance(a, jax.Array) else b)


class ProblemSizes(NamedTuple):
    m: int
    k: int
    n: int
    g: int


class BlockSizes(NamedTuple):
    m: int
    k: int
    n: int


def _gpu_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    A_ref,  # [k, n] or [n, k]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    # outputs
    y_ref,  # [k, n]
    # static problem shapes
    trans_rhs: bool,
    size: ProblemSizes,
    block: BlockSizes,  # hyperparameters
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
    pid = namedtuple("pid", ["gi", "j"])(pl.program_id(0), pl.program_id(1))
    group_sz = group_sizes_ref[pid.gi]
    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    dim_nums = (((1,), (0,)), ((), ())) if not trans_rhs else (((1,), (1,)), ((), ()))
    _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

    @pl.when(group_sz > 0)
    def _():
        # row index into lhs and output
        start_ridx = jnp.where(pid.gi == 0, 0, group_offset_ref[jnp.maximum(pid.gi - 1, 0)])

        def outer_compute(r_offset, _):
            ridx = start_ridx + r_offset * block.m  # r_offset is 0,1,2,... need to map it to actual row indices
            lhs_rows_mask = (r_offset * block.m + jnp.arange(block.m)) < group_sz
            lhs_rows_idx = pl.ds(ridx, block.m)
            rhs_cols_idx = pl.ds(0, block.n)
            rhs_cols_mask = (block.n * pid.j + jnp.arange(block.n)) < size.n

            def inner_compute(k, acc):
                inner_idx = pl.ds(k * block.k, block.k)
                inner_mask = (k * block.k + jnp.arange(block.k)) < size.k
                _load = partial(plgpu.load, other=0)
                x = _load(x_ref.at[lhs_rows_idx, inner_idx], mask=lhs_rows_mask[:, None] & inner_mask[None, :])
                if not trans_rhs:
                    A = _load(A_ref.at[inner_idx, rhs_cols_idx], mask=inner_mask[:, None] & rhs_cols_mask[None, :])
                else:
                    A = _load(A_ref.at[rhs_cols_idx, inner_idx], mask=rhs_cols_mask[:, None] & inner_mask[None, :])
                return acc + _dot_fn(x.astype(compute_dtype), A.astype(compute_dtype)).astype(acc.dtype)

            acc = jnp.zeros((block.m, block.n), dtype=acc_dtype)
            acc = jax.lax.fori_loop(0, cdiv(size.k, block.k), inner_compute, acc)
            acc = acc.astype(y_ref.dtype)
            plgpu.store(y_ref.at[lhs_rows_idx, rhs_cols_idx], acc, mask=lhs_rows_mask[:, None] & rhs_cols_mask[None, :])
            return None

        jax.lax.fori_loop(0, cdiv(group_sz, block.m), outer_compute, None)

    last_offset = group_offset_ref[size.g - 1]

    @pl.when((pid.gi == size.g - 1) & (last_offset < size.m))
    def _():
        col_mask = (block.n * pid.j + jnp.arange(block.n)) < size.n

        def set_zero(i, _):
            row_mask = (last_offset + i * block.m + jnp.arange(block.m)) < size.m
            idx = (pl.ds(last_offset + i * block.m, block.m), pl.ds(0, block.n))
            mask = row_mask[:, None] & col_mask[None, :]
            plgpu.store(y_ref.at[*idx], jnp.zeros((block.m, block.n), dtype=y_ref.dtype), mask=mask)

        jax.lax.fori_loop(0, cdiv(size.m - last_offset, block.m), set_zero, None)


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(3, 12)))
def _gpu_ragged_dot_fwd(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    trans_rhs: bool = False,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Compute grouped matmul on GPU via a Pallas lowering."""
    assert A.ndim == 3 and x.ndim == 2, f"but {A.ndim=} {x.ndim=}"
    assert A.shape[:1] == group_sizes.shape
    n = A.shape[-1] if not trans_rhs else A.shape[-2]
    size = ProblemSizes(m=x.shape[0], k=x.shape[1], n=n, g=A.shape[0])

    # normalize the block sizes for GPU
    block_m, block_k, block_n = [
        pl.next_power_of_2(min(b, s)) for b, s in zip([block_m, block_k, block_n], [size.m, size.k, size.n])
    ]
    block_k, block_n = max(block_k, 16), max(block_n, 16)
    group_offsets = jnp.cumsum(group_sizes, -1)

    A_spec = pl.BlockSpec((None, size.k, block_n), lambda i, j: (i, 0, j))
    if trans_rhs:  # transposed spec
        A_spec = pl.BlockSpec((None, block_n, size.k), lambda i, j: (i, j, 0))
    in_specs = [
        pl.BlockSpec((size.m, size.k), lambda i, j: (0, 0)),
        A_spec,
        pl.BlockSpec((group_sizes.size,), lambda i, j: (0,)),
        pl.BlockSpec((group_offsets.size,), lambda i, j: (0,)),
    ]
    out_shape = jax.ShapeDtypeStruct((size.m, size.n), dtype=x.dtype)
    out_specs = pl.BlockSpec((size.m, block_n), lambda i, j: (0, j))
    grid = (size.g, pl.cdiv(size.n, block_n))
    block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
    dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
    with jax.named_scope("gpu_ragged_dot"):
        y = pl.pallas_call(
            partial(_gpu_ragged_dot_kernel, size=size, block=block_sizes, **dtype_spec, trans_rhs=trans_rhs),
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
            name="gpu_ragged_dot",
        )(x, A, group_sizes, group_offsets)
    res = (x, A, group_sizes)
    return y, res


def _gpu_trans_ragged_dot_kernel(
    # inputs
    x_ref,  # [m, k]
    y_ref,  # [k, n]
    group_sizes_ref,  # [g]
    group_offset_ref,  # [g]
    # outputs
    A_bar_ref,  # [g, k, n]
    # static problem shapes
    size: ProblemSizes,
    block: BlockSizes,  # hyperparameters
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: "jnp.dtype" = jnp.float32,
):
    assert A_bar_ref.shape == (block.k, block.n)
    pid = namedtuple("pid", ["gi", "r", "c"])(pl.program_id(0), pl.program_id(1), pl.program_id(2))
    group_sz = group_sizes_ref[pid.gi]
    compute_dtype = compute_dtype if compute_dtype is not None else x_ref.dtype

    dim_nums = (((0,), (0,)), ((), ()))
    _dot_fn = partial(jax.lax.dot_general, dimension_numbers=dim_nums, preferred_element_type=acc_dtype)

    @pl.when(group_sz > 0)
    def _():
        # row index into lhs and output
        start_ridx = jnp.where(pid.gi == 0, 0, group_offset_ref[jnp.maximum(pid.gi - 1, 0)])

        k_idx = pl.ds(0, block.k)
        k_mask = (pid.r * block.m + jnp.arange(block.k)) < size.k
        cols_idx = pl.ds(0, block.n)
        cols_mask = (pid.c * block.n + jnp.arange(block.n)) < size.n

        def inner_compute(r_offset, acc):
            ridx = start_ridx + r_offset * block.m  # r_offset is 0,1,2,... need to map it to actual row indices
            xy_rows_mask = (r_offset * block.m + jnp.arange(block.m)) < group_sz
            xy_rows_idx = pl.ds(ridx, block.m)

            x = plgpu.load(x_ref.at[xy_rows_idx, k_idx], mask=xy_rows_mask[:, None] & k_mask[None, :], other=0)
            y = plgpu.load(y_ref.at[xy_rows_idx, cols_idx], mask=xy_rows_mask[:, None] & cols_mask[None, :], other=0)
            return acc + _dot_fn(x.astype(compute_dtype), y.astype(compute_dtype)).astype(acc.dtype)

        acc = jnp.zeros((block.k, block.n), dtype=acc_dtype)
        acc = jax.lax.fori_loop(0, cdiv(group_sz, block.m), inner_compute, acc)
        acc = acc.astype(y_ref.dtype)
        plgpu.store(A_bar_ref.at[k_idx, cols_idx], acc, mask=k_mask[:, None] & cols_mask[None, :])

    @pl.when(group_sz == 0)
    def _():
        rmask = (pid.r * block.k + jnp.arange(block.k)) < size.k
        cmask = (pid.c * block.n + jnp.arange(block.n)) < size.n
        plgpu.store(A_bar_ref, jnp.zeros_like(A_bar_ref), mask=rmask[:, None] & cmask[None, :])


# main routine #########################################################################################################


@partial(jax.jit, static_argnums=list(range(3, 11)))
def _gpu_trans_ragged_dot_fwd(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # shape[0] of A_i tile (block_m, block_n)
    block_n: int = DEFAULT_BLOCK_N,  # shape[1] of A_i tile (block_m, block_n)
    block_k: int = DEFAULT_BLOCK_K,  # how many rows in the accumulation loop over block_m
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
    """Compute grouped matmul on GPU via a Pallas lowering."""
    assert y.ndim == 2 and x.ndim == 2 and x.shape[0] == y.shape[0]
    (m, k), n = x.shape, y.shape[-1]
    size = ProblemSizes(m=m, k=k, n=n, g=group_sizes.size)

    block_m, block_n = min(block_m, m), min(block_n, n)

    # normalize the block sizes for GPU
    block_m, block_k, block_n = [
        max(pl.next_power_of_2(min(b, s)), 16)
        for b, s in zip([block_m, block_k, block_n, block_k], [size.m, size.k, size.n])
    ]

    group_offsets = jnp.cumsum(group_sizes, -1)  # we'll read 1 down always
    in_specs = [
        pl.BlockSpec((size.m, block_k), lambda i, r, c: (0, r)),
        pl.BlockSpec((size.m, block_n), lambda i, r, c: (0, c)),
        pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
        pl.BlockSpec((size.g,), lambda i, r, c: (0,)),
    ]

    out_shape = jax.ShapeDtypeStruct((size.g, size.k, size.n), dtype=x.dtype)
    out_specs = pl.BlockSpec((None, block_k, block_n), lambda i, r, c: (i, r, c))
    grid = (size.g, pl.cdiv(size.k, block_k), pl.cdiv(size.n, block_n))

    block_sizes = BlockSizes(m=block_m, k=block_k, n=block_n)
    dtype_spec = dict(compute_dtype=compute_dtype, acc_dtype=acc_dtype)
    with jax.named_scope("gpu_trans_ragged_dot"):
        y = pl.pallas_call(
            partial(_gpu_trans_ragged_dot_kernel, size=size, block=block_sizes, **dtype_spec),
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            interpret=interpret,
            compiler_params=CompilerParams(num_warps=num_warps, num_stages=num_stages),
            name="gpu_trans_ragged_dot",
        )(x, y, group_sizes, group_offsets)
    res = (x, y, group_sizes)
    return y, res


# exported methods #####################################################################################################


NONDIFF_ARGNAMES = (
    "block_m",
    "block_k",
    "block_n",
    "interpret",
    "compute_dtype",
    "acc_dtype",
    "num_warps",
    "num_stages",
)


@partial(jax.custom_vjp, nondiff_argnames=NONDIFF_ARGNAMES + ("trans_rhs",))
@partial(jax.jit, static_argnums=list(range(3, 12)))
def ragged_dot(
    x: jax.Array,  # [m, k]
    A: jax.Array,  # [g, k, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,  # unused, but used by the backwards pass
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    trans_rhs: bool = False,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Ragged dot corresponding to jax.lax.ragged_dot (m, k) x (g, k, n) -> (m, n)"""
    kw = dict(block_m=block_m, block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    return _gpu_ragged_dot_fwd(x, A, group_sizes, **kw, trans_rhs=trans_rhs)[0]



def _gpu_ragged_dot_bwd(
    block_m: int,
    block_k: int,
    block_n: int,
    trans_rhs: bool,
    interpret: bool,
    compute_dtype: Optional["jnp.dtype"],
    acc_dtype: Optional["jnp.dtype"],
    num_warps: int | None,
    num_stages: int | None,
    res: tuple[jax.Array, jax.Array, jax.Array],
    do: jax.Array,
):
    kw = dict(block_m=block_m, block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    x, A, group_sizes = res
    dx = ragged_dot(do, A, group_sizes, **kw, trans_rhs=not trans_rhs)
    dA = trans_ragged_dot(x, do, group_sizes, **kw) if not trans_rhs else trans_ragged_dot(do, x, group_sizes, **kw)
    return dx, dA, None


ragged_dot.defvjp(_gpu_ragged_dot_fwd, _gpu_ragged_dot_bwd)


@partial(jax.custom_vjp, nondiff_argnames=NONDIFF_ARGNAMES)
@partial(jax.jit, static_argnums=list(range(3, 11)))
def trans_ragged_dot(
    x: jax.Array,  # [m, k]
    y: jax.Array,  # [m, n]
    group_sizes: jax.Array,  # [g]
    block_m: int = DEFAULT_BLOCK_M,
    block_k: int = DEFAULT_BLOCK_K,
    block_n: int = DEFAULT_BLOCK_N,
    interpret: bool = False,
    compute_dtype: Optional["jnp.dtype"] = None,
    acc_dtype: Optional["jnp.dtype"] = jnp.float32,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> jax.Array:
    """Tranposed ragged dot corresponding to transpose of ragged dot wrt A argument (m, k) x (m, n) -> (g, k, n)"""
    kw = dict(block_m=block_m, block_n=block_n, block_k=block_k, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    return _gpu_trans_ragged_dot_fwd(x, y, group_sizes, **kw)[0]


def _gpu_trans_ragged_dot_bwd(
    group_sizes: jax.Array,  # [g]
    block_m: int,
    block_k: int,
    block_n: int,
    interpret: bool,
    compute_dtype: Optional["jnp.dtype"],
    acc_dtype: Optional["jnp.dtype"],
    num_warps: int | None,
    num_stages: int | None,
    res: tuple[jax.Array, jax.Array, jax.Array],
    do: jax.Array,
):
    kw = dict(block_m=block_m, block_k=block_k, block_n=block_n, interpret=interpret)
    kw = dict(kw, compute_dtype=compute_dtype, acc_dtype=acc_dtype, num_warps=num_warps, num_stages=num_stages)
    x, y, group_sizes = res
    dx = ragged_dot(y, do, group_sizes, **kw, trans_rhs=True)
    dy = trans_ragged_dot(x, do, group_sizes, **kw)
    return dx, dy, None


trans_ragged_dot.defvjp(_gpu_trans_ragged_dot_fwd, _gpu_trans_ragged_dot_bwd)
