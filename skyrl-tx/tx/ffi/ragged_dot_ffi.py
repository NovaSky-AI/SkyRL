from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import jax
import jax.numpy as jnp

_LIB_PATH = Path(__file__).resolve().parent / "libragged_dot_ffi.so"


@functools.lru_cache(maxsize=1)
def _ensure_registered() -> bool:
    if not _LIB_PATH.exists():
        return False
    try:
        lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))
        jax.ffi.register_ffi_target("ragged_dot_cuda", jax.ffi.pycapsule(lib.RaggedDotCuda), platform="CUDA")
        jax.ffi.register_ffi_target("ragged_dot_bwd_cuda", jax.ffi.pycapsule(lib.RaggedDotBwdCuda), platform="CUDA")
        return True
    except Exception:
        return False


def is_available() -> bool:
    return _ensure_registered()


def _ragged_dot_ffi_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_offset: jax.Array,
    group_offsets_cumsum: jax.Array,
) -> jax.Array:
    if not _ensure_registered():
        raise RuntimeError("ragged_dot_ffi is not available. Build and load the shared library first.")

    out = jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), lhs.dtype)
    call = jax.ffi.ffi_call("ragged_dot_cuda", out, vmap_method=None)
    return call(lhs, rhs, group_offset, group_offsets_cumsum)


def _ragged_dot_bwd_ffi_call(
    lhs: jax.Array,
    grad: jax.Array,
    group_offset: jax.Array,
    group_offsets_cumsum: jax.Array,
    g_local: int,
) -> jax.Array:
    """Backward pass for d_rhs: computes lhs.T @ grad per group -> [G, K, N]."""
    if not _ensure_registered():
        raise RuntimeError("ragged_dot_ffi is not available. Build and load the shared library first.")

    k = lhs.shape[1]
    n = grad.shape[1]

    out = jax.ShapeDtypeStruct((g_local, k, n), lhs.dtype)
    call = jax.ffi.ffi_call("ragged_dot_bwd_cuda", out, vmap_method=None)
    return call(lhs, grad, group_offset, group_offsets_cumsum)


def _apply_mask(result: jax.Array, cumsum: jax.Array, group_offset: jax.Array, g_local: int) -> jax.Array:
    """Zero out tokens outside the local group range [offset, offset+g_local)."""
    offset = group_offset[0]
    token_idx = jnp.arange(result.shape[0])
    valid = (token_idx >= cumsum[offset]) & (token_idx < cumsum[offset + g_local])
    return jnp.where(valid[:, None], result, 0)


@jax.custom_vjp
def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True).astype(jnp.int32)
    result = _ragged_dot_ffi_call(lhs, rhs, group_offset, cumsum)
    return _apply_mask(result, cumsum, group_offset, rhs.shape[0])


def _ragged_dot_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
):
    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True).astype(jnp.int32)
    result = _ragged_dot_ffi_call(lhs, rhs, group_offset, cumsum)
    y = _apply_mask(result, cumsum, group_offset, rhs.shape[0])
    return y, (lhs, rhs, group_offset, cumsum)


def _ragged_dot_bwd(res, g):
    lhs, rhs, group_offset, cumsum = res
    g_local = rhs.shape[0]

    # d_lhs: g @ rhs.T with ragged grouping
    rhs_t = jnp.swapaxes(rhs, 1, 2)
    d_lhs = _apply_mask(_ragged_dot_ffi_call(g, rhs_t, group_offset, cumsum), cumsum, group_offset, g_local)

    # d_rhs: lhs.T @ g accumulated per group
    d_rhs = _ragged_dot_bwd_ffi_call(lhs, g, group_offset, cumsum, g_local)

    return d_lhs, d_rhs, None, None


ragged_dot.defvjp(_ragged_dot_fwd, _ragged_dot_bwd)
