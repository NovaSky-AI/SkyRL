from __future__ import annotations

import ctypes
import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=1)
def _ensure_registered() -> bool:
    if env_path := os.environ.get("TX_RAGGED_DOT_FFI_PATH"):
        lib_path = Path(env_path)
    else:
        here = Path(__file__).resolve().parent
        lib_path = next((p for p in [here / "libragged_dot_ffi.so", here / "ragged_dot_ffi.so"] if p.exists()), None)

    if not lib_path or not lib_path.exists():
        return False

    try:
        lib = ctypes.cdll.LoadLibrary(str(lib_path))
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


@jax.custom_vjp
def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    group_offsets_cumsum = jnp.cumsum(group_sizes, dtype=jnp.int32)
    return _ragged_dot_ffi_call(lhs, rhs, group_offset, group_offsets_cumsum)


def _ragged_dot_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
):
    group_offsets_cumsum = jnp.cumsum(group_sizes, dtype=jnp.int32)
    y = _ragged_dot_ffi_call(lhs, rhs, group_offset, group_offsets_cumsum)
    return y, (lhs, rhs, group_offset, group_offsets_cumsum)


def _ragged_dot_bwd(res, g):
    lhs, rhs, group_offset, group_offsets_cumsum = res

    # d_lhs: g @ rhs.T with ragged grouping - same structure as forward
    # g: [M, N], rhs: [G, K, N] -> rhs.T: [G, N, K], d_lhs: [M, K]
    rhs_t = jnp.swapaxes(rhs, 1, 2)  # [G, N, K]
    d_lhs = _ragged_dot_ffi_call(g, rhs_t, group_offset, group_offsets_cumsum)

    # d_rhs: lhs.T @ g accumulated per group -> [G, K, N]
    g_local = rhs.shape[0]
    d_rhs = _ragged_dot_bwd_ffi_call(lhs, g, group_offset, group_offsets_cumsum, g_local)

    return d_lhs, d_rhs, None, None


ragged_dot.defvjp(_ragged_dot_fwd, _ragged_dot_bwd)
