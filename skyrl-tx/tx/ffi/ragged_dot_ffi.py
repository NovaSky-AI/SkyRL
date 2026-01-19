from __future__ import annotations

import ctypes
import os
from pathlib import Path

import jax
from jax import lax
import jax.numpy as jnp

try:  # JAX >= 0.8
    from jax import ffi as jax_ffi
except Exception:  # pragma: no cover - older JAX fallback
    from jax.experimental import ffi as jax_ffi


_REGISTERED = False
_LOAD_ERROR: Exception | None = None


def _find_library() -> Path | None:
    env_path = os.environ.get("TX_RAGGED_DOT_FFI_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.exists() else None

    here = Path(__file__).resolve().parent
    for name in ("libragged_dot_ffi.so", "ragged_dot_ffi.so"):
        candidate = here / name
        if candidate.exists():
            return candidate
    return None


def _ensure_registered() -> bool:
    global _REGISTERED, _LOAD_ERROR
    if _REGISTERED:
        return True
    if _LOAD_ERROR is not None:
        return False

    lib_path = _find_library()
    if lib_path is None:
        _LOAD_ERROR = FileNotFoundError("ragged_dot_ffi shared library not found.")
        return False

    try:
        lib = ctypes.cdll.LoadLibrary(str(lib_path))
        jax_ffi.register_ffi_target(
            "ragged_dot_cuda",
            jax_ffi.pycapsule(lib.RaggedDotCuda),
            platform="CUDA",
        )
        jax_ffi.register_ffi_target(
            "ragged_dot_bwd_cuda",
            jax_ffi.pycapsule(lib.RaggedDotBwdCuda),
            platform="CUDA",
        )
        _REGISTERED = True
        return True
    except Exception as exc:  # pragma: no cover - load/registration failures
        _LOAD_ERROR = exc
        return False


def is_available() -> bool:
    return _ensure_registered()


def _ragged_dot_ref(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    if group_offset.shape != (1,):
        raise ValueError("group_offset must have shape (1,).")

    offset = group_offset[0]
    m = lhs.shape[0]
    g_local = rhs.shape[0]

    cumsum = jnp.cumulative_sum(group_sizes, include_initial=True)
    shard_start = cumsum[offset]
    shard_end = cumsum[offset + g_local]

    token_idx = jnp.arange(m)
    valid_mask = (token_idx >= shard_start) & (token_idx < shard_end)

    local_group_sizes = lax.dynamic_slice_in_dim(group_sizes, offset, g_local, axis=0)
    adjusted_group_sizes = local_group_sizes.at[0].add(shard_start).at[-1].add(m - shard_end)

    result = lax.ragged_dot(lhs, rhs, adjusted_group_sizes)
    return jnp.where(valid_mask[:, None], result, 0)


def _ragged_dot_ffi_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    group_offsets_cumsum: jax.Array,
) -> jax.Array:
    if not _ensure_registered():
        raise RuntimeError("ragged_dot_ffi is not available. Build and load the shared library first.")

    out = jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), lhs.dtype)
    call = jax_ffi.ffi_call("ragged_dot_cuda", out, vmap_method=None)
    return call(lhs, rhs, group_sizes, group_offset, group_offsets_cumsum)


def _ragged_dot_bwd_ffi_call(
    lhs: jax.Array,
    grad: jax.Array,
    group_sizes: jax.Array,
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
    call = jax_ffi.ffi_call("ragged_dot_bwd_cuda", out, vmap_method=None)
    return call(lhs, grad, group_sizes, group_offset, group_offsets_cumsum)


@jax.custom_vjp
def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    group_offsets_cumsum = jnp.cumsum(group_sizes, dtype=jnp.int32)
    return _ragged_dot_ffi_call(lhs, rhs, group_sizes, group_offset, group_offsets_cumsum)


def _ragged_dot_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
):
    group_offsets_cumsum = jnp.cumsum(group_sizes, dtype=jnp.int32)
    y = _ragged_dot_ffi_call(lhs, rhs, group_sizes, group_offset, group_offsets_cumsum)
    return y, (lhs, rhs, group_sizes, group_offset, group_offsets_cumsum)


def _ragged_dot_bwd(res, g):
    lhs, rhs, group_sizes, group_offset, group_offsets_cumsum = res

    # d_lhs: g @ rhs.T with ragged grouping - same structure as forward
    # g: [M, N], rhs: [G, K, N] -> rhs.T: [G, N, K], d_lhs: [M, K]
    rhs_t = jnp.swapaxes(rhs, 1, 2)  # [G, N, K]
    d_lhs = _ragged_dot_ffi_call(g, rhs_t, group_sizes, group_offset, group_offsets_cumsum)

    # d_rhs: lhs.T @ g accumulated per group -> [G, K, N]
    g_local = rhs.shape[0]
    d_rhs = _ragged_dot_bwd_ffi_call(lhs, g, group_sizes, group_offset, group_offsets_cumsum, g_local)

    return d_lhs, d_rhs, None, None


ragged_dot.defvjp(_ragged_dot_fwd, _ragged_dot_bwd)
