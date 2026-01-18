from __future__ import annotations

import ctypes
import os
from pathlib import Path

import jax
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
        _REGISTERED = True
        return True
    except Exception as exc:  # pragma: no cover - load/registration failures
        _LOAD_ERROR = exc
        return False


def is_available() -> bool:
    return _ensure_registered()


def ragged_dot(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    group_offset: jax.Array,
) -> jax.Array:
    if not _ensure_registered():
        raise RuntimeError("ragged_dot_ffi is not available. Build and load the shared library first.")

    if lhs.dtype != jnp.bfloat16 or rhs.dtype != jnp.bfloat16:
        raise NotImplementedError("ragged_dot_ffi supports bfloat16 only.")
    if group_sizes.dtype != jnp.int32 or group_offset.dtype != jnp.int32:
        raise NotImplementedError("ragged_dot_ffi expects int32 group_sizes and group_offset.")
    if group_offset.shape != (1,):
        raise ValueError("group_offset must have shape (1,).")

    out = jax.ShapeDtypeStruct((lhs.shape[0], rhs.shape[2]), lhs.dtype)
    call = jax_ffi.ffi_call("ragged_dot_cuda", out, vmap_method=None)
    return call(lhs, rhs, group_sizes, group_offset)
