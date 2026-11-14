"""Fused NVFP4 quantize -> dequantize (and pack) kernels on FlashInfer's grid, for sm_100 (``csrc/nvfp4_roundtrip.cu``).

One kernel reads each bf16 element once and writes its round trip (or its codes and scales) once; the quantization
is FlashInfer's own arithmetic (``fp4_quantize`` / ``nvfp4_quantize(per_token_activation=True)``, fast-math scales,
absmax or 4/6 with the MSE fast error path), so codes, scales, the 4/6 choice and the round trip are bit-identical to
FlashInfer's bytes decoded by :func:`.decode.dequantize_rows` (tested). Expert weights take a small amax pre-pass
(the per-expert global scale needs the whole expert) unless the speculative amax applies (:class:`AmaxCache`).

The kernel library is compiled with ``nvcc`` on first use (cached per source hash under ``SKYRL_QAT_NVFP4_KERNEL_DIR``,
default ``$TMPDIR/skyrl_nvfp4_kernels``) and loaded with ctypes. :func:`available` is False, and callers keep the
FlashInfer path, when ``SKYRL_QAT_NVFP4_FUSED=0``, the device is not sm_100 / sm_103, or the build fails.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_SRC = Path(__file__).parent / "csrc" / "nvfp4_roundtrip.cu"
_LOCK = threading.Lock()
_LIB: ctypes.CDLL | None = None
_LIB_FAILED = False
# FlashInfer's per-token NVFP4 decode scale is amax(row) * (1 / (448 * 6)), the fp32 of the double.
_PER_TOKEN_INV = 1.0 / (448.0 * 6.0)
# Per-token kernel configs (warps per row, 16-blocks per lane, rows per CTA), as instantiated in the .cu; a row's
# blocks stay in registers. _pt_config picks the least-padded one, ties to the earlier entry (the order is the
# measured preference on B200: K = 2048 -> (1, 4, 4), K = 6144 -> (6, 2, 1)).
_PT_CONFIGS = ((1, 1, 4), (1, 2, 4), (1, 4, 4), (6, 2, 1), (4, 3, 1), (12, 1, 1), (4, 4, 1), (8, 2, 1), (16, 1, 1),
               (1, 4, 2), (2, 2, 2), (4, 1, 1), (4, 2, 1))  # fmt: skip
_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_I32 = ctypes.c_int


def _nvcc() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    cands = [os.path.join(CUDA_HOME, "bin", "nvcc")] if CUDA_HOME else []
    cands.append(shutil.which("nvcc") or "")
    return next((c for c in cands if c and os.path.exists(c)), None)


def _build() -> ctypes.CDLL:
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found")
    src = _SRC.read_bytes()
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    # -ftz=true: FlashInfer's quantizers are built with -use_fast_math, so their fp32 arithmetic flushes subnormals.
    flags = ["-O3", "-std=c++17", "-ftz=true", "-gencode", "arch=compute_100f,code=sm_100f", "--shared", "-Xcompiler", "-fPIC"]  # fmt: skip
    key = hashlib.sha256(src + ver.encode() + " ".join(flags).encode()).hexdigest()[:16]
    root = Path(os.environ.get("SKYRL_QAT_NVFP4_KERNEL_DIR") or Path(tempfile.gettempdir()) / "skyrl_nvfp4_kernels")
    root.mkdir(parents=True, exist_ok=True)
    so = root / f"nvfp4_roundtrip_{key}.so"
    if not so.exists():
        with open(root / f".{key}.lock", "w") as lock:  # one build per node; other ranks wait for it
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not so.exists():
                tmp = root / f".{key}.{os.getpid()}.so"
                p = subprocess.run([nvcc, *flags, "-o", str(tmp), str(_SRC)], capture_output=True, text=True)
                if p.returncode != 0:
                    raise RuntimeError(f"nvcc failed:\n{p.stderr[-4000:]}")
                os.replace(tmp, so)
    return _load(so)


def _load(so: Path | str) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(so))
    lib.nvfp4rt_expert_amax.argtypes = [_P, _I64, _I64, _I32, _P, _P]
    lib.nvfp4rt_expert_rt.argtypes = [_I32, _P, _I64, _I64, _I32, _P, _I32, _P, _P, _I32, _P, _P, _P, _P, _P, _P,
                                      _I32, _P, _P]  # fmt: skip
    lib.nvfp4rt_per_token.argtypes = [_I32, _I32, _I32, _I32, _P, _I64, _I32, _I64, ctypes.c_float, _P, _I64, _P, _P,
                                      _P, _P, _P, _P]  # fmt: skip
    for f in (lib.nvfp4rt_expert_amax, lib.nvfp4rt_expert_rt, lib.nvfp4rt_per_token):
        f.restype = _I32
    return lib


def _lib() -> ctypes.CDLL | None:
    global _LIB, _LIB_FAILED
    if _LIB is not None or _LIB_FAILED:
        return _LIB
    with _LOCK:
        if _LIB is None and not _LIB_FAILED:
            try:
                _LIB = _build()
            except Exception as e:  # noqa: BLE001 - any build/load failure keeps the FlashInfer path
                _LIB_FAILED = True
                logger.warning(f"fused NVFP4 round-trip kernels unavailable, using FlashInfer: {e}")
    return _LIB


def available(device: torch.device | None = None) -> bool:
    """Whether the fused kernels run on ``device`` (sm_100 family, ``SKYRL_QAT_NVFP4_FUSED`` not ``0``, built)."""
    if os.environ.get("SKYRL_QAT_NVFP4_FUSED", "1") == "0" or not torch.cuda.is_available():
        return False
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] != 10:
        return False
    return _lib() is not None


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"fused NVFP4 {what} launch failed (CUDA error {rc})")


def _stream(dev: torch.device) -> int:
    return torch.cuda.current_stream(dev).cuda_stream


def _chunks(e: int, nb: int, dev: torch.device) -> int:
    # About 32 CTAs of 256 threads per SM across the grid; at least 2048 blocks of 16 per CTA.
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    return max(1, min(-(-sms * 32 // e), -(-nb // 2048)))


def supports(x: torch.Tensor) -> bool:
    """Whether the fused kernels can read ``x``: bf16 on a supported device, unit last stride, 32-byte aligned rows
    (256-bit loads)."""
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 16 == 0
        and x.stride(-1) == 1
        and x.data_ptr() % 32 == 0
        and all(s % 16 == 0 for s in x.stride()[:-1])
        and available(x.device)
    )


class AmaxCache:
    """Speculative per-expert amax: the round trip runs with the amax the same weights had at their last round trip
    and measures the true amax in the same pass; a check pass redoes only the experts whose amax changed (after an
    optimizer step), so results are always exact. Keyed by the stack's address and shape."""

    def __init__(self, max_entries: int = 4096) -> None:
        self._d: OrderedDict[tuple, torch.Tensor] = OrderedDict()
        self._max = max_entries

    def get(self, key: tuple) -> torch.Tensor | None:
        v = self._d.get(key)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def put(self, key: tuple, amax_bits: torch.Tensor) -> None:
        self._d[key] = amax_bits
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()


AMAX_CACHE = AmaxCache()


def expert_quantize(
    w3: torch.Tensor,
    four_over_six: bool,
    out: torch.Tensor | None = None,
    codes: torch.Tensor | None = None,
    sf_linear: torch.Tensor | None = None,
    sf_swizzled: torch.Tensor | None = None,
    speculate: bool = False,
    chunks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous bf16 ``(E, N, K)`` expert weights on FlashInfer's ``fp4_quantize`` grid (per-expert
    ``gs = 2688 / amax``) into any of: the bf16 round trip ``out (E, N, K)``, ``codes (E, N, K/2)`` uint8,
    ``sf_linear (E, N, K/16)`` / ``sf_swizzled (E, N*K/16)`` (128x4) E4M3 bytes. Returns ``(gs, alpha)`` fp32 ``(E,)``.
    ``speculate``: use and refresh :data:`AMAX_CACHE` (one pass over the weights when their amax is unchanged)."""
    lib = _lib()
    e, n, k = w3.shape
    if w3.dtype != torch.bfloat16 or not w3.is_contiguous() or k % 16 or w3.data_ptr() % 32:
        raise ValueError(f"fused NVFP4 expert quantize needs contiguous 32-byte aligned bf16 (E, N, K), got {w3.shape}")
    if sf_swizzled is not None and (n % 128 or k % 64):
        raise ValueError(f"swizzled NVFP4 scales need N % 128 == 0 and K % 64 == 0, got {(n, k)}")
    if w3.numel() // 16 >= 2**32:
        raise ValueError(f"fused NVFP4 expert quantize supports < 2^32 blocks, got {tuple(w3.shape)}")
    dev = w3.device
    kb = k // 16
    nb = n * kb
    gs = torch.empty(e, dtype=torch.float32, device=dev)
    alpha = torch.empty(e, dtype=torch.float32, device=dev)
    if e == 0 or n == 0:
        return gs, alpha
    ch = chunks or _chunks(e, nb, dev)
    stream = _stream(dev)
    outs = [_ptr(out), _ptr(codes), _ptr(sf_linear), _ptr(sf_swizzled)]
    key = (w3.data_ptr(), e, n, k, dev.index)
    spec = AMAX_CACHE.get(key) if speculate else None
    with torch.cuda.device(dev):
        if spec is None:
            partial = torch.empty((e, ch), dtype=torch.int32, device=dev)
            _check(lib.nvfp4rt_expert_amax(w3.data_ptr(), e, nb, ch, partial.data_ptr(), stream), "amax")
            _check(lib.nvfp4rt_expert_rt(int(four_over_six), w3.data_ptr(), e, nb, kb, partial.data_ptr(), ch, None,
                                         None, ch, gs.data_ptr(), alpha.data_ptr(), *outs, n, None, stream), "quantize")  # fmt: skip
            if speculate:
                bits = torch.empty(e, dtype=torch.int32, device=dev)
                # the amax as the kernel reduced it: max over the partials
                torch.amax(partial, dim=1, out=bits)
                AMAX_CACHE.put(key, bits)
        else:
            partial = torch.empty((e, ch), dtype=torch.int32, device=dev)
            _check(lib.nvfp4rt_expert_rt(int(four_over_six), w3.data_ptr(), e, nb, kb, None, 0, spec.data_ptr(),
                                         partial.data_ptr(), ch, gs.data_ptr(), alpha.data_ptr(), *outs, n, None,
                                         stream), "speculative quantize")  # fmt: skip
            # Check pass: experts whose measured amax equals the speculation exit at once; the others are redone.
            _check(lib.nvfp4rt_expert_rt(int(four_over_six), w3.data_ptr(), e, nb, kb, partial.data_ptr(), ch, None,
                                         None, ch, gs.data_ptr(), alpha.data_ptr(), *outs, n, spec.data_ptr(),
                                         stream), "check")  # fmt: skip
            torch.amax(partial, dim=1, out=spec)
    return gs, alpha


def _pt_config(kb: int) -> tuple[int, int, int] | None:
    fits = [c for c in _PT_CONFIGS if c[0] * 32 * c[1] >= kb]
    return min(fits, key=lambda c: c[0] * 32 * c[1]) if fits else None


def per_token_fits(k: int) -> bool:
    """Whether rows of ``k`` elements fit the per-token kernel (a row's blocks in registers: ``k <= 8192``)."""
    return k % 16 == 0 and _pt_config(k // 16) is not None


def per_token_quantize(
    x: torch.Tensor,
    four_over_six: bool,
    out: torch.Tensor | None = None,
    codes: torch.Tensor | None = None,
    sf_linear: torch.Tensor | None = None,
    sf_swizzled: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    dst: torch.Tensor | None = None,
    config: tuple[int, int, int] | None = None,
) -> bool:
    """Per-token NVFP4 of bf16 rows ``x [M, K]`` (unit column stride) on FlashInfer's
    ``nvfp4_quantize(per_token_activation=True)`` grid, into any of: the bf16 round trip ``out [M, K]`` (may be
    ``x``); and, at row ``dst[r]`` (``r`` without ``dst``), ``codes [Mp, K/2]``, ``sf_linear [Mp, K/16]``,
    ``sf_swizzled [Mp * K/16]`` (128x4; rows the kernel does not write are left as they are), decode scales
    ``t [Mp]``. False if ``K`` is too wide for the kernel (:func:`per_token_fits`); the caller then uses FlashInfer."""
    lib = _lib()
    m, k = x.shape
    if not supports(x):
        raise ValueError(f"fused per-token NVFP4 cannot read {x.dtype} {tuple(x.shape)} strides {x.stride()}")
    if out is not None and (out.shape != x.shape or out.dtype != torch.bfloat16 or not supports(out)):
        raise ValueError(f"out must be bf16 {tuple(x.shape)}, unit column stride, 32-byte aligned rows")
    for name, a in (("codes", codes), ("sf_linear", sf_linear), ("sf_swizzled", sf_swizzled), ("t", t)):
        if a is not None and not a.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if sf_swizzled is not None and k % 64:
        raise ValueError(f"swizzled NVFP4 scales need K % 64 == 0, got K={k}")
    kb = k // 16
    cfg = config or _pt_config(kb)
    if cfg is None:
        return False
    if m == 0:
        return True
    if dst is not None:
        dst = dst.to(torch.int64).contiguous()
    with torch.cuda.device(x.device):
        _check(lib.nvfp4rt_per_token(int(four_over_six), *cfg, x.data_ptr(), m, kb, x.stride(0), _PER_TOKEN_INV,
                                     _ptr(out), 0 if out is None else out.stride(0), _ptr(dst), _ptr(codes),
                                     _ptr(sf_linear), _ptr(sf_swizzled), _ptr(t), _stream(x.device)),
               "per-token quantize")  # fmt: skip
    return True
