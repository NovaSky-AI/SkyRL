"""Point Transformer Engine's NVRTC at the pip CUDA it compiles headers from.

TE JIT-compiles some kernels with NVRTC at runtime (e.g. the MXFP8 RMSNorm
forward). It loads ``libnvrtc`` from ``NVRTC_HOME`` / ``CUDA_HOME`` /
``/usr/local/cuda`` *before* the pip ``nvidia-cuda-nvrtc`` wheel, but 2.19 also
points ``NVTE_CUDA_INCLUDE_DIR`` at the pip ``nvidia/cu{major}`` headers.
On a box whose system toolkit is a different major (``/usr/local/cuda`` ->
12.9 next to our cu13 env), NVRTC 12.9 compiles against CUDA 13 headers and
fails: ``NVRTC_ERROR_COMPILATION`` in ``rmsnorm_fwd_kernel.cu`` (syntax errors
in ``cuda_fp8.hpp``). TE 2.16 left the include dir unset, so the system NVRTC
and system headers still matched there.

TE reads ``NVRTC_HOME`` once, at ``import transformer_engine``, so this must
run before the first TE import in the process. ``skyrl.backends.skyrl_train``
calls it on package import, which precedes every SkyRL module that imports TE
or Megatron.

Delete this when TE prefers its pip NVRTC, or pairs the include dir with the
NVRTC it loaded (check ``_load_cuda_library`` in
``transformer_engine/common/__init__.py``).
"""

import glob
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

_TE_CORE_CUDA_MAJORS = (13, 12)


def _has_nvrtc(path: str | os.PathLike) -> bool:
    # Same pattern TE's loader globs for under NVRTC_HOME.
    return bool(glob.glob(f"{path}/**/libnvrtc.so*", recursive=True))


def _te_core_cuda_major() -> int | None:
    """CUDA major of the installed TE core wheel (``transformer-engine-cu{major}``)."""
    for major in _TE_CORE_CUDA_MAJORS:
        try:
            importlib.metadata.distribution(f"transformer-engine-cu{major}")
        except importlib.metadata.PackageNotFoundError:
            continue
        return major
    return None


def _pip_cuda_dir(major: int) -> Path | None:
    """The pip ``nvidia/cu{major}`` directory, if it ships ``libnvrtc``."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / f"cu{major}"
        if (candidate / "lib").is_dir() and _has_nvrtc(candidate / "lib"):
            return candidate
    return None


def pin_te_nvrtc_to_pip_cuda() -> str | None:
    """Set ``NVRTC_HOME`` to the pip CUDA matching TE's headers. Returns the path set, or None.

    Leaves the environment alone when TE is already imported (too late to
    matter), when ``NVTE_CUDA_INCLUDE_DIR`` is set (the user chose the headers,
    so the NVRTC pairing is theirs too), or when ``NVRTC_HOME`` already holds a
    ``libnvrtc``. A ``NVRTC_HOME`` without one is replaced: TE would skip it and
    fall back to ``/usr/local/cuda``, which is the mismatch this exists to avoid.
    """
    if "transformer_engine" in sys.modules or os.environ.get("NVTE_CUDA_INCLUDE_DIR"):
        return None
    current = os.environ.get("NVRTC_HOME")
    if current and _has_nvrtc(current):
        return None
    major = _te_core_cuda_major()
    if major is None:
        return None
    cuda_dir = _pip_cuda_dir(major)
    if cuda_dir is None:
        return None
    os.environ["NVRTC_HOME"] = str(cuda_dir)
    return str(cuda_dir)
