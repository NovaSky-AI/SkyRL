"""Load cuDNN's sublibraries from the ``nvidia-cudnn`` wheel when an older cuDNN is on ``LD_LIBRARY_PATH``.

cuDNN loads its engine sublibraries (``libcudnn_engines_runtime_compiled.so.9`` and the rest) with ``dlopen``
on first use. Those lookups search ``LD_LIBRARY_PATH`` before the wheel's own directory (the sublibraries'
``DT_RUNPATH=$ORIGIN`` comes after it). An image that puts an older cuDNN on ``LD_LIBRARY_PATH`` (e.g.
``/opt/cudnn/lib`` in the SkyRL Megatron image) then mixes versions: torch 2.14 loads cuDNN 9.24 from the
wheel, the sublibraries resolve to the image's copy, and the first cuDNN op fails with ``Unable to load any
of {libcudnn_engines_runtime_compiled.so.9.24.0, ...}`` and ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED``
(e.g. the ``Conv3d`` patch embedding of Qwen-VL vision towers).

When ``LD_LIBRARY_PATH`` holds such a cuDNN, this loads every ``libcudnn*.so.9`` from the wheel up front
with ``RTLD_GLOBAL``, so cuDNN's later ``dlopen`` calls resolve by soname to the already-loaded copies.
``skyrl.backends.skyrl_train`` calls it on package import, before any cuDNN op in trainer workers and vLLM's
worker processes.

TEMPORARY: delete this module (and its call in ``skyrl.backends.skyrl_train``) once the Megatron images
are rebuilt without the system cuDNN (NovaSky-AI/SkyRL#2636).
"""

import ctypes
import glob
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

# libcudnn first, then the graph library the others depend on, then the engines.
_LOAD_ORDER = (
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
)


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _wheel_cudnn_lib_dir() -> str | None:
    """The ``lib`` directory of the installed ``nvidia-cudnn`` wheel, if any."""
    spec = importlib.util.find_spec("nvidia.cudnn") if importlib.util.find_spec("nvidia") else None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        lib_dir = os.path.join(location, "lib")
        if os.path.isfile(os.path.join(lib_dir, "libcudnn.so.9")):
            return lib_dir
    return None


def _wheel_libraries(lib_dir: str) -> list[str]:
    """The wheel's ``libcudnn*.so.9`` in load order."""
    present = {os.path.basename(p): p for p in glob.glob(os.path.join(lib_dir, "libcudnn*.so.9"))}
    ordered = [present.pop(name) for name in _LOAD_ORDER if name in present]
    return ordered + sorted(present.values())


def _real_dirs(paths: list[str]) -> set[str]:
    """Directories the given files resolve to (wheel files may be symlinks into a package cache)."""
    return {os.path.dirname(os.path.realpath(p)) for p in paths}


def _foreign_cudnn_on_ld_library_path(wheel_dirs: set[str]) -> list[str]:
    """``LD_LIBRARY_PATH`` entries holding a ``libcudnn*.so.9`` that is not the wheel's."""
    foreign = []
    for entry in filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
        libs = glob.glob(os.path.join(entry, "libcudnn*.so.9"))
        if libs and not _real_dirs(libs) <= wheel_dirs:
            foreign.append(entry)
    return foreign


def _loaded_cudnn_dirs(maps_path: str = "/proc/self/maps") -> set[str]:
    """Directories of the ``libcudnn*`` objects already mapped into this process."""
    try:
        with open(maps_path) as maps:
            fields = (line.rstrip("\n").split(None, 5) for line in maps if "libcudnn" in line)
            paths = {f[5].removesuffix(" (deleted)") for f in fields if len(f) == 6}
    except OSError:
        return set()
    return {os.path.dirname(p) for p in paths if p.startswith("/")}


def preload_cudnn() -> list[str]:
    """Load the wheel's ``libcudnn*.so.9`` with ``RTLD_GLOBAL`` if needed. Returns the paths loaded.

    A no-op off Linux, without the ``nvidia-cudnn`` wheel, when no other cuDNN is on ``LD_LIBRARY_PATH``,
    and when cuDNN is already loaded from outside the wheel (the process uses a different cuDNN, and
    loading the wheel's next to it would mix versions).
    """
    if not _is_linux():
        return []
    lib_dir = _wheel_cudnn_lib_dir()
    if lib_dir is None:
        return []
    libraries = _wheel_libraries(lib_dir)
    wheel_dirs = _real_dirs(libraries) | {os.path.realpath(lib_dir)}
    foreign = _foreign_cudnn_on_ld_library_path(wheel_dirs)
    if not foreign:
        return []
    other_loaded = _loaded_cudnn_dirs() - wheel_dirs
    if other_loaded:
        logger.warning(f"cuDNN is already loaded from {sorted(other_loaded)}; not preloading {lib_dir}")
        return []
    loaded = []
    for path in libraries:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:  # a missing driver or a broken install must not break package import
            logger.warning(f"Could not preload {path}: {e}")
            continue
        loaded.append(path)
    logger.info(
        f"Preloaded {len(loaded)} cuDNN libraries from {lib_dir} (another cuDNN is on LD_LIBRARY_PATH: {foreign})"
    )
    return loaded
