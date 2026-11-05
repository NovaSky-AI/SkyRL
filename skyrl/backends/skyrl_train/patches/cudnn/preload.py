"""Load cuDNN's sublibraries from the ``nvidia-cudnn`` wheel that torch was built against.

``libcudnn.so.9`` loads its engine sublibraries (``libcudnn_engines_runtime_compiled.so.9`` and the rest)
with ``dlopen`` on first use, and ``dlopen`` searches ``LD_LIBRARY_PATH`` before the library's own
directory. Images that put an older cuDNN on ``LD_LIBRARY_PATH`` (e.g. ``/opt/cudnn/lib`` with 9.12 in the
SkyRL CI image) then mix versions: torch 2.14 loads cuDNN 9.24 from the wheel, the sublibraries resolve
to the image's copy, and the first cuDNN op fails with ``Unable to load any of
{libcudnn_engines_runtime_compiled.so.9.24.0, ...}`` and ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED``
(e.g. the ``Conv3d`` patch embedding of Qwen-VL vision towers).

Loading every ``libcudnn*.so.9`` from the wheel's directory up front, with ``RTLD_GLOBAL``, makes cuDNN's
later ``dlopen`` calls resolve by soname to the already-loaded copies. ``skyrl.backends.skyrl_train``
calls this on package import, before any cuDNN op in trainer workers and vLLM's worker processes.

Delete this when cuDNN resolves its sublibraries relative to ``libcudnn.so.9`` ahead of
``LD_LIBRARY_PATH``, or the images stop shipping a system cuDNN on ``LD_LIBRARY_PATH``.
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


def _loaded_cudnn_dirs() -> set[str]:
    """Directories of the ``libcudnn*`` objects already mapped into this process."""
    try:
        with open("/proc/self/maps") as maps:
            paths = {line.split()[-1] for line in maps if "libcudnn" in line}
    except OSError:
        return set()
    return {os.path.dirname(p) for p in paths if p.startswith("/")}


def _ordered_libraries(lib_dir: str) -> list[str]:
    present = {os.path.basename(p): p for p in glob.glob(os.path.join(lib_dir, "libcudnn*.so.9"))}
    ordered = [present.pop(name) for name in _LOAD_ORDER if name in present]
    return ordered + sorted(present.values())


def preload_cudnn() -> list[str]:
    """Load the wheel's ``libcudnn*.so.9`` with ``RTLD_GLOBAL``. Returns the paths loaded.

    A no-op off Linux, without the ``nvidia-cudnn`` wheel, and when cuDNN is already loaded from another
    directory (the process uses a different cuDNN, and loading the wheel's next to it would mix versions).
    """
    if not sys.platform.startswith("linux"):
        return []
    lib_dir = _wheel_cudnn_lib_dir()
    if lib_dir is None:
        return []
    other_dirs = _loaded_cudnn_dirs() - {os.path.realpath(lib_dir), lib_dir}
    if other_dirs:
        logger.debug(f"cuDNN already loaded from {sorted(other_dirs)}; not preloading {lib_dir}")
        return []
    loaded = []
    for path in _ordered_libraries(lib_dir):
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:  # a missing driver or a broken install must not break package import
            logger.debug(f"Could not preload {path}: {e}")
            break
        loaded.append(path)
    return loaded
