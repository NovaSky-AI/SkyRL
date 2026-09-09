"""Runtime patch: scope vLLM's AOT compile artifact directory to the running GPU.

vLLM stores the artifact at ``torch_aot_compile/<hash>/rank_{rank}_{dp_rank}/model``
(``vllm/compilation/decorators.py``). The hash covers env and config factors but
carries no device, and the directory distinguishes only rank and DP rank, so two
engines that agree on both -- separate placement groups, or separate runs sharing
a cache root, each holding bundle index 0 -- resolve to one path while sitting on
different GPUs. The last save wins, and a load on any other GPU runs Inductor
wrappers holding the saver's stream handle::

    RuntimeError: CUDA driver error: invalid argument
        static_triton_launcher._launch_kernel <- profile_run -> _dummy_run

This inserts the device into that directory (``rank_0_0`` -> ``rank_0_0_dev1``).
The hash-level directory and the ``inductor_cache/`` beneath it stay shared:
Triton keys its own cache by device index, and Inductor's graph hash covers input
tensor devices, so that tree is already partitioned by device. Keying on the
device index rather than the UUID matches what the generated code bakes in, so
workers masked onto their own device 0 keep sharing one artifact.

Remove once vllm#53312 (device index in the per-rank cache path) ships.
"""

import os
from typing import Any

from loguru import logger

_PATCHED = False
_DEVICE_ATTR = "_skyrl_aot_device_path"
_WRAPPED_FLAG = "_skyrl_aot_save_wrapped"


def _device_tag() -> str | None:
    """Tag for the device this worker compiles on, or None when there is no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return f"dev{torch.cuda.current_device()}"
    except Exception as e:
        # A missing tag costs only the isolation, so it must not break startup.
        logger.debug(f"compile-cache device path: could not resolve device ({e})")
        return None


def _device_scoped_path(path: str) -> str:
    """``.../rank_0_0/model`` -> ``.../rank_0_0_dev1/model`` (unchanged on CPU)."""
    tag = _device_tag()
    if tag is None:
        return path
    parent, name = os.path.split(path)
    rank_dir = os.path.basename(parent)
    if not rank_dir.startswith("rank_"):
        logger.warning(f"compile-cache device path: unexpected AOT layout {path!r}, not scoping")
        return path
    if rank_dir.endswith(f"_{tag}"):
        return path
    return os.path.join(os.path.dirname(parent), f"{rank_dir}_{tag}", name)


def _install_save_redirect(cls: type) -> None:
    """Point ``cls.save_aot_compiled_function`` at the device-scoped path.

    ``support_torch_compile`` attaches this method per decorated class, so there
    is no module-level function to wrap. Wrapping happens on the first load
    attempt, which vLLM always makes before it compiles and saves.
    """
    # Flagging the wrapper rather than the class distinguishes a subclass that
    # inherits an already-wrapped method from one that `support_torch_compile`
    # gave its own unwrapped method.
    original_save = cls.save_aot_compiled_function
    if getattr(original_save, _WRAPPED_FLAG, False):
        return

    def save_aot_compiled_function(self: Any) -> None:
        scoped = getattr(self, _DEVICE_ATTR, None)
        if scoped is not None:
            self._aot_compilation_path = scoped
            self._aot_cache_dir = os.path.dirname(scoped)
        return original_save(self)

    setattr(save_aot_compiled_function, _WRAPPED_FLAG, True)
    cls.save_aot_compiled_function = save_aot_compiled_function


def _apply_compile_cache_device_path_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import vllm.compilation.decorators as decorators

    original_try_load = decorators._try_load_aot_compiled_fn

    def try_load_aot_compiled_fn(model: Any, path: str) -> Any:
        scoped = _device_scoped_path(path)
        # Stashed before the load because a miss leads to a compile and a save,
        # which has to land where the load looked.
        try:
            setattr(model, _DEVICE_ATTR, scoped)
            _install_save_redirect(type(model))
        except Exception as e:
            logger.debug(f"compile-cache device path: could not install save redirect ({e})")
            return original_try_load(model, path)
        return original_try_load(model, scoped)

    decorators._try_load_aot_compiled_fn = try_load_aot_compiled_fn
    _PATCHED = True
    logger.info("Patched vLLM AOT compile cache to use a per-device artifact directory")


def apply_compile_cache_device_path_patch() -> None:
    """Redirect both halves of the AOT artifact round trip, once per process.

    vLLM resolves the loader by name at call time, so replacing it covers the
    load; the loader in turn installs the save redirect on the model's class.
    """
    try:
        _apply_compile_cache_device_path_patch()
    except ModuleNotFoundError as e:
        logger.info(f"Skipping compile cache device path due to exception: {e}")
        pass
