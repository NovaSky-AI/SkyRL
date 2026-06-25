"""Shared vLLM layerwise-reload lifecycle for SkyRL's vLLM worker-extension classes.

Provides `LayerwiseReloadWorkerMixin`, the start/finish bracket that both
`vllm_worker.WorkerWrap` and
`new_inference_worker_wrap.NewInferenceWorkerWrap` use to run vLLM's
layerwise reload once per weight sync rather than once per chunk.
"""

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def get_numel_loaded(weight_loader: Callable, args: inspect.BoundArguments) -> tuple[int, object]:
    """
    Determine how many elements would be loaded by a weight loader call.

    Args:
        weight_loader: used to load weights
        args: bound arguments to weight loader

    Returns:
        number of elements loaded by the weight loader, the return value of the
        weight loader
    """
    # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
    from vllm.model_executor.model_loader.reload.meta import CopyCounter

    with CopyCounter() as counter:
        return_value = weight_loader(*args.args, **args.kwargs)

    # A weight loader fills a single destination parameter, so the number of
    # loaded elements is at most that parameter's size. Some loaders copy into
    # the parameter more than once -- e.g. ``composed_weight_loader`` runs an
    # in-place post-load transform (``param.copy_(fn(param))``) on top of the
    # initial copy -- which would make CopyCounter report twice the parameter
    # size. Over-counting inflates the layer's loaded-element total and can
    # finalize the layer before every parameter is loaded, silently dropping
    # the trailing parameter(s) (e.g. Mamba ``mixer.D``). Cap the count at the
    # destination size to keep the per-layer accounting correct.
    numel = counter.copied_numel
    param = args.arguments.get("param", None)
    if isinstance(param, torch.Tensor):
        numel = min(numel, param.numel())
    return numel, return_value


def patch_numel_loaded():
    # vLLM's layerwise reload binds get_numel_loaded at import time
    # (`from .meta import get_numel_loaded`), so its call site at
    # layerwise.py uses the `layerwise` module's own binding. Rebind that
    # attribute to our patched version to substitute the symbol.
    from vllm.model_executor.model_loader.reload import layerwise as _layerwise
    from vllm.model_executor.model_loader.reload import meta as _meta

    _layerwise.get_numel_loaded = get_numel_loaded
    _meta.get_numel_loaded = get_numel_loaded


# Workaround for a vLLM layerwise-reload corruption affecting NemotronH/Mamba.
# MambaMixer2 registers `conv_weights` as a non-persistent buffer that is a
# view of `self.conv1d.weight.data` (shared storage). vLLM's reload code path
# (model_executor/model_loader/reload/layerwise.py) materializes the buffer
# into a fresh uninitialized GPU tensor and then runs
# `kernel_conv_weights.data.copy_(fresh)` in `_copy_and_restore_kernel_tensors`.
# Because the kernel buffer shares storage with `conv1d.weight.data`, this
# writes garbage (NaN-bit-pattern bytes in bf16) into the conv1d weight,
# corrupting all 23 Mamba layers after every weight sync.
#
# Adding "conv_weights" to vLLM's SKIP_TENSORS makes capture/restore/materialize
# skip the buffer entirely, so the view stays intact and conv1d.weight is
# preserved. Must be applied before `record_metadata_for_reloading` runs at
# model construction; this module is imported by vLLM via
# --worker-extension-cls before model init, so the import-time patch is
# correctly ordered.
# Remove this pending https://github.com/vllm-project/vllm/pull/42481 which should
# be included in vLLM 0.21.0
try:
    # Guarded import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
    from vllm.model_executor.model_loader.reload.meta import (
        SKIP_TENSORS as _VLLM_SKIP_TENSORS,
    )

    _VLLM_SKIP_TENSORS.add("conv_weights")
    # use patched version
    patch_numel_loaded()
except ImportError:
    pass


class LayerwiseReloadWorkerMixin:
    """Bracket a multi-chunk weight sync with one vLLM layerwise-reload init/finalize.

    `skyrl_start_weight_update` initializes the layerwise reload once; each chunk then loads
    its weights raw; `skyrl_finish_weight_update` finalizes once over the whole weight set.
    A per-chunk `reload_weights` is the wrong approach: it re-finalizes on every call
    and restores layers absent from that chunk, corrupting a multi-chunk sync.
    """

    vllm_config: "VllmConfig"
    model_runner: "GPUModelRunner"
    model_config: "ModelConfig"
    device: torch.device

    # NOTE: named with a `skyrl_` prefix to avoid colliding with vLLM's own
    # Worker.start_weight_update / finish_weight_update (added in vllm-project/vllm
    # #39212, merge e3b65a5, shipped in vLLM 0.22.0+). vLLM injects the
    # worker-extension class as a *base* of Worker and asserts the extension
    # defines no attribute already present on Worker, so same-named methods abort
    # engine init. The skyrl_-prefixed variants keep SkyRL's IPC weight-sync path
    # (and the MoE set_current_vllm_config wrapping) intact alongside vLLM's native API.
    def skyrl_start_weight_update(self, is_checkpoint_format: bool = True) -> None:
        """
        Prepare the model for a new weight update.

        For checkpoint-format weights, initializes the layerwise reload
        machinery which moves layers to meta device and wraps weight loaders
        to defer processing until all weights for each layer are loaded.

        Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if incoming weights are in checkpoint
                format (need layerwise processing). False if weights are
                already in kernel format (direct copy).
        """
        if getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError(
                "skyrl_start_weight_update called while a weight update is "
                "already active. Call skyrl_finish_weight_update first."
            )

        if is_checkpoint_format:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                initialize_layerwise_reload(model)

        self._skyrl_is_checkpoint_format = is_checkpoint_format
        self._skyrl_weight_update_active = True

    def skyrl_finish_weight_update(self) -> None:
        """
        Finalize the current weight update.

        For checkpoint-format weights, runs layerwise postprocessing
        (quantization repacking, attention weight processing, etc.).
        Must be called after all update_weights_ipc calls are done.
        """
        if not getattr(self, "_skyrl_weight_update_active", False):
            raise RuntimeError("skyrl_start_weight_update must be called before skyrl_finish_weight_update.")

        if self._skyrl_is_checkpoint_format:
            # Lazy import: vllm is a Linux-only optional dependency, so this module stays importable on macOS / CI.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
            )

            model = self.model_runner.model
            with set_current_vllm_config(self.vllm_config), torch.device(self.device):
                finalize_layerwise_reload(model, self.model_config)

        self._skyrl_weight_update_active = False
        self._skyrl_is_checkpoint_format = True
