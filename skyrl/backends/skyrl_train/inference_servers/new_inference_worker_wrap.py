"""
vLLM worker-extension hook for SkyRL (new inference path).

vLLM 0.23.0 handles RL weight sync natively: the trainer drives
``/init_weight_transfer_engine`` -> ``/start_weight_update`` ->
``/update_weights`` (packed CUDA IPC or NCCL) -> ``/finish_weight_update``
against the inference server, and the GPUWorker's ``weight_transfer_engine``
(selected via ``weight_transfer_config``) receives and loads the weights. SkyRL
therefore no longer needs to inject worker methods to receive/load weights, and
the previous ``skyrl_start_weight_update`` / ``update_weights_ipc`` /
``update_weights_nccl`` / ``skyrl_finish_weight_update`` methods are gone.

This class is kept deliberately as the **injection point for SkyRL-specific
worker behavior**: it is passed to vLLM via ``--worker-extension-cls`` and mixed
into the GPUWorker. Add a targeted method/override here only when upstream vLLM
lacks a feature/fix we need, or when our pinned vLLM version doesn't have it yet.

The one piece of custom behavior currently required is applying the ``#44814``
layerwise-reload numel patch (see ``layerwise_reload.patch_numel_loaded``): the
fix is not in vLLM 0.23.0 (it lands in 0.23.1), and the native
``finish_weight_update`` -> ``finalize_layerwise_reload`` -> ``get_numel_loaded``
over-counts elements for composed weight loaders, silently dropping trailing
params (e.g. Mamba ``mixer.D``). We patch the vLLM function globally as soon as
this module is imported in the worker process (vLLM imports it there when
resolving ``--worker-extension-cls``). Remove once we bump to vLLM >= 0.23.1.

Usage:
    Pass as --worker-extension-cls to vLLM:

    vllm serve ... --worker-extension-cls \
        skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap.NewInferenceWorkerWrap
"""

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    patch_numel_loaded,
)

VLLM_NEW_INFERENCE_WORKER_EXTENSION_CLS = f"{__name__}.NewInferenceWorkerWrap"


class NewInferenceWorkerWrap:
    """SkyRL custom-behavior hook injected into the vLLM GPUWorker.

    Intentionally near-empty: vLLM 0.23.0 performs weight sync natively, so
    there are no SkyRL receive/load methods here anymore. Keep this class as
    the place to add worker-side overrides when upstream is missing something
    we need (or our pinned version is). See the module docstring.
    """

    pass


# Apply the #44814 layerwise-reload numel patch at import time. vLLM loads this
# module in each worker process when resolving ``--worker-extension-cls``; the
# native ``/finish_weight_update`` path (now used instead of SkyRL's old
# ``skyrl_finish_weight_update``, which used to apply the patch lazily) would
# otherwise hit the un-capped ``get_numel_loaded``. Guarded so the module stays
# importable where vllm isn't available (non-Linux / CPU CI). Remove once we
# bump to vLLM >= 0.23.1.
try:
    patch_numel_loaded()
except Exception:  # pragma: no cover - vllm not importable in this process
    pass
