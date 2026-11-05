"""Runtime patch: load reloaded weights into the layer as they arrive instead of buffering them.

vLLM's layerwise reload (``model_loader/reload/layerwise.py``) wraps every parameter's
weight loader so that a weight update only records each ``(param, loaded_weight)`` call;
once a layer's last element arrives it materializes the layer, replays the recorded
calls, and re-runs quantization. The recorded calls keep the incoming tensors alive,
and those are full checkpoint tensors: the packed IPC/NCCL receivers clone each one
at its unsharded size (``packed_tensor.py``) and the TP narrowing only happens in the
replayed loader. A routed-expert layer is therefore held whole on every TP rank until
all its experts arrive -- ~19 GiB of bf16 per GLM-5.3 MoE layer, on top of the
materialized shard and the ~89 GiB of FP8 weights of a TP8 engine, which is what
OOMed colocated full-weight syncs of GLM-5.3 on B200.

This materializes the layer on its first incoming weight during a reload and runs the
original loader straight into it (narrowing to the local shard), so each incoming
tensor can be freed as soon as it is copied. Completion and post-processing are the
same as ``_layerwise_process``. Deferred attention layers and first-time loads (no
kernel tensors yet, e.g. online quantization at startup) keep vLLM's path.

Remove once vLLM's layerwise reload stops buffering unsharded incoming tensors.
"""

import inspect
from functools import wraps

from loguru import logger

_PATCHED = False
_LOADED_NAMES = "_skyrl_eager_loaded_names"


def apply_layerwise_reload_eager_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.model_executor.model_loader.reload import layerwise
    except ImportError:
        return False

    from vllm.model_executor.layers.attention import is_deferred_attention_layer
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader.reload.meta import (
        get_numel_loaded,
        materialize_layer,
    )
    from vllm.model_executor.model_loader.reload.utils import (
        get_layer_size,
        get_layer_tensors,
    )

    original_make_loader = layerwise.make_online_process_loader

    def _finish(layer, info, loaded_names):
        # `_layerwise_process` minus materialization and replay: the data is already in place.
        if hasattr(layer, "_already_called_process_weights_after_loading"):
            delattr(layer, "_already_called_process_weights_after_loading")
        for tensor in get_layer_tensors(layer).values():
            tensor.weight_loader = layerwise._get_original_loader(tensor)
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)
            if hasattr(layer, "update_param_tp_status"):
                layer.update_param_tp_status()
        # `_copy_and_restore_kernel_tensors` only reads the names in `loaded_weights`.
        info.loaded_weights = [(name, None) for name in loaded_names]
        layerwise._copy_and_restore_kernel_tensors(layer, info)
        info.reset()

    def make_online_process_loader(layer, param_name):
        buffered_loader = original_make_loader(layer, param_name)
        info = layerwise.get_layerwise_info(layer)
        original_loader = layerwise._get_original_loader(getattr(layer, param_name))
        signature = inspect.signature(original_loader)

        @wraps(original_loader, assigned=("__doc__", "__annotations__"))
        def online_process_loader(*args, **kwargs):
            if info.kernel_tensors is None or is_deferred_attention_layer(layer):
                return buffered_loader(*args, **kwargs)
            if not info.can_load():
                return None  # excessive loading after the layer was processed, as upstream

            info.load_numel_total = get_layer_size(layer)
            layerwise._wrap_parameters_weight_loader(layer)
            # `info.reset()` re-runs the dataclass __init__ and would leave a custom attribute
            # behind, so key per-pass state off the pass itself: the first load of a pass
            # starts a fresh name set, and meta tensors are what still needs materializing.
            if info.load_numel == 0 or not hasattr(info, _LOADED_NAMES):
                setattr(info, _LOADED_NAMES, set())
            if any(t.is_meta for t in get_layer_tensors(layer).values()):
                materialize_layer(layer, info)
            loaded_names = getattr(info, _LOADED_NAMES)

            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            target = getattr(layer, param_name)
            bound.arguments["param"] = target
            num_loaded, ret = get_numel_loaded(layerwise._get_original_loader(target), bound)
            info.load_numel += num_loaded
            loaded_names.add(param_name)

            if info.load_numel >= info.load_numel_total:
                _finish(layer, info, loaded_names)
            return ret

        return online_process_loader

    layerwise.make_online_process_loader = make_online_process_loader
    _PATCHED = True
    logger.info("Installed eager layerwise-reload patch (no buffering of unsharded incoming weights)")
    return True
