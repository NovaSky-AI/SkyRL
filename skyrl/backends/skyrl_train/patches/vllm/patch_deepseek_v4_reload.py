"""Runtime patch: make vLLM's DeepSeek-V4 and V4.1 models survive a weight-sync reload.

vLLM's layerwise reload holds every layer on the meta device until that layer's ``weight_loader`` calls have
received all of its weights. ``DeepseekV4ForCausalLM``'s model-level finalize doesn't fit it. This backports the
upstream vLLM fix, in the upstream shape, to the pinned vLLM (0.33.0). (The attention-sink half of this patch,
vllm-project/vllm#57797, is in vLLM 0.33.0 -- ``make_attention_sink`` -- and was dropped.)

**Model-level finalize** (vllm-project/vllm#64357, open). ``DeepseekV4ForCausalLM.load_weights`` ends every
call with ``process_weights_after_loading`` (the mHC broadcast derived from layer 0's
``hc_attn_fn``). A weight sync calls ``load_weights`` once per chunk, so that runs against layers still on meta
(``NotImplementedError: Cannot copy out of meta tensor; no data!``), and ``finalize_layerwise_reload`` never runs
the model-level hook. As upstream, ``load_weights`` no longer runs the hook (the cold-load path already runs it
once) and ``finalize_layerwise_reload`` runs it once at the end of every reload (``finalize_model_weights``).
Unlike upstream, the reload hook is limited to the classes patched here. As upstream, V4.1's Engram page-cache
drop and huge-page collapse move from ``load_weights`` into the hook.

It applies to DeepSeek-V4 and to V4.1's text model (``vllm.models.deepseek_v41``, same loaders).

TODO: remove once the pinned vLLM includes #64357 (``finalize_layerwise_reload`` calls ``finalize_model_weights``).
"""

import sys
from itertools import islice
from typing import Any, Iterable, Tuple

from loguru import logger

_PATCHED = False
# Classes whose model-level hook runs after a layerwise reload.
_FINALIZE_AFTER_RELOAD: tuple = ()


def finalize_model_weights(model: Any) -> None:
    """vllm#64357's model-level post-load hook, for the classes patched here."""
    if isinstance(model, _FINALIZE_AFTER_RELOAD):
        model.process_weights_after_loading()


def apply_deepseek_v4_reload_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED, _FINALIZE_AFTER_RELOAD
    if _PATCHED:
        return False
    # (causal LM class, AutoWeightsLoader, extra post-load step)
    targets = []
    try:
        from vllm.models.deepseek_v4.nvidia import model as v4_model

        targets.append((v4_model.DeepseekV4ForCausalLM, v4_model.AutoWeightsLoader, None))
    except ImportError:
        pass
    try:
        # DeepSeek-V4.1 (text-only class; see patch_deepseek_v41.py) has its own copy.
        # TODO: remove this target with the rest of the module. vllm#64357 moves V4.1's hook and Engram page-cache
        # step out of ``DeepseekV41LLMForCausalLM.load_weights`` (check it no longer calls
        # ``process_weights_after_loading``).
        from vllm.models.deepseek_v41.nvidia import model as v41_model

        targets.append((v41_model.DeepseekV41LLMForCausalLM, v41_model.AutoWeightsLoader, _collapse_engram_pages))
    except ImportError:
        pass
    if not targets:
        return False
    for causal_lm_cls, auto_weights_loader, after_loading in targets:
        _patch_load_weights(causal_lm_cls, auto_weights_loader, after_loading)
    _FINALIZE_AFTER_RELOAD = tuple(target[0] for target in targets)
    _patch_finalize_layerwise_reload()
    _PATCHED = True
    logger.info("Installed DeepSeek-V4/V4.1 weight-sync reload patch (vllm#64357 finalize)")
    return True


def _patch_load_weights(causal_lm_cls, auto_weights_loader, after_loading=None) -> None:
    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set:
        # vllm#64357: load only; the model-level hook runs once after the whole load or reload.
        return auto_weights_loader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)

    causal_lm_cls.load_weights = load_weights
    if after_loading is None:
        return
    original_hook = causal_lm_cls.process_weights_after_loading

    def process_weights_after_loading(self) -> None:
        original_hook(self)
        after_loading(self)

    causal_lm_cls.process_weights_after_loading = process_weights_after_loading


def _collapse_engram_pages(model) -> None:
    """V4.1's post-load Engram step, moved from load_weights into the hook as in vllm#64357."""
    from vllm.model_executor.model_loader.weight_utils import drop_checkpoint_cache

    config = model.model.vllm_config
    if config.engram_config and config.engram_config.use_thp:
        # Loading weights refills the file cache; release it before MADV_COLLAPSE.
        model_config = config.model_config
        drop_checkpoint_cache(
            model_config.model_weights or model_config.model,
            revision=model_config.revision,
            cache_dir=config.load_config.download_dir,
        )
        for layer in islice(model.model.layers, model.model.start_layer, model.model.end_layer):
            if layer.engram is not None:
                layer.engram.embed_tokens.collapse_huge_pages()


def _patch_finalize_layerwise_reload() -> None:
    import vllm.model_executor.model_loader.reload as reload_pkg
    from vllm.model_executor.model_loader.reload import layerwise

    original = layerwise.finalize_layerwise_reload

    def finalize_layerwise_reload(model, model_config):
        original(model, model_config)
        finalize_model_weights(model)

    # Modules that imported the function by name before this patch ran.
    for module in (
        layerwise,
        reload_pkg,
        sys.modules.get("vllm.model_executor.model_loader.reload.torchao_decorator"),
        sys.modules.get("vllm.v1.worker.gpu_model_runner"),
    ):
        if module is not None and getattr(module, "finalize_layerwise_reload", None) is original:
            module.finalize_layerwise_reload = finalize_layerwise_reload
