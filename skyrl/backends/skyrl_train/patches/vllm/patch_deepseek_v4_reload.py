"""Runtime patch: make vLLM's DeepSeek-V4 and V4.1 models survive a weight-sync reload.

vLLM's layerwise reload holds every layer on the meta device until that layer's ``weight_loader``
calls have received all of its weights. Two pieces of ``DeepseekV4ForCausalLM`` loading don't fit it:

1. **Model-level finalize.** ``DeepseekV4ForCausalLM.load_weights`` ends every call with
   ``process_weights_after_loading`` (``finalize_mega_moe_weights`` + ``finalize_mhc_broadcast_weights``,
   which derives layer 0's ``hc_attn_fn_broadcast`` from ``hc_attn_fn``). A weight sync calls
   ``load_weights`` once per received chunk, so the first chunk finalizes against a meta
   ``hc_attn_fn`` (``NotImplementedError: Cannot copy out of meta tensor; no data!``), and
   ``finalize_layerwise_reload`` never re-runs the model-level hook. While a SkyRL receive is loading
   (``defer_model_finalize``) the hook only marks itself pending; ``run_deferred_model_finalize`` runs
   it once after vLLM's layerwise finalize, as a fresh load does.

2. **Attention sinks.** ``DeepseekV4Model.load_weights`` writes ``attn_sink`` with a direct
   ``params_dict[name][:n].copy_()`` instead of the parameter's ``weight_loader``. Under layerwise
   reload that copy lands in the meta placeholder and is dropped; the attention module logs
   ``Failed to load weights`` and gets its pre-reload storage back. After a level-2 sleep that storage
   is zeroed, so every sink -- real heads and the ``-inf`` padding heads -- serves as 0. Here the sink
   tensors are taken out of the stream when their parameter is on meta and written into the restored
   parameter (same per-rank head slice, ``-inf`` padding) by ``run_deferred_model_finalize``.

Remove once vLLM loads ``attn_sink`` through a weight loader and runs the model-level finalize once at
the end of a layerwise reload.
"""

from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Tuple

from loguru import logger

_PATCHED = False
_DEFER_ATTR = "_skyrl_defer_model_finalize"
_PENDING_ATTR = "_skyrl_model_finalize_pending"
_SINK_STASH_ATTR = "_skyrl_pending_attn_sinks"


def apply_deepseek_v4_reload_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    pairs = []
    try:
        from vllm.models.deepseek_v4.nvidia.model import (
            DeepseekV4ForCausalLM,
            DeepseekV4Model,
        )

        pairs.append((DeepseekV4ForCausalLM, DeepseekV4Model))
    except ImportError:
        pass
    try:
        # DeepSeek-V4.1 (text-only class; see patch_deepseek_v41.py) has its own copies of both.
        from vllm.models.deepseek_v41.nvidia.model import (
            DeepseekV4Model as DeepseekV41Model,
        )
        from vllm.models.deepseek_v41.nvidia.model import (
            DeepseekV41LLMForCausalLM,
        )

        pairs.append((DeepseekV41LLMForCausalLM, DeepseekV41Model))
    except ImportError:
        pass
    if not pairs:
        return False
    for causal_lm_cls, model_cls in pairs:
        _patch_classes(causal_lm_cls, model_cls)
    _PATCHED = True
    logger.info("Installed DeepSeek-V4/V4.1 weight-sync reload patch (deferred finalize, attention sinks)")
    return True


def _patch_classes(causal_lm_cls, model_cls) -> None:
    original_finalize = causal_lm_cls.process_weights_after_loading

    def process_weights_after_loading(self) -> None:
        if getattr(self, _DEFER_ATTR, False):
            setattr(self, _PENDING_ATTR, True)
            return
        original_finalize(self)

    original_load = model_cls.load_weights

    def load_weights(self, weights: Iterable[Tuple[str, Any]]) -> set:
        params = None
        stashed: dict = self.__dict__.setdefault(_SINK_STASH_ATTR, {})
        stashed_now = set()

        def without_meta_sinks() -> Iterator[Tuple[str, Any]]:
            nonlocal params
            for name, weight in weights:
                if "attn_sink" in name:
                    if params is None:
                        params = dict(self.named_parameters())
                    param = params.get(name)
                    if param is not None and param.is_meta:
                        stashed[name] = weight.detach().clone()
                        stashed_now.add(name)
                        continue
                yield name, weight

        return original_load(self, without_meta_sinks()) | stashed_now

    causal_lm_cls.process_weights_after_loading = process_weights_after_loading
    model_cls.load_weights = load_weights


@contextmanager
def defer_model_finalize(model: Any) -> Iterator[None]:
    """Make a patched model's ``process_weights_after_loading`` a pending marker for the duration."""
    setattr(model, _DEFER_ATTR, True)
    try:
        yield
    finally:
        setattr(model, _DEFER_ATTR, False)


def _apply_stashed_attn_sinks(model: Any) -> None:
    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for module in modules():
        stashed = module.__dict__.get(_SINK_STASH_ATTR)
        if not stashed:
            continue
        import torch
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        # Same per-rank head slice as DeepseekV4Model.load_weights.
        n_local_head = module.config.num_attention_heads // get_tensor_model_parallel_world_size()
        start = n_local_head * get_tensor_model_parallel_rank()
        params = dict(module.named_parameters())
        with torch.no_grad():
            for name, weight in stashed.items():
                param = params[name]
                narrow = weight[start : start + n_local_head]
                param[narrow.shape[0] :].fill_(-float("inf"))
                param[: narrow.shape[0]].copy_(narrow)
        logger.info(f"Restored {len(stashed)} DeepSeek-V4 attention sinks after layerwise reload")
        stashed.clear()


def run_deferred_model_finalize(model: Any) -> None:
    """After vLLM's layerwise finalize: restore stashed sinks, then run a pending model-level finalize."""
    _apply_stashed_attn_sinks(model)
    if getattr(model, _PENDING_ATTR, False):
        setattr(model, _PENDING_ATTR, False)
        model.process_weights_after_loading()
