"""Runtime patch: run DeepSeek-V4's model-level finalize once per weight sync, not once per chunk.

``DeepseekV4ForCausalLM.load_weights`` ends every call with ``process_weights_after_loading``
(``finalize_mega_moe_weights`` + ``finalize_mhc_broadcast_weights``, which derives layer 0's
``hc_attn_fn_broadcast`` from ``hc_attn_fn``). A weight sync calls ``load_weights`` once per
received chunk, inside vLLM's layerwise reload, which keeps each layer on the meta device until all
of its weights have arrived. The first chunk therefore finalizes against a meta ``hc_attn_fn`` and
raises ``NotImplementedError: Cannot copy out of meta tensor; no data!``, and
``finalize_layerwise_reload`` never re-runs the model-level hook.

While a SkyRL receive is loading weights (``defer_model_finalize``), the model-level hook only marks
itself pending; ``run_deferred_model_finalize`` runs it once after vLLM's layerwise finalize has put
every layer back on device, which is what a fresh load does.

Remove once vLLM skips the model-level finalize during layerwise reload and runs it at the end.
"""

from contextlib import contextmanager
from typing import Any, Iterator

from loguru import logger

_PATCHED = False
_DEFER_ATTR = "_skyrl_defer_model_finalize"
_PENDING_ATTR = "_skyrl_model_finalize_pending"


def apply_deepseek_v4_reload_finalize_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
    except ImportError:
        return False

    original = DeepseekV4ForCausalLM.process_weights_after_loading

    def process_weights_after_loading(self) -> None:
        if getattr(self, _DEFER_ATTR, False):
            setattr(self, _PENDING_ATTR, True)
            return
        original(self)

    DeepseekV4ForCausalLM.process_weights_after_loading = process_weights_after_loading
    _PATCHED = True
    logger.info("Installed DeepSeek-V4 deferred model-level finalize for weight-sync reloads")
    return True


@contextmanager
def defer_model_finalize(model: Any) -> Iterator[None]:
    """Make a patched model's ``process_weights_after_loading`` a pending marker for the duration."""
    setattr(model, _DEFER_ATTR, True)
    try:
        yield
    finally:
        setattr(model, _DEFER_ATTR, False)


def run_deferred_model_finalize(model: Any) -> None:
    """Run the model-level finalize if a deferred reload left it pending."""
    if getattr(model, _PENDING_ATTR, False):
        setattr(model, _PENDING_ATTR, False)
        model.process_weights_after_loading()
