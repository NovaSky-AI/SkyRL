"""Runtime patch: keep routed-experts capture bound when a MoE quant method rebuilds its kernel.

Backport of vllm-project/vllm#59455 (fixes #59449). ``bind_routed_experts_capturer`` binds the
capture callback once, at model-runner init; for monolithic kernels (FlashInfer TRT-LLM FP8, the
default for FP8 MoE on B200) it lands on ``moe_kernel.impl.fused_experts``. FP8 MoE
``process_weights_after_loading`` rebuilds ``moe_kernel`` (``_init_moe_kernel``), which runs on
every layerwise weight reload, i.e. every full-weight sync. The rebuilt experts had no callback,
so from the first sync on ``routed_experts`` returned the startup profile run's routing for every
prefill token and R3 replayed one fixed expert set over every prompt.

This turns ``FusedMoEMethodBase.moe_kernel`` into a property whose setter carries the callback
over to the rebuilt monolithic experts (and raises if the new kernel cannot capture). Install
before any MoE layer is built.

Remove once the pinned vLLM includes #59455 (merged upstream 2026-10-02; not in 0.30.0).
"""

from loguru import logger

_PATCHED = False


def apply_routed_experts_rebind_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
            FusedMoEMethodBase,
        )
    except ImportError:
        return False
    if isinstance(FusedMoEMethodBase.__dict__.get("moe_kernel"), property):
        return False  # the pinned vLLM already has the fix

    def _monolithic_fused_experts(kernel):
        fused_experts = getattr(getattr(kernel, "impl", None), "fused_experts", None)
        if isinstance(fused_experts, mk.FusedMoEExpertsMonolithic):
            return fused_experts
        return None

    def _get(self):
        return self.__dict__.get("_moe_kernel")

    def _set(self, kernel):
        previous = _monolithic_fused_experts(self.__dict__.get("_moe_kernel"))
        self.__dict__["_moe_kernel"] = kernel
        capture_fn = getattr(previous, "routing_replay_capture_fn", None)
        if capture_fn is None or kernel is None:
            return
        fused_experts = _monolithic_fused_experts(kernel)
        if fused_experts is None or not fused_experts.supports_routing_replay_capture():
            raise ValueError(
                "Routed-experts capture is not supported with monolithic MoE "
                f"kernel {type(getattr(kernel.impl, 'fused_experts', None)).__name__}."
            )
        if fused_experts.routing_replay_capture_fn is capture_fn:
            return
        # Move the previous replay buffer over instead of letting set_capture_fn allocate a new
        # one: decode CUDA graphs recorded at startup write into the old buffer's address, so it
        # must stay alive and stay the buffer the eager path reads. Freeing it let the graphs
        # write int16 expert IDs into whatever reused that memory (device-side index asserts).
        buffer = getattr(previous, "_routing_replay_buffer", None)
        if buffer is None:
            fused_experts.set_capture_fn(capture_fn)
            return
        fused_experts.routing_replay_capture_fn = capture_fn
        fused_experts._routing_replay_buffer = buffer

    FusedMoEMethodBase.moe_kernel = property(_get, _set)
    _PATCHED = True
    logger.info("Installed routed-experts rebind patch (vllm#59455 backport)")
    return True
