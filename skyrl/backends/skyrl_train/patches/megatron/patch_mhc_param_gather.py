"""Publish HyperConnection projection weights before functional reads.

Core's mHC mapping kernels read ``mapping_proj.weight`` without invoking the
child Linear. DDP's module pre-hook therefore never waits for that child's
parameter gather. A prefetched gather can remain live through gradient
finalization, which clears its dispatch flag; the next forward then attempts
to dispatch it a second time. Even before that assertion, the functional read
can race with the gather.

Use Core's parameter-readiness protocol at the read boundary. This preserves
overlap for other buckets, including training with overlap_grad_reduce enabled.
Remove this adapter when the pinned Core publishes mHC projection parameters
before both its native and fused functional mapping paths.
"""

from functools import wraps

from loguru import logger


def make_param_ready_compute_mappings(original, ensure_params_ready):
    """Cover both mapping implementations without changing their computation."""

    @wraps(original)
    def compute_mappings(self, *args, **kwargs):
        ensure_params_ready(self.mapping_proj.parameters())
        return original(self, *args, **kwargs)

    compute_mappings._skyrl_mhc_param_ready = True
    return compute_mappings


def patch_mhc_param_gather() -> None:
    """Install once; Core releases without HyperConnections need no adapter."""
    try:
        from megatron.core.transformer.hyper_connection import HyperConnectionModule
    except ModuleNotFoundError as exc:
        if exc.name == "megatron.core.transformer.hyper_connection":
            return
        raise

    original = HyperConnectionModule.compute_mappings
    if getattr(original, "_skyrl_mhc_param_ready", False):
        return
    from megatron.core.utils import ensure_params_ready

    HyperConnectionModule.compute_mappings = make_param_ready_compute_mappings(original, ensure_params_ready)
    logger.info("Applied HyperConnection functional projection parameter readiness")
