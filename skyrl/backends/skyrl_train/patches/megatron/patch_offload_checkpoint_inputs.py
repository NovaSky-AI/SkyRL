"""Keep full-recompute checkpoint inputs in host memory (opt-in).

Under ``recompute_granularity="full"`` each layer's forward runs under ``no_grad`` inside
megatron-core's ``checkpointed_forward``, and the only tensors autograd keeps are each checkpoint's
inputs -- one hidden-state tensor per layer, alive until backward reaches that layer. For
GLM-5.3-Flash those are the mHC n-stream residuals (``[s / TP, b, 4 * hidden]``): 45 of them,
~1.4 MB per token per GPU at TP8, the largest term in long-context memory after the recompute
working set.

``torch.autograd.graph.save_on_cpu`` around the block-level ``checkpointed_forward`` (in both
``transformer_block`` and ``hybrid_block``, which import it by name) moves exactly
those to host memory and brings each back when backward unpacks it. The recompute inside
backward runs outside the context, so its activations stay on the GPU. Host cost: the same bytes
in host RAM per GPU (e.g. ~45 GiB/GPU at 256k tokens on 64 GPUs); PCIe cost: one D2H + one H2D
copy per layer per step.

Enable with ``SKYRL_OFFLOAD_CHECKPOINT_INPUTS=1``.
"""

import functools
import importlib
import os

import torch
from loguru import logger

_APPLIED = False
# Pageable by default: the pinned path goes through PyTorch's caching host allocator, which rounds
# each block up to a power of two. For GLM-5.3-Flash at 576k tokens a 2.25 GiB checkpoint input
# lands in a 4 GiB block, ~1.4 TB/node of host RAM, and the node ran out. Pageable copies are
# synchronous and slower, but small next to a long-context step.
_PIN_MEMORY = os.environ.get("SKYRL_OFFLOAD_CHECKPOINT_INPUTS_PINNED", "0").lower() in ("1", "true")


# Modules that do `from megatron.core.recompute import checkpointed_forward` and therefore hold
# their own reference; wrapping only one leaves the other's models silently unpatched. Same list as
# patch_dsa_index_share.py, for the megatron-core rev pinned in pyproject.toml.
_IMPORTERS = (
    "megatron.core.transformer.transformer_block",
    "megatron.core.models.hybrid.hybrid_block",
)


def _wrap(inner):
    @functools.wraps(inner)
    def checkpointed_forward(*args, **kwargs):
        with torch.autograd.graph.save_on_cpu(pin_memory=_PIN_MEMORY):
            return inner(*args, **kwargs)

    return checkpointed_forward


def patch_offload_checkpoint_inputs() -> bool:
    """Wrap every importer's ``checkpointed_forward`` in ``save_on_cpu``. Idempotent.

    Wraps each module's own current reference, so it composes with ``patch_dsa_index_share``,
    which rebinds the same name in the same modules and must be applied first.
    """
    global _APPLIED
    if _APPLIED:
        return True
    wrapped = []
    for name in _IMPORTERS:
        module = importlib.import_module(name)
        module.checkpointed_forward = _wrap(module.checkpointed_forward)
        wrapped.append(name.rsplit(".", 1)[-1])
    _APPLIED = True
    logger.info(
        f"Full-recompute checkpoint inputs are kept in host memory (pinned={_PIN_MEMORY}); "
        f"wrapped checkpointed_forward in {', '.join(wrapped)}"
    )
    return True


def release_pinned_offload_cache() -> None:
    """Return cached pinned checkpoint buffers to the OS (pinned mode only).

    PyTorch's caching host allocator keeps freed pinned blocks reserved. After a step's
    forward_backward that is every checkpoint input (~90 GiB/GPU at 512k tokens on GLM-5.3-Flash),
    which the CPU optimizer's own host buffers then have to fit beside -- the pinned run hit the
    node's memory limit there. Called once per forward_backward (i.e. once per optimizer step,
    after all of its microbatches), so the re-pinning is amortized over the step's microbatches.
    """
    if _APPLIED and _PIN_MEMORY and hasattr(torch._C, "_host_emptyCache"):
        torch._C._host_emptyCache()
