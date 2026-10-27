"""Runtime patch: zero DeepSeek-V4.1's BF16 FlashMLA prefill KV workspace before every gather.

``DeepseekV4FlashMLAAttention._forward_prefill`` (``vllm/models/deepseek_v41/nvidia/flashmla.py``, the
``FLASHMLA_SPARSE_DSV41`` backend) gathers each prefill chunk's compressed and sliding-window KV into a buffer
from the shared workspace manager, one ``chunk_M``-row slab per request sized for the longest request in the
chunk. Shorter requests leave slab rows unwritten, and the workspace is reused, so those rows hold whatever an
earlier kernel left there. ``flash_mla_sparse_fwd`` loads them and weights them 0: finite leftovers are harmless,
but a NaN or Inf bit pattern makes the request's attention output (and so its logits) NaN, and its first sampled
token is drawn from a NaN distribution.

Measured with one-token generations over distinct prompts: on the full BF16 model (TP8, CUDA graphs) 0.59% NaN
first tokens as shipped and 0 with this patch (32,768 requests each); on a 6-layer slice (24,576 requests) 0.84% as
shipped, 0 with the workspace zeroed and 46.7% with it NaN-filled (about half of batched requests touch unwritten
rows). Zero- vs 1000-filled first-token logprobs agree to noise, so the rows are masked, not attended.
Single-request prefill never hits it.

Zeroing costs one memset of the gather buffer per prefill chunk per layer.

The root cause is stale slots in the reused sparse-index workspace: ``combine_topk_swa_indices`` writes only each
row's valid prefix, and FlashMLA loads the whole last 64-slot block, so a stale slot can point at a non-finite row of
this gather buffer. Zeroing the buffer makes whatever such a slot points at finite.

TODO: remove this patch (and its call in ``new_inference_worker_wrap.py`` and its test) once SkyRL pins a vLLM
release that includes vllm-project/vllm#60613 ("[DSv4.1] Seed the -1 sentinel in the prefill sparse index
workspace", merged to main 2026-10-23; not in 0.32.0, published 2026-10-18). It adds ``combined_indices.fill_(-1)``
to V4.1's ``combine_topk_swa_indices`` (``vllm/models/deepseek_v41/common/ops/cache_utils.py``), porting
vllm-project/vllm#55299, which fixed DeepSeek-V4's copy (in 0.31.0+, so V4 needs no patch). Before removing, check
that the pinned vLLM's V4.1 ``combine_topk_swa_indices`` seeds ``-1`` (that line, or the in-kernel tail fill of
vllm-project/vllm#63264 / #62365).
"""

from loguru import logger

_PATCHED = False


class _ZeroedGatherWorkspace:
    """Workspace-manager proxy whose ``get_simultaneous`` zeroes the first (gathered-KV) buffer it returns."""

    def __init__(self, manager):
        self._manager = manager

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def get_simultaneous(self, *specs):
        tensors = self._manager.get_simultaneous(*specs)
        tensors[0].zero_()
        return tensors


def apply_deepseek_v41_prefill_workspace_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        import vllm.models.deepseek_v41.nvidia.flashmla as flashmla
    except ImportError:
        return False
    original = flashmla.current_workspace_manager

    def current_workspace_manager():
        return _ZeroedGatherWorkspace(original())

    flashmla.current_workspace_manager = current_workspace_manager
    _PATCHED = True
    logger.info("Installed DeepSeek-V4.1 FlashMLA prefill workspace zero-fill patch")
    return True
