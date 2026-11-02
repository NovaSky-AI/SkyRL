"""Load the pinned FlashMLA extension before vLLM's bundled copy.

megatron-core's DeepSeek-V4 sparse attention calls ``flash_mla.flash_mla_sparse_fwd`` from our pinned
``flash-mla`` wheel (``nv_dev`` @ b7643bd). vLLM ships its own, older FlashMLA build as
``vllm/_flashmla_C.abi3.so``. Both statically link FlashMLA's C++ with the same exported symbols, so
whichever library the process loads first wins symbol resolution for the other. With vLLM's loaded
first, the pinned forward dispatches through the older implementation table, whose SM100 head-64
kernel lacks ``TOPK_LENGTH``: ``[FlashMLA] Error: The chosen implementation does not support all
required features`` on CSA's no-grad (compacted top-k) path. Loaded the other way round, both work.

Megatron worker processes import vLLM (weight sync, LoRA) before megatron-core imports FlashMLA
lazily on the first forward, in an order that depends on what the Ray actor class drags in.
``skyrl.backends.skyrl_train`` calls this on package import, which precedes both.

Delete this when the pinned wheel is built with hidden symbol visibility, or vLLM's bundled FlashMLA
matches it.
"""

import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)


def preload_flash_mla() -> bool:
    """Import the pinned ``flash_mla.cuda`` if it is installed. Returns True if it is now loaded.

    A no-op in processes that already imported vLLM: in vLLM's own workers (which import SkyRL's
    worker extension after vLLM) vLLM's FlashMLA must keep its own symbols.
    """
    if "flash_mla.cuda" in sys.modules:
        return True
    if "vllm" in sys.modules or importlib.util.find_spec("flash_mla") is None:
        return False
    try:
        import flash_mla.cuda  # noqa: F401
    except Exception as e:  # noqa: BLE001 -- a missing CUDA driver must not break package import
        logger.debug(f"Could not preload flash_mla: {e}")
        return False
    return True
