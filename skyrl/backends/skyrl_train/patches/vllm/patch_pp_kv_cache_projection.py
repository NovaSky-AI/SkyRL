"""Runtime patch: allocate a pipeline stage's KV cache only for the layers it owns.

``get_kv_cache_configs`` builds global KV cache groups and projects them onto each PP stage
(``_project_kv_cache_groups_to_worker``). A group with none of its layers on a stage keeps its full
``UniformTypeKVCacheSpecs``, so that stage's config gets cache tensors for layers it does not own, and
``allocate_kv_cache`` hits ``StopIteration`` looking them up in the stage's groups. DeepSeek-V4.1 at PP=2
(0-19 | 20-39) trips it: its ratio-2 layers are all on stage 0.

All of a config's tensors view one shared backing allocation, so dropping the foreign ones frees nothing and
loses nothing.

TODO: remove this patch and its call in ``new_inference_worker_wrap.py`` once SkyRL pins a vLLM release
that includes vllm-project/vllm#60234 (merged to main 2026-10-27, commit 5c1eb11cb3; not in 0.32.0). It builds
each stage's KV cache tensors from the projected ``group.layer_names`` (``kv_cache_planning.py``) instead of the
group spec's full layer list, so no stage gets tensors for layers it does not own. Upstream tracking: issue
vllm-project/vllm#63615; its regression test is vllm-project/vllm#63617. Before removing, check that the pinned
vLLM's tensor loop iterates ``group.layer_names``.
"""

import dataclasses
import functools

from loguru import logger

_PATCHED = False


def apply_pp_kv_cache_projection_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.v1.worker import gpu_model_runner
        from vllm.v1.worker import utils as worker_utils
        from vllm.v1.worker.gpu import attn_utils
    except ImportError:
        return False
    original = worker_utils.allocate_kv_cache

    @functools.wraps(original)
    def allocate_kv_cache(kv_cache_config, *args, **kwargs):
        owned = {name for group in kv_cache_config.kv_cache_groups for name in group.layer_names}
        tensors = [t for t in kv_cache_config.kv_cache_tensors if t.layers[0] in owned]
        if len(tensors) != len(kv_cache_config.kv_cache_tensors):
            kv_cache_config = dataclasses.replace(kv_cache_config, kv_cache_tensors=tensors)
        return original(kv_cache_config, *args, **kwargs)

    for module in (worker_utils, attn_utils, gpu_model_runner):
        if getattr(module, "allocate_kv_cache", None) is original:
            module.allocate_kv_cache = allocate_kv_cache
    _PATCHED = True
    logger.info("Patched vLLM allocate_kv_cache to skip KV cache tensors of layers on other PP stages")
    return True
