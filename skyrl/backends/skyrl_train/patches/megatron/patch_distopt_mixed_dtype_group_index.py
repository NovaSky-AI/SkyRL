"""Point the distributed optimizer's model-param -> main-param index at the right main params.

``DistributedOptimizer._build_optimizer_group_ranges`` records each model param as
``(group_index, position in group_range["params"])``, where ``group_range["params"]`` follows grad-buffer
order -- the bf16 buffer's params, then the fp32 buffer's. ``_build_model_and_main_param_groups`` then sets
the optimizer group's params to ``[*shard_fp32_params, *shard_fp32_from_float16_params]``: fp32 first. In a
param group that holds both dtypes the recorded positions are rotated, so each model param resolves to the
next one's main param and optimizer state.

Training is unaffected (the main <-> model copies walk ``model_*_groups`` / ``shard_*_groups`` in lockstep);
the index is only read by checkpoint save/load (``_get_main_param_and_optimizer_states``,
``_set_main_param_and_optimizer_states``, ``load_state_dict``). Any model that keeps some trained params in
fp32 under bf16 training hits it -- DeepSeek-V4's mHC ``mapping_proj`` and compressor ``ape``, GLM-5.3's mHC
-- and full-finetune checkpoint save fails with a shape mismatch in ``sharded_param_state_*``
(``AssertionError: (torch.Size([362976]), 0, 120968256)``). LoRA rows pass because only adapters train.

TODO: remove once the pinned megatron-core rebuilds the index itself. Upstream fixes, same rebuild as this patch
(``[*model_fp32_params, *model_float16_params]`` per group, at the end of ``DistributedOptimizer.__init__``):
NVIDIA/Megatron-LM#2872 (open, ``main``) and NVIDIA/Megatron-LM#5835 (merged 2026-07-18 into ``dev`` only, not in
SkyRL's ``main`` pin). Check that ``DistributedOptimizer.__init__`` in the pinned version rebuilds
``model_param_group_index_map`` before removing; see ``README.md``.
"""

import functools

from loguru import logger

_APPLIED = False


def rebuild_model_param_group_index_map(model_fp32_groups, model_float16_groups) -> dict:
    """Index each model param by its main param's position: the group's fp32 params, then its fp16/bf16 ones."""
    index_map = {}
    for group_index, (fp32_params, float16_params) in enumerate(zip(model_fp32_groups, model_float16_groups)):
        for order, param in enumerate(fp32_params):
            index_map[param] = (group_index, order)
        for order, param in enumerate(float16_params, start=len(fp32_params)):
            index_map[param] = (group_index, order)
    return index_map


def apply_distopt_mixed_dtype_group_index_patch() -> bool:
    """Wrap ``DistributedOptimizer.__init__`` to rebuild ``model_param_group_index_map``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    except ImportError:
        return False

    inner = DistributedOptimizer.__init__

    @functools.wraps(inner)
    def __init__(self, *args, **kwargs):
        inner(self, *args, **kwargs)
        fp32_groups = getattr(self, "model_fp32_groups", None)
        float16_groups = getattr(self, "model_float16_groups", None)
        if fp32_groups is None or float16_groups is None:
            return
        rebuilt = rebuild_model_param_group_index_map(fp32_groups, float16_groups)
        if rebuilt != self.model_param_group_index_map:
            moved = sum(1 for p, v in rebuilt.items() if self.model_param_group_index_map.get(p) != v)
            logger.info(f"Distributed optimizer: re-indexed {moved} params in mixed fp32/bf16 param groups")
        self.model_param_group_index_map = rebuilt

    DistributedOptimizer.__init__ = __init__
    _APPLIED = True
    return True
