"""Backport of vllm-project/vllm#56327 (LoRA support for GLM-5.3-Flash).

Two pieces, both absent from the pinned vLLM but both required for a Megatron-trained
GLM-5.3-Flash LoRA adapter to load into the engine (``merge_lora=false``):

1. ``Glm5NextForConditionalGeneration.packed_modules_mapping``. Without it the class
   inherits ``Glm4vForConditionalGeneration``'s ``{"qkv_proj": [q,k,v], "gate_up_proj": ...}``,
   which names none of GLM-5.3-Flash's fused runtime projections. The adapter stores the HF
   submodules separately (``q_proj``/``k_proj``/``v_proj``/``b_proj``/``f_a_proj``/``g_a_proj``
   and ``q_a_proj``/``kv_a_proj_with_mqa``), so with no mapping there is nothing to assemble
   them onto and the load fails.

2. ``MergedColumnParallelLinearWithLoRA.output_ids`` honoring ``replicated_shard_ids``.
   KDA's ``in_proj_qkvbfg_a`` is a ``_Glm5NextMergedColumnParallelLinear`` declaring
   ``replicated_shard_ids=(4, 5)``: ``f_a_proj``/``g_a_proj`` are duplicated across TP ranks
   rather than sharded (matching ``parallel_mode="duplicated"`` on the Megatron side), so
   their LoRA-B must be kept whole on every rank instead of sliced by ``tp_rank``. The base
   class already carries the attribute; only the LoRA layer's use of it is missing.

Note the upstream PR does *not* address the ``assert inputs.is_contiguous()`` in
``lora_shrink`` that a LoRA-wrapped ``f_b_proj``/``g_b_proj`` hits (KDA splits the fused
projection into non-contiguous views). Those two modules must still be left out of
``lora_target_modules``.

TODO: remove once https://github.com/vllm-project/vllm/pull/56327 lands in the pinned vLLM.
"""

import logging

logger = logging.getLogger(__name__)

_PATCHED = False

# Verbatim from the PR's Glm5NextForConditionalGeneration.packed_modules_mapping.
_GLM5NEXT_PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvbfg_a": [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
    ],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "wk_weights_proj": ["wk", "weights_proj"],
    "qkv": ["qkv"],
}


def _patch_packed_modules_mapping() -> list[str]:
    """Install the mapping on whichever GLM-5.3-Flash entrypoints this build exposes.

    ``language_model_only`` selects the ``ForCausalLM`` variant, so set both when present.
    """
    import vllm.models.glm5next as glm5next

    patched = []
    for cls_name in ("Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"):
        cls = getattr(glm5next, cls_name, None)
        if cls is None:
            continue
        # Only override the inherited (wrong) mapping, never a build that already has the fix.
        if cls.__dict__.get("packed_modules_mapping") == _GLM5NEXT_PACKED_MODULES_MAPPING:
            continue
        cls.packed_modules_mapping = dict(_GLM5NEXT_PACKED_MODULES_MAPPING)
        patched.append(cls_name)
    return patched


def _patch_replicated_shard_ids() -> bool:
    """Keep LoRA-B whole for shards the base layer marks as replicated."""
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearWithLoRA,
    )

    original_init = MergedColumnParallelLinearWithLoRA.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        replicated_shard_ids = getattr(self.base_layer, "replicated_shard_ids", ())
        if not replicated_shard_ids:
            return
        self.output_ids = tuple(0 if i in replicated_shard_ids else self.tp_rank for i in range(self.n_slices))

    MergedColumnParallelLinearWithLoRA.__init__ = __init__
    return True


def _patch_lora_shrink_contiguity() -> bool:
    """Make the LoRA shrink GEMM accept a non-contiguous input.

    Not part of vllm#56327 -- that PR leaves this untouched. KDA runs one fused
    ``in_proj_qkvbfg_a`` GEMM and ``.split()``s it, so ``f_a``/``g_a`` are views whose
    ``stride(0)`` is the fused width: the ``.view(-1, shape[-1])`` inside ``add_shrink``
    still succeeds (the last dim is unit-stride) but ``lora_shrink`` then trips
    ``assert inputs.is_contiguous()``. Without this, ``f_b_proj``/``g_b_proj`` cannot be
    LoRA targets at all.

    The copy is only taken when the input is actually non-contiguous, and for KDA that is
    a ``[num_tokens, head_dim]`` bf16 tensor (head_dim=128 on GLM-5.3-Flash), so the cost
    is negligible next to the GEMM it feeds.
    """
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU

    original_add_shrink = PunicaWrapperGPU.add_shrink

    def add_shrink(self, y, x, lora_a_stacked, scale, **kwargs):
        if not x.is_contiguous():
            x = x.contiguous()
        return original_add_shrink(self, y, x, lora_a_stacked, scale, **kwargs)

    PunicaWrapperGPU.add_shrink = add_shrink
    return True


def apply_glm5next_lora_packing_patch() -> None:
    """Apply all three pieces once per process; a build without GLM-5.3-Flash is a no-op."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        patched_classes = _patch_packed_modules_mapping()
        _patch_replicated_shard_ids()
        _patch_lora_shrink_contiguity()
    except (ModuleNotFoundError, ImportError) as e:
        logger.info(f"Skipping GLM-5.3-Flash LoRA packing patch: {e}")
        return
    _PATCHED = True
    logger.info(
        "Patched vLLM for GLM-5.3-Flash LoRA (vllm#56327): packed_modules_mapping on "
        f"{patched_classes or '<none>'}, replicated_shard_ids honored in merged LoRA-B "
        "loading, non-contiguous LoRA shrink inputs accepted"
    )
