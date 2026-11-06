"""Fused MoE-LoRA: use vLLM's two-kernel path (shrink, then expand) instead of the one-shot kernel.

vLLM 0.32's ``_fused_moe_lora`` runs a single fused shrink+expand kernel for ranks <= 128. At DeepSeek-V4.1-Flash's
shapes (384 experts, top-6, rank 32, TP8 intermediate 384) it launches ~one program per (expert block, slice) and each
loops serially over K=5120 or N=5120, so it is latency-bound: with tuned tiles the two-kernel path (split shrink and
expand, both Triton) is 1.2-3.7x faster per call on B200 (CUDA-graph timing; 640 tokens: W13 153 -> 62 us, W2 171 ->
46 us). The tiles come from the recipe's VLLM_TUNED_CONFIG_FOLDER files (vllm_tuned_configs/, tuned by
a tuning sweep); the one-shot kernel ignores everything but block_m.

Opt-in: ``SKYRL_VLLM_MOE_LORA_TWO_KERNEL=1`` (applied in every vLLM worker by ``new_inference_worker_wrap``).
"""

import os

from loguru import logger

ENV = "SKYRL_VLLM_MOE_LORA_TWO_KERNEL"
_APPLIED = False


def apply_moe_lora_two_kernel_patch() -> bool:
    """Set vLLM's one-shot rank cutoff to 0 so every MoE-LoRA call takes the two-kernel path. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get(ENV, "0").lower() not in ("1", "true"):
        return False
    try:
        import vllm.lora.ops.triton_ops.fused_moe_lora_op as fused_moe_lora_op
    except ImportError:
        return False
    if not hasattr(fused_moe_lora_op, "_FUSED_MOE_LORA_ONE_SHOT_MAX_RANK"):
        logger.warning(f"{ENV}=1 but vLLM has no one-shot MoE-LoRA cutoff; nothing to change")
        return False
    fused_moe_lora_op._FUSED_MOE_LORA_ONE_SHOT_MAX_RANK = 0
    _APPLIED = True
    logger.info("Fused MoE-LoRA: two-kernel (shrink + expand) path instead of the one-shot kernel")
    return True
