"""Fused MoE-LoRA: 64-bit token offsets in vLLM's two-kernel (shrink + expand) path.

``_fused_moe_lora_kernel`` loads the sorted (token, top-k) ids as int32 and writes its output at
``stride_cm * offs_token``. That product is computed in int32. It overflows once
``num_tokens * top_k * N`` exceeds 2**31. Under DP attention + expert parallelism the MoE layer sees the
tokens of every DP rank (naive DP/EP all-gather). DeepSeek-V4.1-Flash at TP1 has the full intermediate, so
the W13 LoRA output is (tokens, 6, 2 * 2304). With DP8 and 32768 tokens per rank that is
262144 * 6 * 4608 = 7.2e9 elements. The expand kernel then writes out of bounds, and vLLM's profile_run
crashes with an illegal memory access. (TP8 only reaches 32768 * 6 * 2 * 288.)

The kernel picks up ``_get_token_offs`` from the module globals when Triton compiles it. This patch swaps
in an identical helper that returns int64, so every offset built from it (output, activation rows,
top-k weights) is 64-bit. Apply it before the first MoE-LoRA call. vLLM's one-shot kernel has the same
int32 product, so pair this patch with ``SKYRL_VLLM_MOE_LORA_TWO_KERNEL=1``.

Opt-in: ``SKYRL_VLLM_MOE_LORA_INT64_OFFSETS=1`` (applied in every vLLM worker by ``new_inference_worker_wrap``).
"""

import os

from loguru import logger

ENV = "SKYRL_VLLM_MOE_LORA_INT64_OFFSETS"
_APPLIED = False


def apply_moe_lora_int64_offsets_patch() -> bool:
    """Swap vLLM's ``_get_token_offs`` Triton helper for an int64 version. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get(ENV, "0").lower() not in ("1", "true"):
        return False
    try:
        import vllm.lora.ops.triton_ops.fused_moe_lora_op as op
        from vllm.triton_utils import tl, triton
    except ImportError:
        return False
    if not hasattr(op, "_get_token_offs"):
        logger.warning(f"{ENV}=1 but vLLM has no _get_token_offs helper; nothing to change")
        return False

    @triton.jit
    def _get_token_offs(
        sorted_token_ids_ptr,
        lora_id,
        pid_m,
        offs,
        stride_tl,
        max_loras,
        num_valid_tokens,
        naive_block_assignment: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
    ):
        if naive_block_assignment:
            return tl.where(offs == 0, pid_m, num_valid_tokens).to(tl.int64)
        else:
            offs_token_id = pid_m * BLOCK_SIZE_M + offs
            token_ind = stride_tl * lora_id + offs_token_id
            return tl.load(sorted_token_ids_ptr + token_ind, token_ind < max_loras * stride_tl, 0).to(tl.int64)

    op._get_token_offs = _get_token_offs
    if getattr(op, "_FUSED_MOE_LORA_ONE_SHOT_MAX_RANK", 0) > 0:
        logger.warning(
            f"{ENV}=1 fixes the two-kernel path only; the one-shot kernel is still int32 "
            "(set SKYRL_VLLM_MOE_LORA_TWO_KERNEL=1 for DP attention + EP)"
        )
    _APPLIED = True
    logger.info("Fused MoE-LoRA: int64 token offsets in the two-kernel path")
    return True
