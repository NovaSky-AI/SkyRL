"""Load fused (3D) MoE LoRA adapters into vLLM's Qwen4-Exp (Qwen3.8-Flash-Next) model.

Qwen4-Exp stores its 512 routed experts as the fused ``experts.gate_up_proj`` /
``experts.down_proj`` tensors, exactly like Qwen3.5-MoE, and Megatron-Bridge exports the expert
LoRA in the matching flat 3D PEFT layout (``experts.base_layer`` / ``experts``, see
``_convert_moe_experts_lora_to_vllm``). vLLM 0.29.0 marks ``Qwen3_5MoeForConditionalGeneration``
with ``is_3d_moe_weight = True`` so its LoRA manager builds ``FusedMoE3DWithLoRA`` for that
layout, but ``Qwen4ExpForCausalLM`` / ``Qwen4ExpForConditionalGeneration`` (which derive from the
*dense* Qwen3.5 VL wrapper) do not set the flag. The manager then wraps the experts with the 2D
per-expert ``FusedMoEWithLoRA``, whose ``set_lora`` asserts on the 3D tensors
(``assert isinstance(lora_a, list)``) at every weight sync.

Applied at import time of ``new_inference_worker_wrap`` (the SkyRL worker extension), which every
vLLM worker process imports before loading the model. Remove once vLLM declares the flag upstream
(the patch then no-ops).
"""

from __future__ import annotations


def apply_qwen4_exp_lora_patch() -> None:
    """Declare vLLM's Qwen4-Exp classes as fused-3D-MoE for LoRA loading. Idempotent."""
    try:
        from vllm.models.qwen4_exp.nvidia.model import (
            Qwen4ExpForCausalLM,
            Qwen4ExpForConditionalGeneration,
        )
    except ImportError:
        # vLLM absent (CPU-only env) or too old to have the model: nothing to do.
        return

    for cls in (Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration):
        if not getattr(cls, "is_3d_moe_weight", False):
            cls.is_3d_moe_weight = True
