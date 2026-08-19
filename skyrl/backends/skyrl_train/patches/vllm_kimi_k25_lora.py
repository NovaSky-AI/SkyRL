"""Enable LoRA serving for Kimi K2.5-family checkpoints in vLLM.

vLLM's DeepSeek-V2/V3 decoder (the language model inside
``KimiK25ForConditionalGeneration`` checkpoints such as Kimi-K2.7-Code)
declares ``SupportsLoRA``, but the multimodal wrapper does not, so
``--enable-lora`` fails the worker-side ``supports_lora()`` gate. SkyRL trains
these checkpoints text-only (``language_model_only``) with Megatron LoRA
adapters hot-loaded into vLLM (``merge_lora=false`` disk sync), which needs the
wrapper to be LoRA-capable.

This mirrors how Qwen3-VL wrappers declare LoRA in vLLM: the interface
attributes (``supports_lora`` / ``packed_modules_mapping`` /
``embedding_modules``) plus ``get_mm_mapping`` so the LoRA manager resolves
packed submodules and skips the vision tower / projector.

Applied at import time of ``new_inference_worker_wrap`` (the SkyRL worker
extension), which every vLLM worker process imports before loading the model.
Remove once vLLM adds ``SupportsLoRA`` to the class upstream (the patch then
no-ops).
"""

from __future__ import annotations


def apply_kimi_k25_lora_patch() -> None:
    """Declare ``KimiK25ForConditionalGeneration`` LoRA-capable. Idempotent."""
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
        from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration
        from vllm.model_executor.models.module_mapping import MultiModelKeys
    except ImportError:
        # vLLM absent (CPU-only env) or too old to have the model: nothing to do.
        return

    cls = KimiK25ForConditionalGeneration
    if getattr(cls, "supports_lora", False):
        return  # native support (or already patched)

    # The full ``SupportsLoRA`` member set: the worker-side gate is a
    # runtime-checkable Protocol isinstance() on the model *instance*, which
    # requires every protocol member to exist (not just the three documented
    # attributes). Values mirror the inner DeepSeek decoder's declarations.
    cls.supports_lora = True
    # Share the inner decoder's mapping *object*: DeepseekV2ForCausalLM.__init__
    # adds "fused_qkv_a_proj" to its class-level dict at model build time (for
    # q_lora_rank models like Kimi K2.x), and the LoRA manager reads the mapping
    # off the outer model -- sharing keeps the two views consistent.
    cls.packed_modules_mapping = DeepseekV2ForCausalLM.packed_modules_mapping
    cls.embedding_modules = {}
    cls.lora_skip_prefixes = []
    # DeepSeek decoders export per-expert LoRA keys (experts.<idx>.<proj>), not
    # the flat 3D-MoE layout, and use gated MLPs.
    cls.is_3d_moe_weight = False
    cls.is_non_gated_moe = False
    # vLLM 0.26 added an annotation-only `lora_manager` member to the
    # SupportsLoRA protocol. Nominal subclasses (native models) skip the
    # structural isinstance() check, but this patched class is checked
    # structurally, so every protocol member must *exist* on the class —
    # including annotation-only ones. The worker's LoRA mixin assigns the
    # real manager at runtime; None is the documented "not yet set" state.
    cls.lora_manager = None

    def get_mm_mapping(self) -> MultiModelKeys:
        """Multimodal module split so LoRA skips the tower/connector."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="mm_projector",
            tower_model="vision_tower",
        )

    cls.get_mm_mapping = get_mm_mapping
