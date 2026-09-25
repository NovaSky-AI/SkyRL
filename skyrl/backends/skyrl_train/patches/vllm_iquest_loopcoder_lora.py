"""Enable vLLM LoRA serving for IQuest LoopCoder checkpoints.

vLLM 0.28 ships a native ``IQuestLoopCoderForCausalLM`` implementation, but
the class does not declare the ``SupportsLoRA`` interface. LoopCoder shares its
physical attention and MLP projections across recurrent passes, so the usual
adapter layout is already correct: one adapter per physical projection is
applied every time that projection is executed.

The loop gate is intentionally not adapter-enabled. Hugging Face stores it as
a raw per-head parameter rather than a linear module, and SkyRL's Megatron path
keeps it frozen as part of the base model.
"""

from __future__ import annotations


def apply_iquest_loopcoder_lora_patch() -> None:
    """Declare vLLM's native LoopCoder implementation LoRA-capable."""
    try:
        from vllm.model_executor.models.iquest_loopcoder import (
            IQuestLoopCoderForCausalLM,
        )
    except ImportError:
        return

    cls = IQuestLoopCoderForCausalLM
    if getattr(cls, "supports_lora", False):
        return

    cls.supports_lora = True
    cls.packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    cls.embedding_modules = {}
    cls.lora_skip_prefixes = []
    cls.is_3d_moe_weight = False
    cls.is_non_gated_moe = False
    cls.lora_manager = None
