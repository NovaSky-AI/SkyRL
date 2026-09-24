"""Propagate the language model's attention backend to the Qwen3-VL vision encoder.

Megatron-Bridge's ``get_vision_model_config`` builds the ViT ``TransformerConfig``
from scratch and never copies ``attention_backend`` from the language config, so
the vision encoder always gets the default ``AttnBackend.auto``.

megatron-core now calls ``set_attention_backend`` from ``VisionModule.__init__``
too (previously only ``LanguageModule`` did), and it asserts that the
``NVTE_{FLASH,FUSED,UNFUSED}_ATTN`` env vars are consistent across every model
built in the process. ``auto`` requires all three to be 1, which conflicts with
the language model's ``flash``/``fused`` backend (and with ``NVTE_FUSED_ATTN=0``
exported for ``trainer.flash_attn``), so every Qwen3-VL / Qwen3.5 model build
fails with ``NVTE_FUSED_ATTN is set to 0, but attention_backend='auto' ...``.

TE reads those env vars process-wide, so the ViT already ran with the language
model's backend before; copying it onto the vision config keeps that behavior.

``get_vision_model_config`` is imported by name into the Qwen3-VL model and the
Qwen3.5-VL provider modules, so the wrapper is rebound in each of them.
"""

from loguru import logger

_APPLIED = False


def patch_vision_attention_backend() -> None:
    """Wrap Bridge's ``get_vision_model_config`` to inherit the LM attention backend."""
    global _APPLIED
    if _APPLIED:
        return

    try:
        from megatron.bridge.models.qwen_vl import qwen35_vl_provider
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import (
            model as qwen3_vl_model,
        )
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import (
            transformer_config as qwen3_vl_transformer_config,
        )
    except ImportError as e:
        logger.warning(f"Qwen3-VL Bridge modules unavailable; skipping vision attention backend patch: {e}")
        return

    orig_get_vision_model_config = qwen3_vl_transformer_config.get_vision_model_config

    def patched_get_vision_model_config(hf_config, megatron_config=None):
        config = orig_get_vision_model_config(hf_config, megatron_config=megatron_config)
        if megatron_config is not None:
            config.attention_backend = megatron_config.attention_backend
            config.flash_attention_version = getattr(megatron_config, "flash_attention_version", None)
        return config

    for module in (qwen3_vl_transformer_config, qwen3_vl_model, qwen35_vl_provider):
        if getattr(module, "get_vision_model_config", None) is orig_get_vision_model_config:
            module.get_vision_model_config = patched_get_vision_model_config

    _APPLIED = True
    logger.info("Applied Qwen3-VL vision encoder attention backend propagation patch")
