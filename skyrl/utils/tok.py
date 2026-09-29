from typing import Optional

"""Tokenization related utilities"""

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


def get_tokenizer(model_name_or_path, **tokenizer_kwargs) -> AutoTokenizer:
    """Gets tokenizer for the given base model with the given parameters

    Sets the pad token ID to EOS token ID if `None`"""
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    except NotImplementedError:
        # Some repos (e.g. zai-org/GLM-4.7-Flash) declare
        # tokenizer_class="PreTrainedTokenizer" (the abstract slow base) in
        # tokenizer_config.json. Under transformers>=5, PreTrainedTokenizer.__init__
        # eagerly calls get_vocab() and crashes. Fall back to the fast tokenizer.
        tokenizer_kwargs.pop("use_fast", None)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# Module-name pattern for the vision tower and projector of the HF VLM families SkyRL loads
# (Qwen2.5/3-VL: ``visual``; LLaVA-style: ``vision_tower`` / ``multi_modal_projector``; Gemma / others:
# ``vision_model``). PEFT accepts a regex string for ``exclude_modules``.
VISION_TOWER_MODULE_REGEX = r".*(visual|vision_tower|vision_model|multi_modal_projector|mm_projector).*"


def lora_exclude_modules_for_model(is_vlm: bool, configured: Optional[str]) -> Optional[str]:
    """LoRA ``exclude_modules`` to use: the configured value, or the vision-tower pattern for a VLM.

    vLLM applies LoRA only to the language model of a multimodal model and silently drops adapter
    tensors on the vision tower, so training them makes the trainer's policy diverge from the rollout
    policy. Excluding the tower by default keeps the two identical.
    """
    if configured is not None:
        return configured
    return VISION_TOWER_MODULE_REGEX if is_vlm else None


def check_is_vlm(model_config_or_path) -> bool:
    """Returns True if the model config declares a non-null ``vision_config``.

    Accepts either an already-loaded ``PretrainedConfig`` or a model name/path.
    Passing the config avoids a redundant ``AutoConfig.from_pretrained`` round-trip
    when the caller already holds it."""
    if isinstance(model_config_or_path, str):
        model_config = AutoConfig.from_pretrained(model_config_or_path, trust_remote_code=True)
    else:
        model_config = model_config_or_path
    return hasattr(model_config, "vision_config") and getattr(model_config, "vision_config") is not None


def get_processor(model_name_or_path, **tokenizer_kwargs) -> AutoProcessor:
    """Gets processor for the given base model with the given parameters

    Sets the pad token ID to EOS token ID if `None`"""
    tokenizer_kwargs.setdefault("trust_remote_code", True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor
