"""Tokenization related utilities"""

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)


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
