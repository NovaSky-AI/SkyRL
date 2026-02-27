"""Tokenization related utilities"""

from transformers import AutoTokenizer


def get_tokenizer(model_name_or_path, **tokenizer_kwargs) -> AutoTokenizer:
    """Gets tokenizer for the given base model with the given parameters

    Sets the pad token ID to EOS token ID if `None`"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
