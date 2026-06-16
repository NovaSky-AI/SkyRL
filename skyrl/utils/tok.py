"""Tokenization related utilities"""

from transformers import AutoTokenizer, PreTrainedTokenizerFast


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
