def apply_chat_template(tokenizer, messages, *, chat_template_kwargs=None, **kwargs):
    """Apply a tokenizer chat template with per-call controls taking precedence."""
    apply_kwargs = dict(chat_template_kwargs or {})
    apply_kwargs.update(kwargs)
    return tokenizer.apply_chat_template(messages, **apply_kwargs)
