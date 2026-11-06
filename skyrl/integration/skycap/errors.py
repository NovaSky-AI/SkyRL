"""How a harness can tell that its agent stopped because skycap refused a prompt that didn't fit."""

import re

#: skycap answers a prompt longer than the model's context with a 400 whose code is ``context_length_exceeded``;
#: LiteLLM may raise it as ``ContextWindowExceededError``. Rich can wrap a logged code, so whitespace may split it.
CONTEXT_LENGTH_OUTPUT = re.compile(r"context\s*_length\s*_exceeded|ContextWindowExceededError")


def is_context_length_message(text: str) -> bool:
    """Whether ``text`` (an exception message, an agent's log) shows a context-length error."""
    return CONTEXT_LENGTH_OUTPUT.search(text) is not None
