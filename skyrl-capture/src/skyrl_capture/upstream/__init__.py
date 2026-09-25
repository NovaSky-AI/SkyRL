"""Text protocols: one adapter per provider, resolved once at startup.

Everything provider-specific about text capture lives behind `TextProtocol` --
routes, client environment, credential header, error envelope, and reading a
request/response pair back into messages. The proxy forwards bytes without
consulting it about them; the commit path asks it, afterwards, what those
bytes meant. So a provider's schema never reaches the request path, and adding
one is an adapter plus fixtures rather than edits spread across the proxy, a
parser, the SDK and the graph.

Importing this package registers the protocols capture ships: OpenAI (Chat
Completions and Responses) and Anthropic (Messages). An OpenAI-compatible
server -- vLLM, SGLang -- is the OpenAI adapter with a different URL and needs
nothing here. A provider with a genuinely different HTTP schema is a plugin;
see `plugins.py`.
"""

from skyrl_capture.upstream.anthropic import AnthropicProtocol
from skyrl_capture.upstream.openai import OpenAIProtocol
from skyrl_capture.upstream.registry import (
    UnknownProtocol,
    get,
    is_registered,
    names,
    register,
)
from skyrl_capture.upstream.text import (
    PLACEHOLDER_KEY,
    BaseTextProtocol,
    TextProtocol,
)

register(OpenAIProtocol())
register(AnthropicProtocol())

__all__ = [
    "PLACEHOLDER_KEY",
    "AnthropicProtocol",
    "BaseTextProtocol",
    "OpenAIProtocol",
    "TextProtocol",
    "UnknownProtocol",
    "get",
    "is_registered",
    "names",
    "register",
]
