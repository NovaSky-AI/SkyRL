"""Name -> `TextProtocol`. The only thing stored anywhere is the name.

``get`` **raises** on an unknown name rather than falling back. With a closed
enum a silent default to OpenAI was harmless; with an open set it is not -- an
unregistered Anthropic upstream would get Bearer auth and OpenAI extraction
instead of an error.

Token-in/token-out engines register somewhere else, in
`tito/upstream.py`. Two registries rather than one flag on a shared base
class: a text plugin then cannot see token rendering, and a token plugin
cannot see HTTP forwarding.
"""

from __future__ import annotations

from skyrl_capture.upstream.text import TextProtocol


class UnknownProtocol(Exception):
    """No text protocol is registered under this name."""


_PROTOCOLS: dict[str, TextProtocol] = {}


def register(protocol: TextProtocol) -> TextProtocol:
    _PROTOCOLS[protocol.name] = protocol
    return protocol


def get(name: str) -> TextProtocol:
    try:
        return _PROTOCOLS[name]
    except KeyError:
        raise UnknownProtocol(
            f"unsupported upstream type {name!r}; expected one of {', '.join(names())}"
        ) from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_PROTOCOLS))


def is_registered(name: str) -> bool:
    return name in _PROTOCOLS
