"""Stand-ins for inference-capture, so this suite runs without it installed.

`examples/train_integrations/harbor/icap/upstream.py` subclasses capture's
`TokensServer` and registers itself at module scope, so it cannot be imported
unless capture is present. capture is an optional dependency of that example,
not of SkyRL, and the CPU pipeline does not install it.

Stubbing rather than skipping is deliberate. What the wire does -- build a
request, read a response back -- is pure data transformation that never touches
the base class, so the coverage is the same and it runs everywhere. The one
thing a stub cannot prove is that the subclass still fits capture's real
interface; a Harbor run is what shows that, and the fields asserted here are
the contract it has to keep.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

UPSTREAM_MODULE = "examples.train_integrations.harbor.icap.upstream"


class _StubTokensServer:
    """The parts of capture's `TokensServer` the wire inherits or overrides."""

    name = "tokens"
    mode = "tokens"
    client_suffix = "/v1"
    requires_tokenizer = True

    def token_url(self, url: str) -> str:
        return url

    def describe(self) -> dict:
        return {"name": self.name, "mode": self.mode}


class _StubTokenUpstreamError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@pytest.fixture
def registered():
    """Whatever the module registered on import, and the class itself."""
    recorded: list = []

    def register(server):
        recorded.append(server)
        return server

    modules = {
        "inference_capture": types.ModuleType("inference_capture"),
        "inference_capture.tokens": types.ModuleType("inference_capture.tokens"),
        "inference_capture.tokens.types": types.ModuleType("inference_capture.tokens.types"),
        "inference_capture.upstream": types.ModuleType("inference_capture.upstream"),
        "inference_capture.upstream.registry": types.ModuleType(
            "inference_capture.upstream.registry"
        ),
        "inference_capture.upstream.servers": types.ModuleType(
            "inference_capture.upstream.servers"
        ),
    }
    modules["inference_capture.tokens.types"].TokenUpstreamError = _StubTokenUpstreamError
    modules["inference_capture.upstream.registry"].register = register
    modules["inference_capture.upstream.servers"].TokensServer = _StubTokensServer

    saved = {name: sys.modules.get(name) for name in modules}
    saved[UPSTREAM_MODULE] = sys.modules.get(UPSTREAM_MODULE)
    sys.modules.update(modules)
    sys.modules.pop(UPSTREAM_MODULE, None)
    try:
        module = importlib.import_module(UPSTREAM_MODULE)
        # Importing is what registers; that is the contract with capture.
        assert recorded, "the module did not register anything on import"
        yield recorded[0]
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def wire_error():
    return _StubTokenUpstreamError
