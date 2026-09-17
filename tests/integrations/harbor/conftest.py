"""A stand-in for skyrl-capture, so this suite runs without it installed.

`examples/train_integrations/harbor/icap/upstream.py` subclasses capture's
`VLLMTokensServer` and registers itself at module scope, so it cannot be
imported unless capture is present. capture is an optional dependency of one
example, not of SkyRL, and the CPU pipeline does not install it.

The stub carries only what the subclass touches: `generate_path` and the
`token_url` built from it. The wire itself -- the request shape, the response
shape, the session header -- belongs to capture and is tested there, in
`tests/test_upstream_vllm.py`. What is left to check here is that this fork
declares the right name and path and overrides nothing else.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

UPSTREAM_MODULE = "examples.train_integrations.harbor.icap.upstream"

#: Returned by the stub's `token_request`, so a test can tell "inherited" from
#: "overridden" without reimplementing the wire.
INHERITED = ({"inherited": True}, {"X-Session-ID": "from-the-base"})


class _StubVLLMTokensServer:
    """The parts of capture's `VLLMTokensServer` a fork inherits or overrides."""

    name = "vllm"
    mode = "tokens"
    client_suffix = "/v1"
    requires_tokenizer = True
    generate_path = "/inference/v1/generate"

    def token_url(self, url: str) -> str:
        root = url.rstrip("/")
        return root if root.endswith(self.generate_path) else f"{root}{self.generate_path}"

    def token_request(self, **_kwargs):
        return INHERITED

    def describe(self) -> dict:
        return {"name": self.name, "mode": self.mode}


@pytest.fixture
def inherited():
    """What the stub's `token_request` returns, so a test can tell "inherited"
    from "overridden" without reimplementing the wire."""
    return INHERITED


@pytest.fixture
def registered():
    """Whatever the module registered on import."""
    recorded: list = []

    def register(server):
        recorded.append(server)
        return server

    modules = {
        "skyrl_capture": types.ModuleType("skyrl_capture"),
        "skyrl_capture.upstream": types.ModuleType("skyrl_capture.upstream"),
        "skyrl_capture.upstream.registry": types.ModuleType("skyrl_capture.upstream.registry"),
        "skyrl_capture.upstream.servers": types.ModuleType("skyrl_capture.upstream.servers"),
    }
    modules["skyrl_capture.upstream.registry"].register = register
    modules["skyrl_capture.upstream.servers"].VLLMTokensServer = _StubVLLMTokensServer

    saved = {name: sys.modules.get(name) for name in modules}
    saved[UPSTREAM_MODULE] = sys.modules.get(UPSTREAM_MODULE)
    sys.modules.update(modules)
    sys.modules.pop(UPSTREAM_MODULE, None)
    try:
        importlib.import_module(UPSTREAM_MODULE)
        # Importing is what registers; that is the contract with capture.
        assert recorded, "the module did not register anything on import"
        yield recorded[0]
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
