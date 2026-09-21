"""A stand-in for skyrl-capture, so this suite runs without it installed.

`examples/train_integrations/harbor/icap/upstream.py` subclasses capture's
`VLLMTitoProtocol` and registers itself at module scope, so it cannot be
imported unless capture is present. capture is an optional dependency of one
example, not of SkyRL, and the CPU pipeline does not install it.

The stub carries only what the subclass touches, and its `request`/`response`
return sentinels rather than the real wire. That is the point: the wire is
capture's and is tested there, in `tests/test_upstream_vllm.py`. What is left
to check here is what this fork *adds* on top of it -- the path, the name,
lifting `cache_salt` out of the carried fields, and decoding routed experts --
and a sentinel makes "added" distinguishable from "reimplemented".
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

UPSTREAM_MODULE = "examples.train_integrations.harbor.icap.upstream"

#: What the stub's `request` returns, so a test can tell "inherited" from
#: "overridden" without reimplementing the wire.
INHERITED_BODY = {"inherited": True, "sampling_params": {"from": "the base"}}
INHERITED_HEADERS = {"X-Session-ID": "from-the-base"}
#: The same for `response`.
INHERITED_RESULT = {
    "completion_ids": [7, 8, 9],
    "completion_logprobs": [-0.1, -0.2, -0.3],
    "stop_reason": "stop",
    "routed_experts": None,
}


class _StubTokenUpstreamError(Exception):
    """capture's own error type, which a protocol raises to fail a turn."""


class _StubVLLMTitoProtocol:
    """The parts of capture's `VLLMTitoProtocol` a fork inherits or overrides."""

    name = "vllm"
    generate_path = "/inference/v1/generate"

    def url(self, url: str) -> str:
        root = url.rstrip("/")
        return root if root.endswith(self.generate_path) else f"{root}{self.generate_path}"

    def request(self, **kwargs):
        # Records what it was handed, so a test can assert the subclass took
        # `cache_salt` out before the base class filtered the rest.
        self.seen = kwargs
        return dict(INHERITED_BODY), dict(INHERITED_HEADERS)

    def response(self, body):
        self.seen_response = body
        return dict(INHERITED_RESULT)


@pytest.fixture
def inherited():
    return dict(INHERITED_BODY), dict(INHERITED_HEADERS)


@pytest.fixture
def upstream_error():
    """The exception type the stub stands in for."""
    return _StubTokenUpstreamError


@pytest.fixture
def registered():
    """Whatever the module registered on import."""
    recorded: list = []

    def register(protocol):
        recorded.append(protocol)
        return protocol

    tito = types.ModuleType("skyrl_capture.tito")
    upstream = types.ModuleType("skyrl_capture.tito.upstream")
    upstream.register = register
    upstream.VLLMTitoProtocol = _StubVLLMTitoProtocol
    kinds = types.ModuleType("skyrl_capture.tito.types")
    kinds.TokenUpstreamError = _StubTokenUpstreamError

    modules = {
        "skyrl_capture": types.ModuleType("skyrl_capture"),
        "skyrl_capture.tito": tito,
        "skyrl_capture.tito.upstream": upstream,
        "skyrl_capture.tito.types": kinds,
    }

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
