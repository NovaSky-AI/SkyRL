"""Adding a text provider is one class and one register() call.

This is the acceptance criterion for the adapter contract: a provider that did
not exist when the schema was written must work with no migration and no edit
to any file outside its own definition -- not to the proxy, not to a central
parser, not to the SDK, and not to the graph.
"""

from __future__ import annotations

import pytest

from skyrl_capture.upstream import (
    BaseTextProtocol,
    UnknownProtocol,
    registry,
)


class GeminiCompatProtocol(BaseTextProtocol):
    """An OpenAI-compatible endpoint with its own environment variables."""

    name = "gemini-compat"
    environment_map = (
        ("GOOGLE_BASE_URL", "base_url"),
        ("GOOGLE_API_KEY", "api_key"),
    )

    # Inherited unchanged: routes, Bearer auth, /v1 suffix. Only the reading of
    # bodies is borrowed explicitly, because an adapter that says nothing about
    # extraction is an adapter that records nothing.
    def error_body(self, status, message, *, code=None):
        return registry.get("openai").error_body(status, message, code=code)

    def extract(self, **kwargs):
        return registry.get("openai").extract(**kwargs)


@pytest.fixture
def gemini():
    protocol = registry.register(GeminiCompatProtocol())
    yield protocol
    registry._PROTOCOLS.pop(protocol.name, None)


def test_a_new_provider_is_fifteen_lines(gemini):
    assert registry.get("gemini-compat") is gemini
    assert gemini.client_environment("http://x/v1", "k") == {
        "GOOGLE_BASE_URL": "http://x/v1",
        "GOOGLE_API_KEY": "k",
    }
    # Everything else is inherited: OpenAI routes, Bearer auth, /v1 suffix.
    assert gemini.client_suffix == "/v1"
    assert gemini.auth_headers("k") == [(b"authorization", b"Bearer k")]
    assert gemini.endpoint_kind("/v1/chat/completions") == "chat_completions"


async def test_a_new_provider_needs_no_migration(stack_builder, gemini):
    """The provider is a name in a registry and a name in a snapshot, so the
    schema cannot reject it."""

    def upstream(url: str):
        from skyrl_capture.config import TextUpstream

        return TextUpstream(type="gemini-compat", url=f"{url}/v1", api_key="k")

    stack = await stack_builder(upstream_for=upstream)
    created = await stack.create_trajectory()
    assert created["protocol"] == "gemini-compat"
    # The client-facing route and injected variables come from the class.
    assert created["base_url"].endswith("/v1")

    chat = await stack.chat(created, [{"role": "user", "content": "hello"}])
    assert chat.status_code == 200
    graph = await stack.graph(created["id"])
    assert len(graph["nodes"]) == 2, "read back by the adapter this trajectory was given"


def test_an_unregistered_type_is_rejected_at_startup():
    """It used to be caught when a target was created. The upstream is startup
    configuration now, so this is where the check belongs -- and a process that
    cannot forward never starts taking traffic."""
    from skyrl_capture.config import TextUpstream

    with pytest.raises(ValueError, match="expected one of"):
        TextUpstream(type="not-a-provider", url="http://x/v1").validate()


def test_the_registry_raises_rather_than_defaulting():
    """The silent fallback to OpenAI was safe with a closed enum; it is not now.

    An unregistered Anthropic upstream that fell back would get Bearer auth and
    OpenAI extraction instead of an error.
    """
    with pytest.raises(UnknownProtocol) as caught:
        registry.get("anthropic-v2")
    assert "expected one of" in str(caught.value)


def test_the_two_built_in_protocols_differ_where_the_providers_do():
    """Five differences, all of them inside the two adapters."""
    import orjson

    openai = registry.get("openai")
    anthropic = registry.get("anthropic")

    assert openai.client_suffix == "/v1"
    # The Anthropic SDK appends /v1 itself, so the route must not.
    assert anthropic.client_suffix == ""
    assert openai.auth_headers("k") == [(b"authorization", b"Bearer k")]
    assert anthropic.auth_headers("k") == [(b"x-api-key", b"k")]
    assert "ANTHROPIC_API_KEY" in anthropic.client_environment("http://x")
    assert anthropic.endpoint_kind("/v1/messages") == "messages"

    openai_error = orjson.loads(openai.error_body(401, "nope"))
    anthropic_error = orjson.loads(anthropic.error_body(401, "nope"))
    assert openai_error["error"]["type"] == "invalid_request_error"
    assert anthropic_error["type"] == "error"
    assert anthropic_error["error"]["type"] == "authentication_error"


def test_a_token_engine_is_not_a_text_protocol():
    """The two registries are separate so neither plugin can see the other.

    `tokens` names an engine wire, not something a client speaks, so asking
    the text registry for it is an error rather than a half-working adapter.
    """
    from skyrl_capture.tito import upstream as tito

    assert not registry.is_registered("tokens")
    assert tito.is_registered("tokens") and tito.is_registered("vllm")
    with pytest.raises(UnknownProtocol):
        registry.get("vllm")


# -- CLI error handling -----------------------------------------------------
def test_an_unreachable_control_plane_is_one_line_not_a_traceback():
    """The most common way to use the tool wrong must not look like a defect."""
    import pytest as _pytest

    from skyrl_capture.sdk import CaptureClient, CaptureError

    client = CaptureClient("http://127.0.0.1:9", timeout=1.0)
    with _pytest.raises(CaptureError) as caught:
        client.list_trajectories()
    message = str(caught.value)
    assert "cannot reach the control plane" in message
    assert "http://127.0.0.1:9" in message
    assert "skyrl-capture serve" in message, "the message says what to do about it"
    client.close()
