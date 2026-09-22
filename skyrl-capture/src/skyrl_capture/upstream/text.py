"""The text-protocol contract, and what every implementation shares.

One adapter owns everything provider-specific about one text protocol: where
its routes live, which environment variables an unchanged client reads, which
header carries a credential, what its errors look like, and -- after the
response has been forwarded -- how to read a request/response pair back into
messages.

**Nothing on the forward path consults it about the body.** The proxy sends the
bytes it received, so an unknown field or a feature the provider shipped last
week passes through untouched; the adapter sees those bytes only afterwards,
where a parse failure is recorded as a derivation error and cannot reach the
client. That is also why the adapters here read JSON shallowly with orjson
rather than validating against a provider SDK's models: a model that rejects an
unknown field would turn a compatible request into a failed one, and would tie
forwarding compatibility to an SDK release. An adapter that wants an official
SDK's parser may use one internally -- that is what the contract makes an
implementation detail.

What an adapter supplies is listed on `TextProtocol` and nothing more. It does
not implement forwarding, recording, graph construction or lifecycle, and it
cannot alter message identity: normalizing and hashing are the domain's, in
`domain/extraction.py`, applied the same way to every provider.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import orjson

from skyrl_capture.domain.extraction import ExtractedExchange, ExtractedMessage  # noqa: F401

#: Route suffix -> endpoint kind. Shared because classification is path-based
#: for every protocol understood today; an adapter with novel routes overrides
#: `endpoint_kind`.
KIND_BY_SUFFIX = (
    ("/chat/completions", "chat_completions"),
    ("/responses", "responses"),
    ("/messages", "messages"),
    ("/completions", "completions"),
    ("/embeddings", "embeddings"),
    ("/models", "models"),
)

#: What an SDK gets when a deployment has no ingress credential of its own.
#: Capture ignores it; it exists because a client library insists on a value.
PLACEHOLDER_KEY = "unused"


@runtime_checkable
class TextProtocol(Protocol):
    name: str
    #: Appended to the client-facing trajectory route, e.g. ``/v1`` or ``""``.
    client_suffix: str

    def client_environment(self, base_url: str, api_key: str = PLACEHOLDER_KEY) -> dict[str, str]: ...

    def upstream_url(self, base_url: str, suffix: str, query: str) -> str: ...

    def auth_headers(self, api_key: str) -> list[tuple[bytes, bytes]]: ...

    def error_body(self, status: int, message: str, *, code: str | None = None) -> bytes: ...

    def endpoint_kind(self, suffix: str) -> str: ...

    def extract(
        self,
        *,
        endpoint_kind: str,
        request_body: bytes,
        response_body: bytes,
        streaming: bool,
        status: int | None,
    ) -> ExtractedExchange: ...


class BaseTextProtocol:
    """An OpenAI-shaped adapter. Subclass to add a provider.

    A provider that speaks OpenAI's protocol over a different base URL needs
    nothing but a name -- that is what an OpenAI-compatible vLLM or SGLang
    server is, and it is why neither has a class here. One with different
    environment variables or a different credential header overrides those two
    methods; one with a genuinely different body shape overrides `extract`.
    """

    name = "base"
    client_suffix = "/v1"

    #: ``(variable, field)`` pairs, where field is ``base_url`` or ``api_key``.
    #: Both the modern and legacy OpenAI names are set, because SDK versions in
    #: the wild differ on which they read.
    environment_map: tuple[tuple[str, str], ...] = (
        ("OPENAI_BASE_URL", "base_url"),
        ("OPENAI_API_BASE", "base_url"),
        ("OPENAI_API_KEY", "api_key"),
    )

    def client_environment(self, base_url: str, api_key: str = PLACEHOLDER_KEY) -> dict[str, str]:
        """The variables that point an unchanged client at a capture route.

        ``api_key`` is a **placeholder**, not a credential: capture
        authenticates nothing on the way in, and most provider SDKs refuse to
        construct a client without some value in the variable. A deployment
        that authenticates its ingress passes its own here.
        """
        values = {"base_url": base_url, "api_key": api_key}
        return {name: values[field] for name, field in self.environment_map}

    def upstream_url(self, base_url: str, suffix: str, query: str) -> str:
        """Map a captured route path onto the upstream's URL.

        OpenAI-style bases already end in ``/v1``, so the duplicate segment is
        collapsed rather than producing ``/v1/v1/chat/completions``.
        """
        root = base_url.rstrip("/")
        path = suffix if suffix.startswith("/") else f"/{suffix}"
        if root.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        url = f"{root}{path}"
        return f"{url}?{query}" if query else url

    def auth_headers(self, api_key: str) -> list[tuple[bytes, bytes]]:
        return [(b"authorization", b"Bearer " + api_key.encode())]

    def endpoint_kind(self, suffix: str) -> str:
        tail = suffix.rsplit("?", 1)[0].rstrip("/")
        for candidate, kind in KIND_BY_SUFFIX:
            if tail.endswith(candidate):
                return kind
        return "other"

    def error_body(self, status: int, message: str, *, code: str | None = None) -> bytes:
        raise NotImplementedError

    def extract(
        self,
        *,
        endpoint_kind: str,
        request_body: bytes,
        response_body: bytes,
        streaming: bool,
        status: int | None,
    ) -> ExtractedExchange:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "client_suffix": self.client_suffix}


# -- server-sent events -------------------------------------------------------
#
# The framing is HTTP's rather than any provider's, so it is shared here; what
# the frames *mean* is each adapter's.
def iter_events(body: bytes) -> list[tuple[str | None, str]]:
    """Split an SSE byte stream into ``(event_name, data)`` pairs.

    Tolerant by design: a truncated final frame -- a client disconnect, an
    upstream failure mid-stream -- yields whatever complete frames arrived
    rather than failing the parse.
    """
    events: list[tuple[str | None, str]] = []
    text = body.decode("utf-8", errors="replace")
    for block in text.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith(":"):
                continue
        if data_lines:
            events.append((name, "\n".join(data_lines)))
    return events


def decode_frames(body: bytes) -> list[dict[str, Any]]:
    """Every frame in a stream, decoded where it is JSON and kept where it is not."""
    frames: list[dict[str, Any]] = []
    for name, data in iter_events(body):
        if data == "[DONE]":
            frames.append({"event": name, "done": True})
            continue
        try:
            frames.append({"event": name, "data": orjson.loads(data)})
        except orjson.JSONDecodeError:
            frames.append({"event": name, "raw": data})
    return frames
