"""What every API dialect provides, and the plain data that passes between a dialect and a backend.

A dialect is one wire format: how its request becomes skycap's turns (the
chat-template message dicts the graph stores and the renderer reads), and how
one sampled turn goes back on the wire. Backends and the graph never see the
wire; they see ``CallRequest`` and ``Turn``.

Dialects are stateless, one instance per ``DialectType``. The enum is what is
passed around and recorded (``CallInfo.dialect``); ``get_dialect`` gives the
instance. Per-call state lives in ``CallRequest`` and ``Turn``, per-stream state
in an ``Assembler``.
"""

from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import orjson
from aiohttp import web


class DialectType(Enum):
    #: The value is what a record stores in ``calls[].dialect``.
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"

    def get_dialect(self) -> Dialect:
        """The one stateless instance of this type."""
        return _dialects()[self]


@functools.cache
def _dialects() -> dict[DialectType, Dialect]:
    # Imported here: the implementations import this module.
    from skycap.dialects.openai_chat import ChatDialect
    from skycap.dialects.openai_responses import ResponsesDialect

    return {dialect.type: dialect for dialect in (ChatDialect(), ResponsesDialect())}


#: Request fields recorded on a call as its sampling settings, in the chat spelling every dialect maps to.
SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)


class RequestError(ValueError):
    """A request capture refuses before forwarding it."""


def error_body(message: str, *, kind: str = "invalid_request_error", code: str | None = None) -> bytes:
    """OpenAI's error object, which both OpenAI dialects answer errors with."""
    return orjson.dumps({"error": {"message": message, "type": kind, "param": None, "code": code}})


@dataclass(slots=True)
class CallRequest:
    """One harness call, read by its dialect."""

    #: The conversation as chat-template turns: what the graph stores and the renderer reads.
    messages: list[dict[str, Any]]
    #: Function tools, in the chat-completions spelling.
    tools: list[dict[str, Any]] | None
    model: str | None
    stream: bool
    #: Spelled as ``SAMPLING_KEYS``.
    sampling: dict[str, Any]
    #: The request body as sent.
    body: dict[str, Any]
    dialect: DialectType = DialectType.OPENAI_CHAT


@dataclass(slots=True)
class Turn:
    """One sampled turn, before it goes on the wire (token mode)."""

    #: The parsed reply, or the raw-content one.
    message: dict[str, Any]
    #: ``stop``, ``length`` or ``tool_calls``.
    finish_reason: str
    #: The model name the reply reports.
    model: str | None
    prompt_tokens: int
    completion_tokens: int
    #: A digest of the prompt length and the sampled ids, for ids a client sends back.
    key: str
    #: The server's ``use_raw_content``.
    raw_content: bool = False


@dataclass(slots=True)
class Reply:
    """A reply read from an upstream (text mode)."""

    message: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


class Assembler(ABC):
    """Rebuilds the reply an upstream's SSE stream carried, one event at a time.

    Subclasses keep the unread bytes in ``_buffer``.
    """

    __slots__ = ()
    _buffer: bytes

    def feed(self, chunk: bytes) -> list[tuple[bytes, bool]]:
        """Read ``chunk``. Returns each event it completed, raw with its separator, and whether the turn
        had ended once that event was read."""
        self._buffer += chunk
        events = []
        while (end := _event_end(self._buffer)) is not None:
            raw, self._buffer = self._buffer[: end[1]], self._buffer[end[1] :]
            self.event(raw[: end[0]])
            events.append((raw, self.ending))
        return events

    def flush(self) -> bytes:
        """The bytes after the last complete event, read as one last event."""
        rest, self._buffer = self._buffer, b""
        if rest.strip():
            self.event(rest)
        return rest

    @property
    @abstractmethod
    def ending(self) -> bool:
        """Whether the event that ends the turn has been read."""

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the stream has nothing more to send."""

    @abstractmethod
    def event(self, event: bytes) -> None:
        """Read one event, without its separator."""

    @abstractmethod
    def reply(self) -> Reply: ...


class Dialect(ABC):
    type: ClassVar[DialectType]
    #: The API's path under a base URL: the harness route under ``/t/{id}/v1``, and the upstream's in text mode.
    path: ClassVar[str]
    #: Whether a raw-content reply (``use_raw_content``) means anything in this API.
    raw_content: ClassVar[bool] = False

    @abstractmethod
    def parse_request(self, body: Any) -> CallRequest:
        """The call, read from its JSON body. Raises ``RequestError`` for one capture refuses."""

    def api_key(self, headers: Mapping[str, str]) -> str | None:
        """The key the client sent, or None. OpenAI's clients send ``Authorization: Bearer <key>``."""
        parts = headers.get("Authorization", "").split(maxsplit=1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1].strip()

    def error(
        self,
        call: CallRequest | None,
        message: str,
        status: int,
        *,
        kind: str = "invalid_request_error",
        code: str | None = None,
    ) -> web.Response:
        """The error response for ``call``."""
        return web.Response(
            body=error_body(message, kind=kind, code=code), status=status, content_type="application/json"
        )

    # -- token mode: answer with a sampled turn ---------------------------------
    def unsupported_in_token_mode(self, call: CallRequest) -> str | None:
        """Why skycap can't serve ``call`` when it samples itself, or None. Text mode's upstream may serve it."""
        return None

    @abstractmethod
    def reply(self, call: CallRequest, turn: Turn) -> dict[str, Any]:
        """The whole reply body."""

    @abstractmethod
    def stream(self, reply: dict[str, Any]) -> Iterator[bytes]:
        """``reply`` as SSE frames. Built from the finished reply, so a retry can replay the same bytes."""

    def replayed(self, turn: Turn) -> dict[str, Any]:
        """The turn as ``parse_request`` rebuilds it when the client sends the reply back.

        The model node is stored under this turn's match hash, so the client's replay finds it even
        where the reply's wire form doesn't map back to exactly the parsed message.
        """
        return turn.message

    # -- text mode: read an upstream's reply ---------------------------------------
    @abstractmethod
    def parse_reply(self, body: Any) -> Reply:
        """The reply in a non-streamed body. Raises ``ValueError`` for one that can't be recorded."""

    @abstractmethod
    def assembler(self) -> Assembler:
        """A new assembler for one streamed reply."""


#: An SSE line ending: CRLF, LF or a bare CR, mixed freely within one stream.
_LINE_BREAK = re.compile(rb"\r\n|\r|\n")


def sse_data(event: bytes) -> bytes | None:
    """An SSE event's data as the SSE spec reads it: its ``data`` fields joined with a newline, or None
    when it has none. A line starting with ``:`` is a comment, and one space after the colon is dropped."""
    data = []
    for line in _LINE_BREAK.split(event):
        name, colon, value = line.partition(b":")
        if name != b"data":
            continue
        if colon and value.startswith(b" "):
            value = value[1:]
        data.append(value)
    return b"\n".join(data) if data else None


def sse_json(event: bytes) -> Any:
    """An SSE event's data decoded as JSON, or None when it has no data or the data isn't JSON."""
    data = sse_data(event)
    if data is None:
        return None
    try:
        return orjson.loads(data)
    except orjson.JSONDecodeError:
        return None


def _event_end(buffer: bytes) -> tuple[int, int] | None:
    """Where the first complete SSE event ends: (end of event, start of next), or None until one has.

    An event ends at a blank line, and lines end in CRLF, LF or a bare CR. A CR at the end of the buffer may
    be the first half of a CRLF, so a line it ends waits for the next byte. When it ends the blank line, the
    event is over either way: it is returned at once, and a LF that follows reads as an empty event.
    """
    line_start = 0
    while (match := _LINE_BREAK.search(buffer, line_start)) is not None:
        start, end = match.span()
        if start == line_start:
            return start, end
        if end == len(buffer) and buffer[start:end] == b"\r":
            return None
        line_start = end
    return None
