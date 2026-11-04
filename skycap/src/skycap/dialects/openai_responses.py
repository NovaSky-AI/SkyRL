"""The OpenAI Responses API as a dialect: items in, turns out; one sampled turn in, items out.

**Request.** Responses sends a list of items where chat sends messages. A
model turn arrives split into several items (``reasoning``, ``message``, one
``function_call`` per call), so items are regrouped into chat-template turns by
vLLM's rule (``vllm/entrypoints/openai/responses/utils.py``,
``_construct_message_from_response_item``): a ``reasoning`` or assistant
``message`` item fills that field of the previous turn if the previous turn is
an assistant turn and the field is still empty, else starts a new assistant
turn; a ``function_call`` is appended to the previous assistant turn's calls,
else starts one. Items with a role are turns of that role, and ``instructions``
is a leading ``system`` turn. (Note that later on in the renderer, ``developer``
role items may be converted to ``system`` role items.)

skycap is stricter than vLLM where vLLM would drop part of the input and so
change what the model sees: a non-function tool, a second content part in a
reasoning item or assistant message, a trailing partial assistant item (vLLM
continues it), and server-held state (``previous_response_id``). Those are
refused, never dropped. Fields whose chat counterpart the chat dialect ignores
(``tool_choice``, ``parallel_tool_calls``, ``text``, ``reasoning``, ``include``)
are ignored here too, and forwarded in text mode.

Reasoning goes in the turn's ``reasoning_content``, the name skycap's renderers
parse it into (vLLM's own turns call it ``reasoning``).

**Reply.** One turn becomes up to N items, in the order a chat template renders
them: ``reasoning``, ``message``, then one ``function_call`` per call. Ids derive
from the sampled tokens, so a replayed reply is byte-identical. The stream is
built from the finished reply.

**Matching.** The items of a reply don't always regroup into exactly the parsed
turn: a parsed ``content: ""`` beside tool calls has no ``message`` item, so the
replay rebuilds a turn with no ``content`` at all. ``replayed`` runs the reply's
items back through the request's regrouping, and the model node is stored under
that turn's hash, so the replay finds it whatever the round trip loses.
"""

from __future__ import annotations

import itertools
import secrets
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import orjson
from aiohttp import web

from skycap.dialects.base import (
    SAMPLING_KEYS,
    Assembler,
    CallRequest,
    Dialect,
    DialectType,
    Reply,
    RequestError,
    Turn,
    sse_json,
)

#: Request fields refused in both modes: they ask for state skycap doesn't keep.
STATEFUL_FIELDS = ("previous_response_id", "conversation")
#: Content part types read as text in a user, system or developer message.
INPUT_TEXT_PARTS = ("input_text", "text", "output_text")
#: Statuses of an item vLLM would continue rather than answer after.
PARTIAL_STATUSES = ("in_progress", "incomplete")
#: Output item types a reply's turn is rebuilt from.
TURN_ITEM_TYPES = ("reasoning", "message", "function_call")
#: Response events that end a streamed reply.
TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


class ResponsesDialect(Dialect):
    type = DialectType.OPENAI_RESPONSES
    path = "/responses"
    # `use_raw_content` is for chat harnesses that parse the reply text themselves and replay only `content`
    # (Terminus-2): the whole turn, reasoning included, then comes back unchanged. A Responses client acts only on
    # structured items, so raw text with inline tool calls would never run one, and it replays reasoning as its
    # own item, so parsed replies already come back unchanged. A call on a raw-content server is refused (400)
    # rather than answered against the server's setting; the chat route still serves such harnesses.
    raw_content = False

    def parse_request(self, body: Any) -> CallRequest:
        if not isinstance(body, dict):
            raise RequestError("request body must be a JSON object")
        for name in STATEFUL_FIELDS:
            if body.get(name) is not None:
                raise RequestError(f"`{name}` is not supported: skycap keeps no responses; send the whole history")
        if body.get("background"):
            raise RequestError("`background` is not supported: skycap keeps no responses")
        messages: list[dict[str, Any]] = []
        instructions = body.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise RequestError("`instructions` must be a string")
        if instructions:
            messages.append({"role": "system", "content": instructions})
        items = body.get("input")
        if isinstance(items, str):
            messages.append({"role": "user", "content": items})
        elif isinstance(items, list):
            _refuse_trailing_partial(items)
            messages.extend(items_to_turns(items, "input"))
        else:
            raise RequestError("`input` must be a string or a list of items")
        sampling = {key: body[key] for key in SAMPLING_KEYS if body.get(key) is not None}
        sampling.pop("max_tokens", None)
        sampling.pop("max_completion_tokens", None)
        if body.get("max_output_tokens") is not None:
            sampling["max_tokens"] = body["max_output_tokens"]
        return CallRequest(
            messages=messages,
            tools=_tools(body.get("tools")),
            model=body.get("model"),
            stream=bool(body.get("stream")),
            sampling=sampling,
            body=body,
            dialect=self.type,
        )

    def error(
        self,
        call: CallRequest | None,
        message: str,
        status: int,
        *,
        kind: str = "invalid_request_error",
        code: str | None = None,
    ) -> web.Response:
        """A context overflow on a streamed call is a ``response.failed`` event, the way OpenAI reports it.

        A streaming client reads failures from the stream: Codex treats an HTTP 400 as fatal, but reads
        ``response.failed`` with ``context_length_exceeded`` as a full context window, and compacts.
        """
        if call is None or not call.stream or code != "context_length_exceeded":
            return super().error(call, message, status, kind=kind, code=code)
        failed = {
            **_echo(call),
            "id": f"resp_{secrets.token_hex(16)}",
            "created_at": int(time.time()),
            "status": "failed",
            "model": call.model,
            "output": [],
            "usage": None,
            "incomplete_details": None,
            "error": {"code": code, "message": message},
        }
        events = _Events()
        body = events("response.created", response={**failed, "status": "in_progress", "error": None})
        body += events("response.failed", response=failed)
        return web.Response(body=body, content_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    def unsupported_in_token_mode(self, call: CallRequest) -> str | None:
        if call.body.get("truncation") == "auto":
            return '`truncation: "auto"` is not supported: skycap does not drop history to fit; it fails the call'
        if call.body.get("context_management"):
            return "`context_management` is not supported: skycap does not compact history"
        return None

    # -- token mode -----------------------------------------------------------------
    def reply(self, call: CallRequest, turn: Turn) -> dict[str, Any]:
        cut_off = turn.finish_reason == "length"
        return {
            **_echo(call),
            "id": f"resp_{turn.key}",
            "created_at": int(time.time()),
            "status": "incomplete" if cut_off else "completed",
            "model": turn.model,
            "output": output_items(turn),
            "usage": {
                "input_tokens": turn.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": turn.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": turn.prompt_tokens + turn.completion_tokens,
            },
            "incomplete_details": {"reason": "max_output_tokens"} if cut_off else None,
            "error": None,
        }

    def stream(self, reply: dict[str, Any]) -> Iterator[bytes]:
        """``response.created``, then each item opened, filled in one delta and closed, then the final event.

        A client keeps an item when its ``response.output_item.done`` arrives, and the turn when the
        final event does; the SDK's stream helper also needs each message's ``content_part.added``.
        """
        events = _Events()
        opened = {**reply, "status": "in_progress", "output": [], "usage": None, "incomplete_details": None}
        yield events("response.created", response=opened)
        yield events("response.in_progress", response=opened)
        for index, item in enumerate(reply["output"]):
            at = {"item_id": item["id"], "output_index": index}
            yield events("response.output_item.added", output_index=index, item=_opened(item))
            if item["type"] == "message":
                part = item["content"][0]
                yield events("response.content_part.added", **at, content_index=0, part={**part, "text": ""})
                if part["text"]:
                    yield events("response.output_text.delta", **at, content_index=0, delta=part["text"], logprobs=[])
                yield events("response.output_text.done", **at, content_index=0, text=part["text"], logprobs=[])
                yield events("response.content_part.done", **at, content_index=0, part=part)
            elif item["type"] == "reasoning":
                text = item["content"][0]["text"]
                yield events("response.reasoning_text.delta", **at, content_index=0, delta=text)
                yield events("response.reasoning_text.done", **at, content_index=0, text=text)
            elif item["type"] == "function_call":
                if item["arguments"]:
                    yield events("response.function_call_arguments.delta", **at, delta=item["arguments"])
                yield events(
                    "response.function_call_arguments.done", **at, name=item["name"], arguments=item["arguments"]
                )
            yield events("response.output_item.done", output_index=index, item=item)
        yield events(f"response.{reply['status']}", response=reply)

    def replayed(self, turn: Turn) -> dict[str, Any]:
        turns = items_to_turns(output_items(turn), "output")
        if len(turns) != 1 or turns[0].get("role") != "assistant":
            raise ValueError(f"a reply's items regrouped into {len(turns)} turns, not one assistant turn")
        return turns[0]

    # -- text mode ------------------------------------------------------------------
    def parse_reply(self, body: Any) -> Reply:
        """The turn a ``response`` object carries. A failed or filtered one, or one with items skycap
        can't hold in a turn, is not a turn: the caller records the call as a failure."""
        if not isinstance(body, dict):
            raise ValueError("response is not a JSON object")
        status = body.get("status")
        reason = (body.get("incomplete_details") or {}).get("reason")
        if status == "incomplete" and reason != "max_output_tokens":
            raise ValueError(f"response is incomplete ({reason})")
        if status not in ("completed", "incomplete"):
            raise ValueError(f"response status is {status!r}")
        output = body.get("output")
        if not isinstance(output, list):
            raise ValueError("response has no output")
        for item in output:
            if not isinstance(item, dict) or item.get("type") not in TURN_ITEM_TYPES:
                kind = item.get("type") if isinstance(item, dict) else type(item).__name__
                raise ValueError(f"response output item {kind!r} can't be recorded as a turn")
        turns = items_to_turns(output, "output") or [{"role": "assistant", "content": ""}]
        if len(turns) != 1 or turns[0].get("role") != "assistant":
            raise ValueError(f"response output is {len(turns)} turns, not one assistant turn")
        message = turns[0]
        if status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        usage = body.get("usage")
        if isinstance(usage, dict):
            usage = {"prompt_tokens": usage.get("input_tokens"), "completion_tokens": usage.get("output_tokens")}
        return Reply(message=message, finish_reason=finish_reason, usage=usage)

    def assembler(self) -> Assembler:
        return ResponsesAssembler()


@dataclass(slots=True)
class ResponsesAssembler(Assembler):
    """Rebuilds the turn a Responses event stream carried: its items as each ``output_item.done`` closes
    them, and the final response object."""

    items: dict[int, dict[str, Any]] = field(default_factory=dict)
    response: dict[str, Any] | None = None
    _buffer: bytes = b""

    @property
    def ending(self) -> bool:
        return self.response is not None

    @property
    def done(self) -> bool:
        return self.response is not None

    def event(self, event: bytes) -> None:
        payload = sse_json(event)
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if kind == "response.output_item.done" and isinstance(payload.get("item"), dict):
            self.items[int(payload.get("output_index") or 0)] = payload["item"]
        elif kind in TERMINAL_EVENTS and isinstance(payload.get("response"), dict):
            self.response = payload["response"]
        elif kind == "error":
            self.response = {"status": "failed", "error": payload}

    def reply(self) -> Reply:
        if self.response is None:
            raise ValueError("stream ended without a final response event")
        response = dict(self.response)
        if not response.get("output"):
            # A final event may leave the items out; they were sent as each one closed.
            response["output"] = [self.items[index] for index in sorted(self.items)]
        return DialectType.OPENAI_RESPONSES.get_dialect().parse_reply(response)


# -- items and turns ------------------------------------------------------------------
def items_to_turns(items: Sequence[Any], where: str) -> list[dict[str, Any]]:
    """Responses items as chat-template turns, by vLLM's rule (module docstring)."""
    turns: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise RequestError(f"`{where}[{index}]` must be an object")
        turn = _from_item(item, turns[-1] if turns else None, f"{where}[{index}]")
        if turn is not None:
            turns.append(turn)
    return turns


def _from_item(item: dict[str, Any], previous: dict[str, Any] | None, where: str) -> dict[str, Any] | None:
    """A new turn for ``item``, or None when it went into ``previous``, the last turn so far."""
    assistant = previous if previous is not None and previous.get("role") == "assistant" else None
    kind = item.get("type")
    if kind == "function_call":
        call = _function_call(item, where)
        if assistant is not None:
            assistant.setdefault("tool_calls", []).append(call)
            return None
        return {"role": "assistant", "tool_calls": [call]}
    if kind == "reasoning":
        reasoning = _reasoning(item, where)
        if assistant is not None and assistant.get("reasoning_content") is None:
            assistant["reasoning_content"] = reasoning
            return None
        return {"role": "assistant", "reasoning_content": reasoning}
    if kind == "function_call_output":
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            raise RequestError(f"`{where}.call_id` must be a string")
        return {"role": "tool", "tool_call_id": call_id, "content": _tool_output(item.get("output"), where)}
    if kind in (None, "message") and "role" in item:
        role = item.get("role")
        if role == "assistant":
            text = _assistant_text(item.get("content"), where)
            if assistant is not None and assistant.get("content") is None:
                assistant["content"] = text
                return None
            return {"role": "assistant", "content": text}
        if role in ("user", "system", "developer"):
            return {"role": role, "content": _input_text(item.get("content"), where)}
        raise RequestError(f"`{where}` has an unsupported role {role!r}")
    # vLLM's wording.
    raise RequestError(f"Unsupported input item type: {kind}")


def _function_call(item: dict[str, Any], where: str) -> dict[str, Any]:
    if item.get("namespace"):
        raise RequestError(f"`{where}`: namespaced tool calls are not supported; send function tools only")
    for name in ("call_id", "name", "arguments"):
        if not isinstance(item.get(name), str):
            raise RequestError(f"`{where}.{name}` must be a string")
    return {
        "id": item["call_id"],
        "type": "function",
        "function": {"name": item["name"], "arguments": item["arguments"]},
    }


def _reasoning(item: dict[str, Any], where: str) -> str:
    """The reasoning text: the item's one ``content`` part, else its one ``summary`` part, else empty."""
    if item.get("encrypted_content"):
        raise RequestError(f"`{where}`: encrypted reasoning is not supported; send it as `content`")
    for name in ("content", "summary"):
        parts = item.get(name)
        if not parts:
            continue
        if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict):
            raise RequestError(f"`{where}.{name}` must hold exactly one part; skycap keeps one reasoning text per turn")
        text = parts[0].get("text")
        if not isinstance(text, str):
            raise RequestError(f"`{where}.{name}[0].text` must be a string")
        return text
    return ""


def _assistant_text(content: Any, where: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict):
        raise RequestError(f"`{where}.content` must be a string or exactly one part; skycap keeps one text per turn")
    part = content[0]
    if part.get("type") not in ("output_text", "text") or not isinstance(part.get("text"), str):
        raise RequestError(f"`{where}.content[0]` of type {part.get('type')!r} is not supported; send `output_text`")
    return part["text"]


def _input_text(content: Any, where: str) -> str:
    """A message's text: a string, or its text parts joined with a newline, as vLLM joins them."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError(f"`{where}.content` must be a string or a list of parts")
    texts = []
    for index, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        if kind not in INPUT_TEXT_PARTS or not isinstance(part.get("text"), str):
            raise RequestError(f"`{where}.content[{index}]` of type {kind!r} is not supported; send text parts")
        texts.append(part["text"])
    return "\n".join(texts)


def _tool_output(output: Any, where: str) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return _input_text(output, f"{where}.output")
    raise RequestError(f"`{where}.output` must be a string or a list of parts")


def _refuse_trailing_partial(items: list[Any]) -> None:
    """vLLM continues a trailing assistant item that is ``in_progress`` or ``incomplete``; skycap refuses it.

    Continuing would sample into the middle of a turn the graph holds as finished.
    """
    if not items or not isinstance(items[-1], dict):
        return
    last = items[-1]
    is_assistant = last.get("type") == "reasoning" or (
        last.get("type") in (None, "message") and last.get("role") == "assistant"
    )
    if is_assistant and last.get("status") in PARTIAL_STATUSES:
        raise RequestError(
            f"a trailing assistant item with status {last['status']!r} is not supported: skycap does not "
            "continue a partial turn"
        )


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    """Function tools in the chat spelling: every field but ``type`` under ``function``, as vLLM does."""
    if tools is None or tools == []:
        return None
    if not isinstance(tools, list):
        raise RequestError("`tools` must be a list")
    converted = []
    for index, tool in enumerate(tools):
        kind = tool.get("type") if isinstance(tool, dict) else None
        if kind != "function":
            raise RequestError(f"tool type {kind!r} (`tools[{index}]`) is not supported: skycap serves function tools")
        if not isinstance(tool.get("name"), str):
            raise RequestError(f"`tools[{index}].name` must be a string")
        function = {key: value for key, value in tool.items() if key != "type" and value is not None}
        converted.append({"type": "function", "function": function})
    return converted


def output_items(turn: Turn) -> list[dict[str, Any]]:
    """A sampled turn as output items: reasoning, message, then one per call. Ids derive from ``turn.key``.

    A turn with nothing at all still gets one empty message, so the client's history holds the turn.
    """
    message = turn.message
    items: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        items.append(
            {
                "type": "reasoning",
                "id": f"rs_{turn.key}",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning}],
                "encrypted_content": None,
            }
        )
    content = message.get("content")
    calls = message.get("tool_calls") or []
    if content or (not items and not calls):
        items.append(
            {
                "type": "message",
                "id": f"msg_{turn.key}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content or "", "annotations": []}],
            }
        )
    for index, call in enumerate(calls):
        function = call.get("function") or {}
        items.append(
            {
                "type": "function_call",
                "id": f"fc_{turn.key}_{index}",
                "call_id": call.get("id"),
                "name": function.get("name"),
                "arguments": function.get("arguments") or "",
                "status": "completed",
            }
        )
    return items


def _opened(item: dict[str, Any]) -> dict[str, Any]:
    """``item`` as ``response.output_item.added`` announces it: before any of its content."""
    if item["type"] == "message":
        return {**item, "status": "in_progress", "content": []}
    if item["type"] == "reasoning":
        return {**item, "content": []}
    if item["type"] == "function_call":
        return {**item, "status": "in_progress", "arguments": ""}
    return item


def _echo(call: CallRequest) -> dict[str, Any]:
    """The request fields a response object repeats, as sent: the schema requires them, and vLLM echoes them."""
    body = call.body
    return {
        "object": "response",
        "instructions": body.get("instructions"),
        "metadata": body.get("metadata"),
        "tools": body.get("tools") or [],
        "tool_choice": body.get("tool_choice", "auto"),
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"),
        "previous_response_id": None,
        "reasoning": body.get("reasoning"),
        "text": body.get("text") or {"format": {"type": "text"}},
        "truncation": body.get("truncation") or "disabled",
    }


class _Events:
    """SSE frames of one response, numbered in order."""

    def __init__(self) -> None:
        self._numbers = itertools.count()

    def __call__(self, kind: str, **fields: Any) -> bytes:
        payload = {"type": kind, "sequence_number": next(self._numbers), **fields}
        return f"event: {kind}\ndata: ".encode() + orjson.dumps(payload) + b"\n\n"
