"""The Responses dialect in text mode: forwarded to an upstream that speaks Responses, and recorded."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
import orjson
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skycap.server import CaptureServer
from skycap.text import TextBackend
from tests.conftest import openai_client


def said(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_up",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def called(call_id: str) -> dict:
    return {"type": "function_call", "id": "fc_up", "call_id": call_id, "name": "search", "arguments": "{}"}


class MockResponses:
    """An upstream speaking the Responses API. Every call gets ``output`` with ``status`` (and ``reason``),
    whole or streamed in ``chunk``-byte pieces."""

    def __init__(self) -> None:
        self.requests: list[bytes] = []
        self.output: list[dict] = [said("a")]
        self.status = "completed"
        self.reason: str | None = None
        self.chunk = 7
        #: The line ending of every line, and so (twice) of every event.
        self.eol = b"\n"
        #: Seconds to keep a streamed reply's connection open after its last event.
        self.linger = 0.0
        #: The final event's type; by default `response.<status>`, as the API sends it. vLLM 0.31 sends
        #: `response.completed` whatever the status.
        self.final: str | None = None
        #: The bytes of the last streamed reply, as sent.
        self.sent = b""

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/responses", self.responses)
        return app

    def response(self, status: str | None = None, output: list | None = None) -> dict:
        status = status or self.status
        return {
            "id": "resp_up",
            "object": "response",
            "status": status,
            "model": "policy",
            "output": self.output if output is None else output,
            "incomplete_details": {"reason": self.reason} if self.reason else None,
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }

    def events(self) -> bytes:
        frames = [{"type": "response.created", "response": self.response("in_progress", [])}]
        for index, item in enumerate(self.output):
            frames.append({"type": "response.output_item.added", "output_index": index, "item": item})
            frames.append({"type": "response.output_item.done", "output_index": index, "item": item})
        frames.append({"type": self.final or f"response.{self.status}", "response": self.response()})
        eol = self.eol
        return b"".join(
            b"event: "
            + frame["type"].encode()
            + eol
            + b"data: "
            + orjson.dumps({**frame, "sequence_number": n})
            + eol
            + eol
            for n, frame in enumerate(frames)
        )

    async def responses(self, request: web.Request) -> web.StreamResponse:
        raw = await request.read()
        self.requests.append(raw)
        if not orjson.loads(raw).get("stream"):
            return web.Response(body=orjson.dumps(self.response()), content_type="application/json")
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        self.sent = self.events()
        for start in range(0, len(self.sent), self.chunk):
            await response.write(self.sent[start : start + self.chunk])
        await asyncio.sleep(self.linger)
        await response.write_eof()
        return response


@dataclass
class ResponsesStack:
    upstream: MockResponses
    server: CaptureServer
    url: str
    http: aiohttp.ClientSession

    async def create(self) -> dict:
        async with self.http.post(f"{self.url}/trajectories", json={}) as response:
            return await response.json()

    async def call(self, created: dict, body: dict) -> tuple[int, bytes]:
        async with self.http.post(f"{created['base_url']}/responses", data=orjson.dumps(body)) as response:
            return response.status, await response.read()

    async def document(self, created: dict) -> dict:
        async with self.http.get(f"{self.url}/trajectories/{created['id']}") as response:
            return await response.json()


@asynccontextmanager
async def responses_stack(**options: Any) -> AsyncIterator[ResponsesStack]:
    upstream = MockResponses()
    upstream_server = TestServer(upstream.app())
    await upstream_server.start_server()
    server = CaptureServer(TextBackend(str(upstream_server.make_url("/v1")), **options))
    capture = TestServer(server.app())
    await capture.start_server()
    try:
        async with aiohttp.ClientSession() as http:
            yield ResponsesStack(upstream, server, str(capture.make_url("")).rstrip("/"), http)
    finally:
        await capture.close()
        await upstream_server.close()


def calls(document: dict) -> list[dict]:
    return [call for node in document["nodes"] for call in node["calls"]]


# -- forwarding and recording ---------------------------------------------------------------------------
async def test_a_call_is_forwarded_as_sent_and_its_reply_relayed_and_recorded() -> None:
    async with responses_stack() as stack:
        created = await stack.create()
        body = {"model": "policy", "input": [{"role": "user", "content": "q"}], "client_metadata": {"x": 1}}
        status, reply = await stack.call(created, body)
        document = await stack.document(created)

    assert status == 200 and stack.upstream.requests == [orjson.dumps(body)]
    assert orjson.loads(reply) == stack.upstream.response()
    assert [node["message"] for node in document["nodes"]] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert calls(document) == [
        {
            **calls(document)[0],
            "dialect": "openai_responses",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
    ]


async def test_a_replayed_reply_continues_the_same_path() -> None:
    async with responses_stack() as stack:
        stack.upstream.output = [said("a"), called("c1")]
        created = await stack.create()
        llm = openai_client(created["base_url"])
        history: list[dict] = [{"role": "user", "content": "q"}]
        first = await llm.responses.create(model="policy", input=history)
        history += [item.model_dump(exclude_none=True) for item in first.output]
        history.append({"type": "function_call_output", "call_id": "c1", "output": "found"})
        await llm.responses.create(model="policy", input=history)
        graph = stack.server.trajectories[created["id"]].graph

    assert len(graph.paths()) == 1 and len(graph.nodes) == 4


async def test_a_stream_reaches_the_client_byte_for_byte_and_the_sdk_reads_it() -> None:
    async with responses_stack() as stack:
        created = await stack.create()
        status, body = await stack.call(created, {"model": "policy", "input": "q", "stream": True})
        sent = stack.upstream.sent
        async with openai_client(created["base_url"]).responses.stream(model="policy", input="q") as stream:
            final = await stream.get_final_response()
        document = await stack.document(created)

    assert status == 200 and body == sent
    assert final.output_text == "a"
    assert [call["finish_reason"] for call in calls(document)] == ["stop", "stop"]


# -- a length cut-off ------------------------------------------------------------------------------------
@pytest.mark.parametrize("stream, chunk", [(False, 7), (True, 1), (True, 7), (True, 100_000)])
@pytest.mark.parametrize("final", [None, "response.completed"], ids=["api", "vllm"])
async def test_a_cut_off_reaches_the_client_as_the_upstream_sent_it_and_is_recorded_as_length(
    stream: bool, chunk: int, final: str | None
) -> None:
    async with responses_stack() as stack:
        stack.upstream.status, stack.upstream.reason = "incomplete", "max_output_tokens"
        stack.upstream.chunk, stack.upstream.final = chunk, final
        created = await stack.create()
        _, body = await stack.call(created, {"model": "policy", "input": "q", "stream": stream})
        document = await stack.document(created)

    assert body == (stack.upstream.sent if stream else orjson.dumps(stack.upstream.response()))
    assert [call["finish_reason"] for call in calls(document)] == ["length"]


@pytest.mark.parametrize("stream", [False, True])
async def test_any_other_incomplete_reply_is_relayed_untouched_and_recorded_as_a_failure(stream: bool) -> None:
    async with responses_stack() as stack:
        stack.upstream.status, stack.upstream.reason = "incomplete", "content_filter"
        created = await stack.create()
        _, body = await stack.call(created, {"model": "policy", "input": "q", "stream": stream})
        document = await stack.document(created)

    assert body == (stack.upstream.sent if stream else orjson.dumps(stack.upstream.response()))
    assert document["nodes"] == [] and "content_filter" in document["failures"][0]["error"]


async def test_a_failed_response_is_relayed_and_recorded_as_a_failure() -> None:
    async with responses_stack() as stack:
        stack.upstream.status = "failed"
        created = await stack.create()
        status, _ = await stack.call(created, {"model": "policy", "input": "q", "stream": True})
        document = await stack.document(created)

    assert status == 200 and document["nodes"] == [] and "failed" in document["failures"][0]["error"]


# -- what text mode forwards and what it refuses --------------------------------------------------------
@pytest.mark.parametrize(
    "fields",
    [
        {"truncation": "auto"},
        {"context_management": [{"type": "compaction", "compact_threshold": 1000}]},
        {"tool_choice": "required", "parallel_tool_calls": False},
    ],
)
async def test_fields_token_mode_refuses_or_ignores_reach_the_upstream(fields: dict) -> None:
    async with responses_stack() as stack:
        created = await stack.create()
        status, _ = await stack.call(created, {"model": "policy", "input": "q", **fields})

    assert status == 200
    assert all(orjson.loads(stack.upstream.requests[0])[name] == value for name, value in fields.items())


@pytest.mark.parametrize(
    "fields",
    [
        {"previous_response_id": "resp_1"},
        {"input": [{"type": "message", "role": "assistant", "content": "half", "status": "incomplete"}]},
        {"tools": [{"type": "web_search"}]},
    ],
)
async def test_what_the_dialect_refuses_never_reaches_the_upstream(fields: dict) -> None:
    async with responses_stack() as stack:
        created = await stack.create()
        status, _ = await stack.call(created, {"model": "policy", "input": "q", **fields})

    assert status == 400 and stack.upstream.requests == []


@pytest.mark.parametrize("eol", [b"\r", b"\r\n"])
async def test_a_stream_with_bare_cr_or_crlf_lines_is_relayed_without_waiting_for_the_upstream_to_close(
    eol: bytes,
) -> None:
    """The relay ends at the final event, not at EOF: an upstream that keeps the connection open must not stall."""
    async with responses_stack() as stack:
        stack.upstream.eol, stack.upstream.linger, stack.upstream.chunk = eol, 30.0, 3
        stack.upstream.status, stack.upstream.reason = "incomplete", "max_output_tokens"
        created = await stack.create()
        status, body = await asyncio.wait_for(
            stack.call(created, {"model": "policy", "input": "q", "stream": True}), timeout=5
        )
        document = await stack.document(created)

    assert status == 200 and body == stack.upstream.sent
    assert [call["finish_reason"] for call in calls(document)] == ["length"]
