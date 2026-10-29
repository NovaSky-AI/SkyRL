"""The Responses dialect in token mode, end to end: the OpenAI SDK's Responses client, a capture server,
the fake renderer and the mock engine."""

from __future__ import annotations

from typing import Any

import openai
import orjson
import pytest

from skycap import hashing
from skycap.hashing import MatchKey
from tests.conftest import openai_client
from tests.fake_renderer import END, decode, encode
from tests.test_tokens import TokenStack, client, token_stack, user

TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": "Search.",
        "parameters": {"type": "object", "properties": {}},
    }
]
CHAT_TOOLS = [{"type": "function", "function": {k: v for k, v in TOOLS[0].items() if k != "type"}}]
PLAIN_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"


def items_of(response: Any) -> list[dict]:
    """What a client sends back: each output item as the SDK parsed it."""
    return [item.model_dump(exclude_none=True) for item in response.output]


def thinking_then_tool(prompt: list[int], sampling: dict) -> list[int]:
    """Reasoning plus a tool call and no text, parsed as ``content: ""``; else a plain answer."""
    if "TOOL" in decode(prompt[-40:]):
        return [*encode("THINK:plan|CALL:search:{}"), END]
    return [*encode(f"re{len(prompt)}"), END]


async def tool_loop(stack: TokenStack, created: dict, *, strip: bool = False) -> list[Any]:
    """user → call → tool result → answer → user → answer, sending back each reply's items."""
    llm = client(created["base_url"])
    history: list[dict] = [user("use a TOOL")]
    replies = []
    for follow_up in ("result", "thanks", None):
        reply = await llm.responses.create(model="policy", input=history, tools=TOOLS)
        replies.append(reply)
        sent = items_of(reply)
        if strip:
            sent = [{k: v for k, v in item.items() if k not in ("id", "status")} for item in sent]
        history += sent
        if follow_up == "result":
            history.append({"type": "function_call_output", "call_id": reply.output[-1].call_id, "output": "found"})
        elif follow_up is not None:
            history.append(user(follow_up))
    return replies


async def document(stack: TokenStack, trajectory_id: str) -> dict:
    async with stack.http.get(f"{stack.url}/trajectories/{trajectory_id}") as response:
        return await response.json()


async def post(stack: TokenStack, created: dict, body: dict, **headers: str) -> tuple[int, bytes, str]:
    async with stack.http.post(f"{created['base_url']}/responses", json=body, headers=headers) as response:
        return response.status, await response.read(), response.content_type


def event_types(body: bytes) -> list[str]:
    return [orjson.loads(line[6:])["type"] for line in body.split(b"\n") if line.startswith(b"data: ")]


# -- one path ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("strip", [False, True], ids=["verbatim replay", "ids and statuses dropped"])
async def test_a_tool_loop_over_responses_bridges_every_call_after_the_first(strip: bool) -> None:
    async with token_stack() as stack:
        created = await stack.create()
        replies = await tool_loop(stack, created, strip=strip)
        finished = await stack.finish(created["id"])

    assert [item.type for item in replies[0].output] == ["function_call"]
    assert [item.type for item in replies[1].output] == ["message"]
    assert finished["unbridged_calls"] == 0
    assert len(finished["samples"]) == 1


async def test_reasoning_and_a_call_with_empty_content_are_matched_on_replay() -> None:
    """The case ``replayed`` exists for: the parsed ``content: ""`` has no item, so the replay has no
    ``content``. The node is stored as the replay, under its hash, and the next call bridges from it."""
    async with token_stack(completion=thinking_then_tool) as stack:
        created = await stack.create()
        replies = await tool_loop(stack, created)
        graph = stack.server.trajectories[created["id"]].graph
        key = MatchKey.tokens(hashing.tools_hash(CHAT_TOOLS), "policy")
        model_nodes = [node for node in graph if node.author == "model"]
        finished = await stack.finish(created["id"])

    assert [item.type for item in replies[0].output] == ["reasoning", "function_call"]
    assert finished["unbridged_calls"] == 0 and len(finished["samples"]) == 1
    first = model_nodes[0]
    assert first.message == {
        "role": "assistant",
        "reasoning_content": "plan",
        "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
    }
    assert all(key(node.message) == node.match_hash for node in model_nodes)


async def test_the_same_conversation_over_chat_and_responses_sends_the_same_prompts() -> None:
    async with token_stack() as stack:
        over_responses = await stack.create()
        await tool_loop(stack, over_responses)
        responses_prompts = [request["token_ids"] for request in stack.engine.requests]

        over_chat = await stack.create()
        llm = client(over_chat["base_url"])
        messages: list[dict] = [user("use a TOOL")]
        for follow_up in ("result", "thanks", None):
            reply = await llm.chat.completions.create(model="policy", messages=messages, tools=CHAT_TOOLS)
            message = reply.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            if follow_up == "result":
                messages.append({"role": "tool", "tool_call_id": message.tool_calls[0].id, "content": "found"})
            elif follow_up is not None:
                messages.append(user(follow_up))
        chat_prompts = [request["token_ids"] for request in stack.engine.requests[len(responses_prompts) :]]

    assert responses_prompts == chat_prompts


# -- the stream ------------------------------------------------------------------------------------------
async def test_the_sdks_stream_helper_rebuilds_the_same_response_as_the_json_reply() -> None:
    async with token_stack(completion=thinking_then_tool) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        whole = await llm.responses.create(model="policy", input=[user("use a TOOL")], tools=TOOLS)
        async with llm.responses.stream(model="policy", input=[user("use a TOOL")], tools=TOOLS) as stream:
            kinds = [event.type async for event in stream]
            streamed = await stream.get_final_response()

    assert items_of(streamed) == items_of(whole)
    assert streamed.status == whole.status == "completed"
    assert kinds.count("response.output_item.done") == 2 and kinds[-1] == "response.completed"


# -- a length cut-off -----------------------------------------------------------------------------------------
async def test_a_cut_off_is_reported_incomplete_and_recorded_as_length() -> None:
    async with token_stack() as stack:
        stack.engine.finish_reason = "length"
        created = await stack.create()
        whole = await client(created["base_url"]).responses.create(model="policy", input="hi")
        _, streamed, _ = await post(stack, created, {"model": "policy", "input": "hi", "stream": True})
        calls = [call for node in (await document(stack, created["id"]))["nodes"] for call in node["calls"]]

    assert whole.status == "incomplete" and whole.incomplete_details.reason == "max_output_tokens"
    assert event_types(streamed)[-1] == "response.incomplete"
    assert [call["finish_reason"] for call in calls] == ["length", "length"]


# -- errors and refusals --------------------------------------------------------------------------------------
async def test_an_overflow_is_a_failed_response_when_streamed_and_a_400_otherwise() -> None:
    async with token_stack(max_model_len=5) as stack:
        created = await stack.create()
        status, body, content_type = await post(stack, created, {"model": "policy", "input": "hi", "stream": True})
        with pytest.raises(openai.BadRequestError) as refused:
            await client(created["base_url"]).responses.create(model="policy", input="hi")

    assert status == 200 and content_type == "text/event-stream"
    assert event_types(body) == ["response.created", "response.failed"]
    assert b'"code":"context_length_exceeded"' in body
    assert refused.value.code == "context_length_exceeded"


@pytest.mark.parametrize(
    "fields, match",
    [
        ({"truncation": "auto"}, "truncation"),
        ({"context_management": [{"type": "compaction"}]}, "context_management"),
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"tools": [{"type": "custom", "name": "apply_patch"}]}, "tool type 'custom'"),
        ({"input": [{"type": "item_reference", "id": "x"}]}, "Unsupported input item type"),
    ],
)
async def test_what_token_mode_cant_serve_is_refused_before_sampling(fields: dict, match: str) -> None:
    async with token_stack() as stack:
        created = await stack.create()
        status, body, _ = await post(stack, created, {"model": "policy", "input": "hi", **fields})

    assert status == 400 and match in orjson.loads(body)["error"]["message"]
    assert stack.engine.requests == []


@pytest.mark.parametrize(
    "fields",
    [
        {"tool_choice": "required"},
        {"parallel_tool_calls": False},
        {"text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}},
        {"reasoning": {"effort": "none"}},
        {"include": ["message.output_text.logprobs"], "top_logprobs": 3},
    ],
)
async def test_fields_chat_ignores_are_ignored_and_echoed(fields: dict) -> None:
    async with token_stack() as stack:
        created = await stack.create()
        status, body, _ = await post(stack, created, {"model": "policy", "input": "hi", "tools": TOOLS, **fields})

    reply = orjson.loads(body)
    assert status == 200 and reply["status"] == "completed"
    for name in ("tool_choice", "parallel_tool_calls"):
        if name in fields:
            assert reply[name] == fields[name]


async def test_a_raw_content_server_refuses_responses_and_still_serves_chat() -> None:
    async with token_stack(use_raw_content=True) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        with pytest.raises(openai.BadRequestError, match="raw content"):
            await llm.responses.create(model="policy", input="hi")
        reply = await llm.chat.completions.create(model="policy", messages=[user("hi")])

    assert reply.choices[0].message.content


async def test_the_responses_route_takes_the_trajectorys_own_key() -> None:
    async with token_stack() as stack:
        stack.server.require_api_key = True
        created = await stack.create()
        with pytest.raises(openai.AuthenticationError):
            await openai_client(created["base_url"], api_key="wrong").responses.create(model="policy", input="hi")
        reply = await openai_client(created["base_url"], api_key=created["api_key"]).responses.create(
            model="policy", input="hi"
        )

    assert reply.status == "completed"


async def test_a_marked_retry_gets_the_original_reply_without_sampling_again() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        body = {"model": "policy", "input": "hi", "stream": True}
        _, first, _ = await post(stack, created, body)
        _, retried, _ = await post(stack, created, body, **{"x-stainless-retry-count": "1"})

    assert retried == first and len(stack.engine.requests) == 1


# -- the record and the developer role ------------------------------------------------------------------------
async def test_each_call_records_the_dialect_it_came_in_on() -> None:
    async with token_stack() as stack:
        over_responses = await stack.create()
        await client(over_responses["base_url"]).responses.create(model="policy", input="hi")
        over_chat = await stack.create()
        await client(over_chat["base_url"]).chat.completions.create(model="policy", messages=[user("hi")])
        dialects = [
            [call["dialect"] for node in (await document(stack, created["id"]))["nodes"] for call in node["calls"]]
            for created in (over_responses, over_chat)
        ]

    assert dialects == [["openai_responses"], ["openai_chat"]]


async def test_developer_messages_fold_into_one_system_message_for_a_template_without_the_role() -> None:
    async with token_stack() as stack:
        stack.server.backend.chat_template = PLAIN_TEMPLATE
        created = await stack.create()
        llm = client(created["base_url"])
        history: list[dict] = [{"role": "developer", "content": "D"}, user("hi")]
        first = await llm.responses.create(model="policy", instructions="I", input=history)
        history += [*items_of(first), user("again")]
        await llm.responses.create(model="policy", instructions="I", input=history)
        prompts = [decode(request["token_ids"]) for request in stack.engine.requests]
        stored = [node.message for node in stack.server.trajectories[created["id"]].graph if node.author == "client"]
        finished = await stack.finish(created["id"])

    assert prompts[0].startswith("systemI\n\nDuserhi") and "developer" not in prompts[0]
    assert stored[0] == {"role": "system", "content": "I\n\nD"}
    assert finished["unbridged_calls"] == 0


async def test_a_chat_request_with_developer_messages_folds_the_same_way() -> None:
    async with token_stack() as stack:
        stack.server.backend.chat_template = PLAIN_TEMPLATE
        created = await stack.create()
        await client(created["base_url"]).responses.create(
            model="policy", instructions="I", input=[{"role": "developer", "content": "D"}, user("hi")]
        )
        other = await stack.create()
        await client(other["base_url"]).chat.completions.create(
            model="policy",
            messages=[{"role": "system", "content": "I"}, {"role": "developer", "content": "D"}, user("hi")],
        )

    assert stack.engine.requests[0]["token_ids"] == stack.engine.requests[1]["token_ids"]


async def test_a_template_that_knows_the_developer_role_gets_it() -> None:
    async with token_stack() as stack:
        stack.server.backend.chat_template = "{% if m.role == 'developer' %}{% endif %}"
        created = await stack.create()
        await client(created["base_url"]).responses.create(
            model="policy", input=[{"role": "developer", "content": "D"}, user("hi")]
        )

    assert decode(stack.engine.requests[0]["token_ids"]).startswith("developerDuserhi")
