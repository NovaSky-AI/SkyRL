"""The Responses dialect's conversions, one rule at a time: items to turns, a turn to items, and back.

No server: these call the dialect directly. ``test_responses_tokens.py`` and
``test_responses_text.py`` run the same rules through a capture server.
"""

from __future__ import annotations

import copy
from typing import Any

import orjson
import pytest

from skycap.dialects import CallRequest, DialectType, RequestError, Turn
from skycap.dialects.base import sse_data
from skycap.dialects.openai_chat import ChatDialect, StreamAssembler
from skycap.dialects.openai_responses import (
    ResponsesAssembler,
    ResponsesDialect,
    output_items,
)
from skycap.hashing import MatchKey

RESPONSES = DialectType.OPENAI_RESPONSES.get_dialect()
KEY = MatchKey.tokens("", "policy")


def parse(items: Any, **body: Any) -> CallRequest:
    return RESPONSES.parse_request({"model": "policy", "input": items, **body})


def turns(items: Any, **body: Any) -> list[dict]:
    return parse(items, **body).messages


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def reasoning(text: str | None = None, *, summary: str | None = None, **fields: Any) -> dict:
    item: dict[str, Any] = {"type": "reasoning", "id": "rs_1", "summary": [], **fields}
    if text is not None:
        item["content"] = [{"type": "reasoning_text", "text": text}]
    if summary is not None:
        item["summary"] = [{"type": "summary_text", "text": summary}]
    return item


def said(text: str, **fields: Any) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        **fields,
    }


def called(call_id: str, name: str = "search", arguments: str = "{}", **fields: Any) -> dict:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments, **fields}


def result(call_id: str, output: Any = "ok") -> dict:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def tool_call(call_id: str, name: str = "search", arguments: str = "{}") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def sampled(message: dict, *, reason: str = "stop", key: str = "k1") -> Turn:
    return Turn(message=message, finish_reason=reason, model="policy", prompt_tokens=7, completion_tokens=3, key=key)


def refused(match: str, items: Any, **body: Any) -> None:
    with pytest.raises(RequestError, match=match):
        parse(items, **body)


# -- the request: what the body says beyond its items -----------------------------------
def test_a_string_input_is_one_user_turn() -> None:
    call = parse("hello")

    assert call.messages == [user("hello")]
    assert call.dialect is DialectType.OPENAI_RESPONSES
    assert call.model == "policy" and call.stream is False and call.tools is None


def test_instructions_are_a_leading_system_turn_and_an_empty_string_adds_none() -> None:
    assert turns("hi", instructions="Be brief.") == [{"role": "system", "content": "Be brief."}, user("hi")]
    assert turns("hi", instructions="") == [user("hi")]
    assert turns("hi", instructions=None) == [user("hi")]


@pytest.mark.parametrize(
    "body, match",
    [
        ([], "must be a JSON object"),
        ({"model": "policy"}, "`input` must be a string or a list"),
        ({"input": 3}, "`input` must be a string or a list"),
        ({"input": {"role": "user"}}, "`input` must be a string or a list"),
        ({"input": "hi", "instructions": ["a"]}, "`instructions` must be a string"),
    ],
)
def test_a_body_that_isnt_a_request_is_refused(body: Any, match: str) -> None:
    with pytest.raises(RequestError, match=match):
        RESPONSES.parse_request(body)


@pytest.mark.parametrize("field", ["previous_response_id", "conversation"])
def test_server_held_state_is_refused(field: str) -> None:
    refused(f"`{field}` is not supported", "hi", **{field: "resp_1"})


def test_background_is_refused_but_store_is_ignored() -> None:
    refused("`background` is not supported", "hi", background=True)
    assert turns("hi", background=False, store=True) == [user("hi")]


@pytest.mark.parametrize(
    "fields",
    [
        {"tool_choice": "required"},
        {"tool_choice": {"type": "function", "name": "search"}},
        {"tool_choice": "none"},
        {"parallel_tool_calls": False},
        {"text": {"format": {"type": "json_schema", "name": "x", "schema": {}}, "verbosity": "low"}},
        {"reasoning": {"effort": "none", "summary": "auto"}},
        {"include": ["reasoning.encrypted_content", "message.output_text.logprobs"], "top_logprobs": 5},
        {"metadata": {"a": "b"}, "service_tier": "auto", "prompt_cache_key": "p", "safety_identifier": "s"},
        {"client_metadata": {"x": 1}, "store": False, "stream_options": {"include_obfuscation": False}},
    ],
)
def test_fields_chat_ignores_and_bookkeeping_are_ignored(fields: dict) -> None:
    """A field whose chat counterpart the chat dialect ignores is ignored here too, and stays in ``body``."""
    call = parse("hi", **fields)

    assert call.messages == [user("hi")]
    assert all(call.body[name] == value for name, value in fields.items())
    assert RESPONSES.unsupported_in_token_mode(call) is None


def test_sampling_is_spelled_as_chat_and_max_output_tokens_is_max_tokens() -> None:
    call = parse("hi", temperature=0.6, top_p=0.9, top_k=20, max_output_tokens=64, seed=3)

    assert call.sampling == {"temperature": 0.6, "top_p": 0.9, "top_k": 20, "seed": 3, "max_tokens": 64}


def test_chat_spellings_of_max_tokens_are_not_responses_fields() -> None:
    assert parse("hi", max_tokens=5, max_completion_tokens=6).sampling == {}


def test_stream_and_model_are_read() -> None:
    call = parse("hi", stream=True)

    assert call.stream is True and call.model == "policy"


# -- tools -----------------------------------------------------------------------------
def test_function_tools_take_the_chat_spelling_in_order() -> None:
    tools = [
        {"type": "function", "name": "b", "description": "B", "parameters": {"type": "object"}, "strict": False},
        {"type": "function", "name": "a", "description": None, "parameters": None},
    ]

    assert parse("hi", tools=tools).tools == [
        {
            "type": "function",
            "function": {"name": "b", "description": "B", "parameters": {"type": "object"}, "strict": False},
        },
        {"type": "function", "function": {"name": "a"}},
    ]


def test_no_tools_is_none() -> None:
    assert parse("hi", tools=[]).tools is None
    assert parse("hi", tools=None).tools is None


@pytest.mark.parametrize("kind", ["custom", "web_search", "namespace", "local_shell", "file_search", "mcp", None])
def test_a_non_function_tool_is_refused_by_type_and_position(kind: str | None) -> None:
    tools = [{"type": "function", "name": "ok"}, {"type": kind, "name": "apply_patch"}]

    refused(rf"tool type {kind!r} \(`tools\[1\]`\) is not supported", "hi", tools=tools)


@pytest.mark.parametrize(
    "tools, match",
    [
        ({"type": "function"}, "`tools` must be a list"),
        ([{"type": "function"}], r"`tools\[0\].name` must be a string"),
        (["search"], "tool type None"),
    ],
)
def test_malformed_tools_are_refused(tools: Any, match: str) -> None:
    refused(match, "hi", tools=tools)


# -- items with a role -------------------------------------------------------------------
@pytest.mark.parametrize("role", ["user", "system", "developer"])
def test_a_role_item_is_a_turn_of_that_role_and_developer_is_kept(role: str) -> None:
    assert turns([{"type": "message", "role": role, "content": "x"}]) == [{"role": role, "content": "x"}]
    assert turns([{"role": role, "content": "x"}]) == [{"role": role, "content": "x"}]


def test_text_parts_are_joined_with_a_newline_and_item_metadata_is_dropped() -> None:
    item = {
        "type": "message",
        "id": "msg_9",
        "status": "completed",
        "role": "user",
        "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
    }

    assert turns([item]) == [user("a\nb")]


def test_an_empty_part_list_is_empty_text() -> None:
    assert turns([{"role": "user", "content": []}]) == [user("")]


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image", "image_url": "data:image/png;base64,AA"},
        {"type": "input_file", "file_id": "f"},
        {"type": "input_audio", "input_audio": {}},
        {"type": "input_text"},
        "plain",
    ],
)
def test_a_non_text_part_is_refused_by_position(part: Any) -> None:
    refused(r"`input\[0\]\.content\[1\]`", [{"role": "user", "content": [{"type": "input_text", "text": "a"}, part]}])


@pytest.mark.parametrize("role", ["tool", "function", "critic"])
def test_an_unknown_role_is_refused(role: str) -> None:
    refused(f"unsupported role '{role}'", [{"type": "message", "role": role, "content": "x"}])


def test_a_role_item_with_content_neither_string_nor_list_is_refused() -> None:
    refused("must be a string or a list of parts", [{"role": "user", "content": {"text": "x"}}])


# -- regrouping a model turn (vLLM's rule) --------------------------------------------
def test_reasoning_message_and_calls_are_one_assistant_turn() -> None:
    items = [user("q"), reasoning("think"), said("answer"), called("c1"), called("c2", "fetch", '{"u":1}')]

    assert turns(items) == [
        user("q"),
        {
            "role": "assistant",
            "reasoning_content": "think",
            "content": "answer",
            "tool_calls": [tool_call("c1"), tool_call("c2", "fetch", '{"u":1}')],
        },
    ]


def test_each_function_call_output_is_its_own_tool_turn() -> None:
    items = [user("q"), called("c1"), called("c2"), result("c1", "one"), result("c2", "two")]

    assert turns(items)[2:] == [
        {"role": "tool", "tool_call_id": "c1", "content": "one"},
        {"role": "tool", "tool_call_id": "c2", "content": "two"},
    ]


def test_a_second_message_starts_a_new_turn() -> None:
    assert turns([user("q"), said("a"), said("b")])[1:] == [
        {"role": "assistant", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_a_call_with_no_assistant_turn_before_it_starts_one_with_calls_only() -> None:
    assert turns([user("q"), called("c1")])[1] == {"role": "assistant", "tool_calls": [tool_call("c1")]}


def test_a_call_after_a_tool_result_starts_a_new_turn() -> None:
    items = [user("q"), called("c1"), result("c1"), called("c2")]

    assert turns(items)[1:] == [
        {"role": "assistant", "tool_calls": [tool_call("c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "tool_calls": [tool_call("c2")]},
    ]


def test_reasoning_after_a_message_fills_that_turns_empty_reasoning() -> None:
    """vLLM fills an empty field of the previous assistant turn, whatever the items' order."""
    assert turns([user("q"), said("a"), reasoning("r")])[1] == {
        "role": "assistant",
        "content": "a",
        "reasoning_content": "r",
    }


def test_a_message_after_a_call_fills_that_turns_empty_content() -> None:
    assert turns([user("q"), called("c1"), said("a")])[1] == {
        "role": "assistant",
        "tool_calls": [tool_call("c1")],
        "content": "a",
    }


def test_a_second_reasoning_item_starts_a_new_turn() -> None:
    assert turns([user("q"), reasoning("r1"), reasoning("r2")])[1:] == [
        {"role": "assistant", "reasoning_content": "r1"},
        {"role": "assistant", "reasoning_content": "r2"},
    ]


def test_a_message_after_a_full_turn_starts_a_new_one() -> None:
    assert turns([user("q"), reasoning("r"), said("a"), said("b")])[1:] == [
        {"role": "assistant", "reasoning_content": "r", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_an_assistant_item_after_a_user_turn_never_joins_it() -> None:
    assert turns([user("q"), reasoning("r")])[1] == {"role": "assistant", "reasoning_content": "r"}


def test_an_assistant_message_with_string_content_or_no_type_regroups_alike() -> None:
    plain = {"role": "assistant", "content": "a"}

    assert turns([user("q"), plain, called("c1")]) == turns([user("q"), said("a"), called("c1")])


# -- reasoning items --------------------------------------------------------------------
def test_reasoning_text_comes_from_content_before_summary() -> None:
    assert turns([user("q"), reasoning("from content", summary="from summary")])[1]["reasoning_content"] == (
        "from content"
    )


def test_reasoning_falls_back_to_the_summary_and_then_to_empty() -> None:
    assert turns([user("q"), reasoning(summary="s")])[1] == {"role": "assistant", "reasoning_content": "s"}
    empty = {"type": "reasoning", "content": [], "summary": []}
    assert turns([user("q"), empty])[1] == {"role": "assistant", "reasoning_content": ""}


def test_encrypted_reasoning_is_refused_and_a_null_blob_is_not() -> None:
    refused("encrypted reasoning is not supported", [user("q"), reasoning("r", encrypted_content="gAAA")])
    assert turns([user("q"), reasoning("r", encrypted_content=None)])[1]["reasoning_content"] == "r"


@pytest.mark.parametrize("field", ["content", "summary"])
def test_a_reasoning_item_with_two_parts_is_refused(field: str) -> None:
    item = reasoning()
    item[field] = [{"type": "reasoning_text", "text": "a"}, {"type": "reasoning_text", "text": "b"}]

    refused(rf"`input\[1\].{field}` must hold exactly one part", [user("q"), item])


def test_a_reasoning_part_without_text_is_refused() -> None:
    refused(r"`input\[1\].content\[0\].text` must be a string", [user("q"), reasoning(content=[{"type": "x"}])])


# -- assistant messages ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "content",
    [
        [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}],
        [],
        None,
    ],
)
def test_an_assistant_message_needs_exactly_one_text(content: Any) -> None:
    refused("must be a string or exactly one part", [user("q"), {"role": "assistant", "content": content}])


def test_a_refusal_part_is_refused() -> None:
    item = {"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}

    refused(r"of type 'refusal' is not supported", [user("q"), item])


# -- function calls and their outputs --------------------------------------------------------
@pytest.mark.parametrize("field", ["call_id", "name", "arguments"])
def test_a_function_call_needs_string_fields(field: str) -> None:
    item = called("c1")
    item[field] = {"q": 1} if field == "arguments" else None

    refused(rf"`input\[1\].{field}` must be a string", [user("q"), item])


def test_a_namespaced_call_is_refused() -> None:
    refused("namespaced tool calls are not supported", [user("q"), called("c1", namespace="mcp__fs")])


def test_tool_output_parts_are_joined_and_an_image_is_refused() -> None:
    parts = [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]

    assert turns([user("q"), called("c1"), result("c1", parts)])[2]["content"] == "a\nb"
    refused(
        r"`input\[2\].output.content\[0\]`",
        [user("q"), called("c1"), result("c1", [{"type": "input_image", "image_url": "x"}])],
    )


@pytest.mark.parametrize("output", [None, 3, {"text": "x"}])
def test_a_tool_output_that_isnt_text_is_refused(output: Any) -> None:
    refused("`input\\[2\\].output` must be a string", [user("q"), called("c1"), result("c1", output)])


def test_a_tool_output_without_a_call_id_is_refused() -> None:
    refused(r"`input\[1\].call_id` must be a string", [user("q"), {"type": "function_call_output", "output": "x"}])


# -- items skycap can't hold ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kind",
    [
        "item_reference",
        "custom_tool_call",
        "custom_tool_call_output",
        "local_shell_call",
        "web_search_call",
        "compaction",
        "image_generation_call",
    ],
)
def test_an_unsupported_item_type_is_refused_in_vllms_words(kind: str) -> None:
    refused(f"Unsupported input item type: {kind}", [user("q"), {"type": kind, "id": "x"}])


def test_an_item_that_isnt_an_object_is_refused_by_position() -> None:
    refused(r"`input\[1\]` must be an object", [user("q"), "said"])


# -- a trailing partial assistant item, which vLLM would continue ----------------------------------
@pytest.mark.parametrize("status", ["in_progress", "incomplete"])
@pytest.mark.parametrize("item", [said("half"), reasoning("half"), {"role": "assistant", "content": "half"}])
def test_a_trailing_partial_assistant_item_is_refused(item: dict, status: str) -> None:
    refused(f"status '{status}'", [user("q"), {**item, "status": status}])


def test_a_partial_item_earlier_in_the_history_is_read_as_finished() -> None:
    items = [user("q"), said("half", status="incomplete"), user("go on")]

    assert turns(items)[1] == {"role": "assistant", "content": "half"}


def test_a_trailing_finished_assistant_item_or_partial_user_item_is_fine() -> None:
    assert turns([user("q"), said("a", status="completed")])[1] == {"role": "assistant", "content": "a"}
    assert turns([{**user("q"), "status": "incomplete"}]) == [user("q")]


# -- token-mode refusals ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fields, refused_here",
    [
        ({"truncation": "auto"}, True),
        ({"truncation": "disabled"}, False),
        ({"context_management": [{"type": "compaction", "compact_threshold": 1000}]}, True),
        ({"context_management": []}, False),
        ({}, False),
    ],
)
def test_only_server_side_truncation_and_compaction_are_refused_in_token_mode(fields: dict, refused_here: bool) -> None:
    assert (RESPONSES.unsupported_in_token_mode(parse("hi", **fields)) is not None) is refused_here


# -- a sampled turn as items -------------------------------------------------------------------------
def test_a_turn_is_reasoning_message_then_calls_with_ids_from_its_key() -> None:
    message = {
        "role": "assistant",
        "reasoning_content": "r",
        "content": "a",
        "tool_calls": [tool_call("call_x"), tool_call("call_y", "fetch", '{"u":1}')],
    }

    assert output_items(sampled(message)) == [
        {
            "type": "reasoning",
            "id": "rs_k1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "r"}],
            "encrypted_content": None,
        },
        {
            "type": "message",
            "id": "msg_k1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "a", "annotations": []}],
        },
        {
            "type": "function_call",
            "id": "fc_k1_0",
            "call_id": "call_x",
            "name": "search",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call",
            "id": "fc_k1_1",
            "call_id": "call_y",
            "name": "fetch",
            "arguments": '{"u":1}',
            "status": "completed",
        },
    ]


@pytest.mark.parametrize(
    "message, types",
    [
        ({"role": "assistant", "content": ""}, ["message"]),
        ({"role": "assistant", "content": None}, ["message"]),
        ({"role": "assistant"}, ["message"]),
        ({"role": "assistant", "content": "", "reasoning_content": "r"}, ["reasoning"]),
        ({"role": "assistant", "content": "", "tool_calls": [tool_call("c")]}, ["function_call"]),
        ({"role": "assistant", "content": "a", "reasoning_content": ""}, ["message"]),
        (
            {"role": "assistant", "content": None, "reasoning_content": "r", "tool_calls": [tool_call("c")]},
            ["reasoning", "function_call"],
        ),
    ],
)
def test_empty_fields_get_no_item_but_an_empty_turn_gets_one_empty_message(message: dict, types: list) -> None:
    items = output_items(sampled(message))

    assert [item["type"] for item in items] == types
    if types == ["message"] and not message.get("content"):
        assert items[0]["content"][0]["text"] == ""


def test_a_call_with_empty_arguments_still_has_an_arguments_string() -> None:
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": None}}

    assert output_items(sampled({"role": "assistant", "tool_calls": [call]}))[0]["arguments"] == ""


@pytest.mark.parametrize(
    "reason, status, details",
    [
        ("stop", "completed", None),
        ("tool_calls", "completed", None),
        ("length", "incomplete", {"reason": "max_output_tokens"}),
    ],
)
def test_status_follows_the_finish_reason(reason: str, status: str, details: dict | None) -> None:
    body = RESPONSES.reply(parse("q"), sampled(user_reply(), reason=reason))

    assert body["status"] == status
    assert body["incomplete_details"] == details


def user_reply() -> dict:
    return {"role": "assistant", "content": "a"}


def test_the_reply_echoes_the_request_fields_as_sent() -> None:
    tools = [{"type": "function", "name": "f", "parameters": {}}]
    call = parse(
        "q",
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        instructions="I",
        metadata={"m": 1},
        temperature=0.5,
        top_p=0.8,
        max_output_tokens=9,
        truncation="disabled",
        reasoning={"effort": "low"},
    )

    body = RESPONSES.reply(call, sampled(user_reply()))

    assert {name: body[name] for name in ("tools", "tool_choice", "parallel_tool_calls", "instructions")} == {
        "tools": tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "instructions": "I",
    }
    assert (body["metadata"], body["temperature"], body["top_p"], body["max_output_tokens"]) == ({"m": 1}, 0.5, 0.8, 9)
    assert body["reasoning"] == {"effort": "low"} and body["truncation"] == "disabled"


def test_the_reply_defaults_what_the_request_left_out() -> None:
    body = RESPONSES.reply(parse("q"), sampled(user_reply()))

    assert body["tools"] == [] and body["tool_choice"] == "auto" and body["parallel_tool_calls"] is True
    assert body["text"] == {"format": {"type": "text"}} and body["truncation"] == "disabled"
    assert body["previous_response_id"] is None and body["error"] is None
    assert body["object"] == "response" and body["id"] == "resp_k1" and body["model"] == "policy"


def test_usage_counts_the_turns_tokens() -> None:
    usage = RESPONSES.reply(parse("q"), sampled(user_reply()))["usage"]

    assert usage["input_tokens"] == 7 and usage["output_tokens"] == 3 and usage["total_tokens"] == 10


# -- the round trip: what the client sends back is what was stored ---------------------------------------
SHAPES = {
    "text": {"role": "assistant", "content": "a"},
    "reasoning and text": {"role": "assistant", "content": "a", "reasoning_content": "r"},
    "reasoning and two calls, empty content": {
        "role": "assistant",
        "content": "",
        "reasoning_content": "r",
        "tool_calls": [tool_call("c1"), tool_call("c2", "fetch", "{}")],
    },
    "calls, no content": {"role": "assistant", "content": None, "tool_calls": [tool_call("c1")]},
    "text and a call": {"role": "assistant", "content": "a", "tool_calls": [tool_call("c1")]},
    "empty": {"role": "assistant", "content": ""},
    "reasoning only": {"role": "assistant", "content": "", "reasoning_content": "r"},
    "empty reasoning and text": {"role": "assistant", "content": "a", "reasoning_content": ""},
}


def _codex_replay(item: dict) -> dict:
    """The item as a client may send it back: without the server's ids and statuses."""
    return {key: value for key, value in item.items() if key not in ("id", "status")}


@pytest.mark.parametrize("message", SHAPES.values(), ids=SHAPES.keys())
@pytest.mark.parametrize("strip", [False, True], ids=["verbatim", "ids and statuses dropped"])
def test_a_replayed_reply_regroups_into_the_stored_turn(message: dict, strip: bool) -> None:
    turn = sampled(message)
    stored = RESPONSES.replayed(turn)
    items = RESPONSES.reply(parse("q"), turn)["output"]
    if strip:
        items = [_codex_replay(item) for item in items]

    history = turns([user("q"), *items, user("next")])

    assert history == [user("q"), stored, user("next")]
    assert KEY(history[1]) == KEY(stored)


def test_replayed_differs_from_the_parsed_turn_exactly_where_the_wire_cant_say_it() -> None:
    """``content: ""`` beside calls has no message item, so the replay has no ``content`` and hashes apart."""
    message = SHAPES["reasoning and two calls, empty content"]
    stored = RESPONSES.replayed(sampled(message))

    assert "content" not in stored
    assert KEY(stored) != KEY(message)
    assert {k: v for k, v in stored.items()} == {k: v for k, v in message.items() if k != "content"}


def test_replayed_does_not_change_the_sampled_turn() -> None:
    message = copy.deepcopy(SHAPES["reasoning and two calls, empty content"])
    RESPONSES.replayed(sampled(message))

    assert message == SHAPES["reasoning and two calls, empty content"]


def test_a_client_that_drops_the_reasoning_item_sends_a_different_turn() -> None:
    """Not repaired: an edited history is a fork, and the record shows it."""
    turn = sampled(SHAPES["reasoning and text"])
    items = [item for item in RESPONSES.reply(parse("q"), turn)["output"] if item["type"] != "reasoning"]

    assert KEY(turns([user("q"), *items])[1]) != KEY(RESPONSES.replayed(turn))


def test_chat_replays_the_parsed_turn_itself() -> None:
    message = SHAPES["reasoning and two calls, empty content"]

    assert ChatDialect().replayed(sampled(message)) is message


# -- the stream -----------------------------------------------------------------------------------------
def frames(body: bytes) -> list[tuple[str, dict]]:
    out = []
    for frame in body.split(b"\n\n"):
        if not frame:
            continue
        event, data = frame.split(b"\n")
        assert event.startswith(b"event: ") and data.startswith(b"data: ")
        payload = orjson.loads(data[6:])
        assert payload["type"] == event[7:].decode()
        out.append((payload["type"], payload))
    return out


def stream_of(message: dict, **options: Any) -> list[tuple[str, dict]]:
    reply = RESPONSES.reply(parse("q", stream=True), sampled(message, **options))
    return frames(b"".join(RESPONSES.stream(reply)))


def test_the_stream_opens_fills_and_closes_each_item_in_order() -> None:
    events = stream_of(SHAPES["reasoning and text"] | {"tool_calls": [tool_call("c1")]})

    assert [kind for kind, _ in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [payload["sequence_number"] for _, payload in events] == list(range(len(events)))


def test_the_stream_closes_each_item_as_the_reply_has_it_and_ends_with_the_reply() -> None:
    message = SHAPES["reasoning and two calls, empty content"]
    reply = RESPONSES.reply(parse("q", stream=True), sampled(message))
    events = frames(b"".join(RESPONSES.stream(reply)))

    done = [payload["item"] for kind, payload in events if kind == "response.output_item.done"]
    assert done == reply["output"]
    assert events[-1][1]["response"] == reply
    assert events[0][1]["response"]["status"] == "in_progress" and events[0][1]["response"]["output"] == []


def test_an_item_is_announced_empty() -> None:
    added = [p["item"] for k, p in stream_of(SHAPES["text and a call"]) if k == "response.output_item.added"]

    assert added[0]["content"] == [] and added[0]["status"] == "in_progress"
    assert added[1]["arguments"] == "" and added[1]["status"] == "in_progress"


def test_empty_text_and_arguments_send_no_delta() -> None:
    call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": ""}}
    kinds = [kind for kind, _ in stream_of({"role": "assistant", "content": "", "tool_calls": [call]})]
    kinds += [kind for kind, _ in stream_of(SHAPES["empty"])]

    assert "response.output_text.delta" not in kinds and "response.function_call_arguments.delta" not in kinds
    assert "response.output_text.done" in kinds and "response.function_call_arguments.done" in kinds


def test_a_cut_off_stream_ends_incomplete() -> None:
    events = stream_of(user_reply(), reason="length")

    assert events[-1][0] == "response.incomplete"
    assert events[-1][1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


# -- errors -----------------------------------------------------------------------------------------------
def test_an_overflow_on_a_streamed_call_is_a_failed_response_in_the_stream() -> None:
    response = RESPONSES.error(parse("q", stream=True), "too long", 400, code="context_length_exceeded")

    assert response.status == 200 and response.content_type == "text/event-stream"
    events = frames(response.body)
    assert [kind for kind, _ in events] == ["response.created", "response.failed"]
    assert events[1][1]["response"]["error"] == {"code": "context_length_exceeded", "message": "too long"}
    assert events[1][1]["response"]["status"] == "failed"


@pytest.mark.parametrize(
    "stream, code",
    [(False, "context_length_exceeded"), (True, None), (True, "trajectory_closed")],
)
def test_other_errors_are_http_errors(stream: bool, code: str | None) -> None:
    response = RESPONSES.error(parse("q", stream=stream), "no", 400, code=code)

    assert response.status == 400
    assert orjson.loads(response.body)["error"] == {
        "message": "no",
        "type": "invalid_request_error",
        "param": None,
        "code": code,
    }


def test_an_error_without_a_call_is_an_http_error() -> None:
    assert RESPONSES.error(None, "no", 410, code="context_length_exceeded").status == 410


# -- text mode: reading an upstream's reply -------------------------------------------------------------
def response_object(output: list, status: str = "completed", reason: str | None = None, **fields: Any) -> dict:
    return {
        "id": "resp_up",
        "object": "response",
        "status": status,
        "output": output,
        "incomplete_details": {"reason": reason} if reason else None,
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
        **fields,
    }


def test_a_completed_reply_is_one_turn_with_chat_usage() -> None:
    reply = RESPONSES.parse_reply(response_object([reasoning("r"), said("a")]))

    assert reply.message == {"role": "assistant", "reasoning_content": "r", "content": "a"}
    assert reply.finish_reason == "stop"
    assert reply.usage == {"prompt_tokens": 11, "completion_tokens": 4}


def test_a_reply_with_calls_finishes_with_tool_calls() -> None:
    assert RESPONSES.parse_reply(response_object([called("c1")])).finish_reason == "tool_calls"


def test_a_max_output_tokens_cut_off_is_length() -> None:
    assert RESPONSES.parse_reply(response_object([said("a")], "incomplete", "max_output_tokens")).finish_reason == (
        "length"
    )


@pytest.mark.parametrize(
    "body, match",
    [
        (response_object([said("a")], "incomplete", "content_filter"), "incomplete"),
        (response_object([], "failed"), "status is 'failed'"),
        (response_object([], "in_progress"), "status is 'in_progress'"),
        (response_object([{"type": "web_search_call", "id": "w"}]), "'web_search_call' can't be recorded"),
        (response_object([said("a"), said("b")]), "2 turns"),
        (response_object(None), "no output"),
        ([], "not a JSON object"),
    ],
)
def test_a_reply_that_isnt_one_recordable_turn_is_an_error(body: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RESPONSES.parse_reply(body)


def test_an_empty_output_is_an_empty_turn() -> None:
    assert RESPONSES.parse_reply(response_object([])).message == {"role": "assistant", "content": ""}


def test_a_reply_without_usage_records_none() -> None:
    assert RESPONSES.parse_reply(response_object([said("a")], usage=None)).usage is None


def sse(*payloads: dict, separator: bytes = b"\n\n") -> bytes:
    return b"".join(
        b"event: " + payload["type"].encode() + b"\ndata: " + orjson.dumps(payload) + separator for payload in payloads
    )


UPSTREAM_STREAM = (
    {"type": "response.created", "response": response_object([], "in_progress")},
    {"type": "response.output_item.added", "output_index": 0, "item": said("")},
    {"type": "response.output_text.delta", "output_index": 0, "delta": "a"},
    {"type": "response.output_item.done", "output_index": 0, "item": said("a")},
    {"type": "response.output_item.done", "output_index": 1, "item": called("c1")},
    {"type": "response.completed", "response": response_object([said("a"), called("c1")])},
)


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\n\r\n"])
@pytest.mark.parametrize("size", [1, 7, 10_000])
def test_the_assembler_reads_a_stream_in_any_chunking(size: int, separator: bytes) -> None:
    data = sse(*UPSTREAM_STREAM, separator=separator)
    assembler = ResponsesAssembler()
    events = []
    for start in range(0, len(data), size):
        events += assembler.feed(data[start : start + size])
    assert assembler.flush() == b""

    assert b"".join(raw for raw, _ in events) == data
    # A chunk that ends on the CR of a CRLF blank line ends the event at once; its LF is an empty event.
    assert [ending for raw, ending in events if raw.strip()] == [False] * 5 + [True]
    assert assembler.done
    assert assembler.reply().message == {"role": "assistant", "content": "a", "tool_calls": [tool_call("c1")]}


def test_the_assembler_rebuilds_the_output_from_closed_items_when_the_final_event_leaves_it_out() -> None:
    assembler = ResponsesAssembler()
    assembler.feed(sse(*UPSTREAM_STREAM[:-1], {"type": "response.completed", "response": response_object([])}))

    assert assembler.reply().message == {"role": "assistant", "content": "a", "tool_calls": [tool_call("c1")]}


def test_a_stream_without_a_final_event_or_with_an_error_event_is_not_a_reply() -> None:
    cut = ResponsesAssembler()
    cut.feed(sse(*UPSTREAM_STREAM[:-1]))
    with pytest.raises(ValueError, match="without a final response event"):
        cut.reply()

    errored = ResponsesAssembler()
    errored.feed(sse(UPSTREAM_STREAM[0], {"type": "error", "code": "server_error", "message": "x"}))
    assert errored.ending
    with pytest.raises(ValueError, match="status is 'failed'"):
        errored.reply()


def test_the_assembler_reads_a_last_event_with_no_separator_on_flush() -> None:
    data = sse(*UPSTREAM_STREAM)[:-2]
    assembler = ResponsesAssembler()
    assembler.feed(data)

    assert not assembler.ending
    assert assembler.flush().endswith(b"}")
    assert assembler.ending


# The API ends a cut-off stream with `response.incomplete`; vLLM (0.31) with `response.completed` whose
# response says `status: "incomplete"`. Either is read as the cut-off it is, and relayed as sent.
@pytest.mark.parametrize("kind", ["response.incomplete", "response.completed"])
def test_a_cut_off_stream_is_read_as_length_whichever_final_event_carries_it(kind: str) -> None:
    assembler = ResponsesAssembler()
    final = {"type": kind, "response": response_object([said("a")], "incomplete", "max_output_tokens")}
    events = assembler.feed(sse(final))

    assert [ending for _, ending in events] == [True]
    reply = assembler.reply()
    assert reply.finish_reason == "length" and reply.message == {"role": "assistant", "content": "a"}


# -- the dialects themselves ---------------------------------------------------------------------------------
def test_each_type_has_one_stateless_instance_at_its_path() -> None:
    assert DialectType.OPENAI_CHAT.get_dialect() is DialectType.OPENAI_CHAT.get_dialect()
    assert isinstance(DialectType.OPENAI_RESPONSES.get_dialect(), ResponsesDialect)
    assert [(t.value, t.get_dialect().path) for t in DialectType] == [
        ("openai_chat", "/chat/completions"),
        ("openai_responses", "/responses"),
    ]
    assert all(t.get_dialect().type is t for t in DialectType)


@pytest.mark.parametrize(
    "headers, key",
    [
        ({"Authorization": "Bearer abc"}, "abc"),
        ({"Authorization": "bearer   abc  "}, "abc"),
        ({"Authorization": "Basic abc"}, None),
        ({"Authorization": "Bearer"}, None),
        ({"x-api-key": "abc"}, None),
        ({}, None),
    ],
)
@pytest.mark.parametrize("dialect", list(DialectType))
def test_openai_dialects_read_the_key_from_a_bearer_header(
    dialect: DialectType, headers: dict, key: str | None
) -> None:
    assert dialect.get_dialect().api_key(headers) == key


def test_the_chat_dialect_marks_a_raw_reply_with_null_reasoning_and_responses_takes_none() -> None:
    chat = DialectType.OPENAI_CHAT.get_dialect()
    turn = sampled({"role": "assistant", "content": "<think>x</think>a"})
    turn.raw_content = True

    body = chat.reply(chat.parse_request({"messages": [user("q")]}), turn)

    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "<think>x</think>a",
        "reasoning_content": None,
    }
    assert chat.raw_content is True and RESPONSES.raw_content is False


# -- SSE framing (dialects.base): line endings and multi-line data, for both dialects -----------------------------

#: Every blank-line separator the SSE spec allows: a line's ending, then the blank line's.
SEPARATORS = [b"\n\n", b"\r\n\r\n", b"\r\r", b"\n\r\n", b"\r\n\n", b"\r\n\r", b"\n\r", b"\r\r\n"]


def framed(*payloads: dict, separators: list[bytes], eol: bytes = b"\n") -> bytes:
    """One event per payload, its separator taken in turn from ``separators``."""
    return b"".join(
        b"event: "
        + payload["type"].encode()
        + eol
        + b"data: "
        + orjson.dumps(payload)
        + separators[i % len(separators)]
        for i, payload in enumerate(payloads)
    )


def feed_all(assembler: Any, data: bytes, size: int) -> list[tuple[bytes, bool]]:
    events = []
    for start in range(0, len(data), size):
        events += assembler.feed(data[start : start + size])
    return events


@pytest.mark.parametrize("size", [1, 2, 3, 5, 10_000])
@pytest.mark.parametrize(
    "separators, eol",
    [([separator], separator[:1] if separator != b"\r\n\r\n" else b"\r\n") for separator in SEPARATORS]
    + [(SEPARATORS, b"\r")],
    ids=[repr(s) for s in SEPARATORS] + ["mixed"],
)
def test_every_line_ending_the_spec_allows_splits_events_in_any_chunking(
    separators: list[bytes], eol: bytes, size: int
) -> None:
    data = framed(*UPSTREAM_STREAM, separators=separators, eol=eol)
    assembler = ResponsesAssembler()
    events = feed_all(assembler, data, size)
    rest = assembler.flush()

    assert b"".join(raw for raw, _ in events) + rest == data
    endings = [ending for raw, ending in events if sse_data(raw) is not None]
    assert endings == [False] * 5 + [True]
    assert assembler.reply().message == {"role": "assistant", "content": "a", "tool_calls": [tool_call("c1")]}


@pytest.mark.parametrize("separator", [b"\r\r", b"\n\r", b"\r\n\r"])
def test_a_final_event_ending_in_a_bare_cr_is_read_without_waiting_for_more(separator: bytes) -> None:
    """The blank line has begun, so the event is over whether a LF follows or not: an upstream that keeps the
    connection open after it must not stall the relay."""
    assembler = ResponsesAssembler()
    events = assembler.feed(framed(UPSTREAM_STREAM[-1], separators=[separator]))

    assert len(events) == 1 and events[0][1] and assembler.done


def test_a_crlf_split_across_chunks_does_not_end_the_event_early() -> None:
    assembler = ResponsesAssembler()
    payload = b"data: " + orjson.dumps(UPSTREAM_STREAM[-1])

    assert assembler.feed(payload + b"\r") == []
    events = assembler.feed(b"\n\r\n")

    assert [raw for raw, _ in events] == [payload + b"\r\n\r\n"] and assembler.done


def multi_line(payload: dict, eol: bytes = b"\n") -> bytes:
    """``payload`` pretty-printed over several data lines, as the SSE spec allows."""
    text = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    return (
        b"event: "
        + payload["type"].encode()
        + eol
        + b"".join(b"data: " + line + eol for line in text.splitlines())
        + eol
    )


CUT_OFF_EVENT = {
    "type": "response.incomplete",
    "response": response_object([said("a")], "incomplete", "max_output_tokens"),
}


@pytest.mark.parametrize("eol", [b"\n", b"\r\n", b"\r"])
def test_data_split_over_lines_is_joined_before_it_is_decoded(eol: bytes) -> None:
    assembler = ResponsesAssembler()
    assembler.feed(multi_line(CUT_OFF_EVENT, eol))

    reply = assembler.reply()
    assert reply.finish_reason == "length" and reply.message == {"role": "assistant", "content": "a"}


@pytest.mark.parametrize(
    "event, data",
    [
        (b"data: a\ndata: b", b"a\nb"),
        (b"data:a", b"a"),
        (b"data:  two", b" two"),
        (b"data", b""),
        (b": keepalive\ndata: x", b"x"),
        (b"event: e\rid: 1\rdata: x\r\ndata: y", b"x\ny"),
        (b": only a comment", None),
        (b"event: e", None),
        (b"", None),
    ],
)
def test_sse_data_reads_an_event_as_the_spec_does(event: bytes, data: bytes | None) -> None:
    assert sse_data(event) == data


@pytest.mark.parametrize("separator", [b"\n\n", b"\r\r", b"\r\n\r\n"])
def test_the_chat_assembler_reads_bare_cr_streams_and_multi_line_data(separator: bytes) -> None:
    chunk = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}
    text = orjson.dumps(chunk, option=orjson.OPT_INDENT_2)
    eol = separator[: len(separator) // 2]
    data = b"".join(b"data: " + line + eol for line in text.splitlines()) + eol + b"data: [DONE]" + separator
    assembler = StreamAssembler()
    events = feed_all(assembler, data, 3)

    assert b"".join(raw for raw, _ in events) == data
    assert assembler.done
    assert assembler.reply().message == {"role": "assistant", "content": "hi"}
