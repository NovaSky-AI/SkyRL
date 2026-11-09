"""Distinguish extracted arguments from failed tool-call parsing."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")

from renderers import ParsedResponse, ParsedToolCall, ToolCallParseStatus  # noqa: E402

from skycap.tokens.backend import finish_reason  # noqa: E402
from skycap.tokens.renderer import RenderersRenderer, tool_call_id  # noqa: E402


def parse(*calls: ParsedToolCall, content: str = "") -> dict:
    """Exercise the adapter with the parser's structured output, without a tokenizer download."""
    parsed = ParsedResponse(content=content, reasoning_content="hmm", tool_calls=list(calls))
    renderer = RenderersRenderer.__new__(RenderersRenderer)
    renderer._slots = queue.Queue()
    renderer._slots.put((SimpleNamespace(parse_response=lambda *args, **kwargs: parsed), None))
    return renderer.parse([1, 2, 3], None)


def test_extracted_argument_object_preserves_schema_invalid_value() -> None:
    arguments = {"volume": "120 plus the remaining volume"}
    call = ParsedToolCall(
        raw="original call", name="run_well", arguments=arguments, status=ToolCallParseStatus.INVALID_JSON
    )

    message = parse(call)

    assert message["content"] == ""
    assert message["reasoning_content"] == "hmm"
    (returned,) = message["tool_calls"]
    assert returned["function"]["name"] == "run_well"
    encoded = returned["function"]["arguments"]
    assert json.loads(encoded) == arguments
    assert finish_reason("stop", message) == "tool_calls"
    assert finish_reason("length", message) == "length"


@pytest.mark.parametrize("arguments", [[], False, 0, "", '{"volume":'])
def test_invalid_argument_shapes_are_not_replaced_with_an_empty_object(arguments: object) -> None:
    message = parse(ParsedToolCall(raw="original call", name="run_well", arguments=arguments))

    encoded = message["tool_calls"][0]["function"]["arguments"]
    assert encoded == (arguments if isinstance(arguments, str) else json.dumps(arguments))


def test_missing_arguments_are_an_empty_object() -> None:
    # Harnesses and chat templates index into the decoded arguments, so `null` would break them.
    message = parse(ParsedToolCall(raw="original call", name="run_well", arguments=None))

    assert message["tool_calls"][0]["function"]["arguments"] == "{}"


def test_parser_rejected_unknown_tool_is_not_promoted() -> None:
    message = parse(
        ParsedToolCall(raw="missing_tool", name="missing_tool", arguments={}, status=ToolCallParseStatus.UNKNOWN_TOOL)
    )

    assert message["content"] == "missing_tool"
    assert "tool_calls" not in message
    assert finish_reason("stop", message) == "stop"


def test_adapter_does_not_add_tool_name_validation() -> None:
    # Some model parsers accept unadvertised names. Keep their decision, too.
    message = parse(ParsedToolCall(raw="missing_tool", name="missing_tool", arguments={}))

    assert message["tool_calls"][0]["function"] == {"name": "missing_tool", "arguments": "{}"}


@pytest.mark.parametrize("arguments", ['{"volume":', None, [], False, 0, ""])
def test_failed_argument_parse_is_text_even_with_a_name(arguments: object) -> None:
    raw = '<tool_call>run_well {"volume":</tool_call>'
    message = parse(
        ParsedToolCall(raw=raw, name="run_well", arguments=arguments, status=ToolCallParseStatus.INVALID_JSON)
    )

    assert message["content"] == raw
    assert "tool_calls" not in message
    assert finish_reason("stop", message) == "stop"


@pytest.mark.parametrize("status", [ToolCallParseStatus.UNCLOSED_BLOCK, ToolCallParseStatus.MALFORMED_STRUCTURE])
def test_broken_structure_does_not_become_an_executable_partial_call(status: ToolCallParseStatus) -> None:
    raw = '<invoke name="run_well"><parameter name="volume">120</parameter><broken>'
    message = parse(ParsedToolCall(raw=raw, name="run_well", arguments={"volume": 120}, status=status))

    assert message["content"] == raw
    assert "tool_calls" not in message
    assert finish_reason("stop", message) == "stop"


@pytest.mark.parametrize(
    "status",
    [ToolCallParseStatus.INVALID_JSON, ToolCallParseStatus.MISSING_NAME, ToolCallParseStatus.UNCLOSED_BLOCK],
)
def test_unidentified_call_is_preserved_as_text(status: ToolCallParseStatus) -> None:
    raw = '{"arguments": {"volume": 120}}'
    message = parse(ParsedToolCall(raw=raw, status=status), content="Attempting a well.")

    assert message["content"] == f"Attempting a well.\n{raw}"
    assert message["reasoning_content"] == "hmm"
    assert "tool_calls" not in message


def test_mixed_attempts_keep_call_order_and_stable_ids() -> None:
    calls = [
        ParsedToolCall(raw="first", name="run_well", arguments={"volume": 120}),
        ParsedToolCall(raw="broken framing", status=ToolCallParseStatus.MALFORMED_STRUCTURE),
        ParsedToolCall(
            raw="third", name="run_well", arguments={"volume": "bad number"}, status=ToolCallParseStatus.INVALID_JSON
        ),
        ParsedToolCall(raw="fourth", name="run_well", arguments={}, id="native_id"),
        ParsedToolCall(raw="unknown", name="missing_tool", arguments={}, status=ToolCallParseStatus.UNKNOWN_TOOL),
    ]

    message = parse(*calls)

    assert message == parse(*calls)
    assert message["content"] == "broken framing\nunknown"
    assert [call["id"] for call in message["tool_calls"]] == [
        tool_call_id([1, 2, 3], 0),
        tool_call_id([1, 2, 3], 2),
        "native_id",
    ]
