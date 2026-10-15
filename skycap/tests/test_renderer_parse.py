"""The chat adapter preserves failed tool attempts for harness validation."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")

from renderers import ParsedResponse, ParsedToolCall, ToolCallParseStatus  # noqa: E402

from skycap.tokens.renderer import RenderersRenderer, tool_call_id  # noqa: E402


def parse(*calls: ParsedToolCall, content: str = "") -> dict:
    """Exercise the adapter with the parser's structured output, without a tokenizer download."""
    parsed = ParsedResponse(content=content, reasoning_content="hmm", tool_calls=list(calls))
    renderer = RenderersRenderer.__new__(RenderersRenderer)
    renderer._slots = queue.Queue()
    renderer._slots.put((SimpleNamespace(parse_response=lambda *args, **kwargs: parsed), None))
    return renderer.parse([1, 2, 3], None)


@pytest.mark.parametrize("arguments", [{"volume": "120 plus the remaining volume"}, '{"volume":'])
def test_invalid_arguments_reach_the_harness_unchanged(arguments: dict | str) -> None:
    call = ParsedToolCall(
        raw="original call", name="run_well", arguments=arguments, status=ToolCallParseStatus.INVALID_JSON
    )

    message = parse(call)

    assert message["content"] == ""
    assert message["reasoning_content"] == "hmm"
    (returned,) = message["tool_calls"]
    assert returned["function"]["name"] == "run_well"
    encoded = returned["function"]["arguments"]
    assert (json.loads(encoded) if isinstance(arguments, dict) else encoded) == arguments


@pytest.mark.parametrize("arguments", [None, [], False, 0, ""])
def test_invalid_argument_shapes_are_not_replaced_with_an_empty_object(arguments: object) -> None:
    message = parse(ParsedToolCall(raw="original call", name="run_well", arguments=arguments))

    encoded = message["tool_calls"][0]["function"]["arguments"]
    assert encoded == (arguments if isinstance(arguments, str) else json.dumps(arguments))


def test_unknown_tool_names_reach_the_harness() -> None:
    message = parse(
        ParsedToolCall(raw="missing_tool", name="missing_tool", arguments={}, status=ToolCallParseStatus.UNKNOWN_TOOL)
    )

    assert message["tool_calls"][0]["function"] == {"name": "missing_tool", "arguments": "{}"}


@pytest.mark.parametrize("status", [ToolCallParseStatus.UNCLOSED_BLOCK, ToolCallParseStatus.MALFORMED_STRUCTURE])
def test_broken_structure_does_not_become_an_executable_partial_call(status: ToolCallParseStatus) -> None:
    raw = '<invoke name="run_well"><parameter name="volume">120</parameter><broken>'
    message = parse(ParsedToolCall(raw=raw, name="run_well", arguments={"volume": 120}, status=status))

    assert message["content"] == raw
    assert "tool_calls" not in message


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
        ParsedToolCall(raw="third", name="run_well", arguments='{"volume":', status=ToolCallParseStatus.INVALID_JSON),
        ParsedToolCall(raw="fourth", name="run_well", arguments={}, id="native_id"),
    ]

    message = parse(*calls)

    assert message == parse(*calls)
    assert message["content"] == "broken framing"
    assert [call["id"] for call in message["tool_calls"]] == [
        tool_call_id([1, 2, 3], 0),
        tool_call_id([1, 2, 3], 2),
        "native_id",
    ]
