"""The renderers-backed renderer, against a stand-in for the library.

The real ``renderers`` package is an optional extra pinned to a git revision,
so the suite must not need it installed. These tests inject a module with the
same surface and check the mapping in both directions: what we ask the library
for, and what we make of what it returns.

What is worth pinning here is the behaviour a bare chat template does not have
-- reasoning content carried on the message, tool calls parsed into structure,
malformed attempts left alone, and a bridge whose reused prefix is verified
rather than trusted.
"""

from __future__ import annotations

import enum
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from skyrl_capture.tito.renderer import build_renderer


class FakeStatus(enum.Enum):
    OK = "ok"
    INVALID_JSON = "invalid_json"


@dataclass
class FakeToolCall:
    raw: str
    name: str | None = None
    arguments: Any = None
    id: str | None = None
    status: FakeStatus = FakeStatus.OK


@dataclass
class FakeParsed:
    content: str
    reasoning_content: str | None = None
    tool_calls: list[FakeToolCall] = field(default_factory=list)


@dataclass
class FakeRendered:
    token_ids: list[int]
    message_indices: list[int]


class FakeRenderer:
    """Stands in for one checked-out renderer from the pool."""

    def __init__(self, recorder: dict[str, Any]) -> None:
        self._recorder = recorder
        self.bridge_result: FakeRendered | None = None

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        self._recorder["render"] = {
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": add_generation_prompt,
        }
        return FakeRendered(token_ids=[1, 2, 3], message_indices=[0, 0, -1])

    def bridge_to_next_turn(self, prompt_ids, completion_ids, new_messages, *, tools=None):
        self._recorder["bridge"] = {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "new_messages": new_messages,
        }
        return self.bridge_result

    def parse_response(self, token_ids, *, tools=None):
        return self._recorder["parsed"]

    def get_stop_token_ids(self):
        return [99, 100]


class FakePool:
    def __init__(self, renderer: FakeRenderer) -> None:
        self._renderer = renderer

    class _Checkout:
        def __init__(self, renderer: FakeRenderer) -> None:
            self._renderer = renderer

        def __enter__(self) -> FakeRenderer:
            return self._renderer

        def __exit__(self, *exc: Any) -> None:
            return None

    def checkout(self) -> _Checkout:
        return self._Checkout(self._renderer)

    def render(self, *args: Any, **kwargs: Any):
        return self._renderer.render(*args, **kwargs)

    def parse_response(self, *args: Any, **kwargs: Any):
        return self._renderer.parse_response(*args, **kwargs)

    def get_stop_token_ids(self):
        return self._renderer.get_stop_token_ids()


@pytest.fixture
def prime(monkeypatch):
    """Install a fake ``renderers`` module and return (renderer, recorder)."""
    recorder: dict[str, Any] = {}
    fake_renderer = FakeRenderer(recorder)
    pool = FakePool(fake_renderer)

    module = types.ModuleType("renderers")
    base = types.ModuleType("renderers.base")
    base.ToolCallParseStatus = FakeStatus

    def create_renderer_pool(name, config=None, *, size=16, chat_template_kwargs=None):
        recorder["pool"] = {"name": name, "config": config, "size": size}
        return pool

    class AutoRendererConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    module.create_renderer_pool = create_renderer_pool
    module.AutoRendererConfig = AutoRendererConfig
    module.base = base
    monkeypatch.setitem(sys.modules, "renderers", module)
    monkeypatch.setitem(sys.modules, "renderers.base", base)

    built = build_renderer(tokenizer="Qwen/Qwen3-8B", model="qwen3")
    return built, recorder, fake_renderer


def test_any_tokenizer_name_selects_the_library_and_keeps_thinking(prime):
    renderer, recorder, _ = prime
    # A plain tokenizer name is the library's: there is no bare-template path.
    assert renderer.name == "prime:Qwen/Qwen3-8B"
    assert recorder["pool"]["name"] == "Qwen/Qwen3-8B"
    # Reasoning is retained by default: dropping it would re-render an earlier
    # assistant turn differently from the tokens that were sampled.
    assert recorder["pool"]["config"].kwargs == {"thinking_retention": "all"}


def test_render_asks_for_the_generation_prompt(prime):
    renderer, recorder, _ = prime
    rendered = renderer.render([{"role": "user", "content": "hi"}])
    assert recorder["render"]["add_generation_prompt"] is True
    assert rendered.token_ids == (1, 2, 3)
    assert rendered.message_indices == (0, 0, -1)


def test_reasoning_content_survives_onto_the_message(prime):
    renderer, recorder, _ = prime
    recorder["parsed"] = FakeParsed(content="42", reasoning_content="let me think")
    message = renderer.parse_response([1, 2])
    assert message["content"] == "42"
    assert message["reasoning_content"] == "let me think"


def test_only_cleanly_parsed_tool_calls_become_structure(prime):
    renderer, recorder, _ = prime
    recorder["parsed"] = FakeParsed(
        content="",
        tool_calls=[
            FakeToolCall(raw="{}", name="search", arguments={"q": "x"}, id="call_1"),
            FakeToolCall(raw="{broken", name="edit", status=FakeStatus.INVALID_JSON),
        ],
    )
    message = renderer.parse_response([1, 2])
    assert [call["function"]["name"] for call in message["tool_calls"]] == ["search"]
    # Arguments are serialized, because that is what the wire format carries.
    assert message["tool_calls"][0]["function"]["arguments"] == '{"q":"x"}'


def test_a_bridge_that_preserves_the_prefix_is_accepted(prime):
    renderer, recorder, fake = prime
    fake.bridge_result = FakeRendered(
        token_ids=[1, 2, 3, 4, 5, 6], message_indices=[0, 0, 1, 1, 2, -1]
    )
    rendered = renderer.bridge([1, 2, 3], [4], [{"role": "user", "content": "next"}])
    assert rendered is not None
    assert rendered.reused_prefix_length == 4
    assert rendered.token_ids == (1, 2, 3, 4, 5, 6)


def test_bridged_attribution_covers_the_tail_only(prime):
    """Indices describe what the turn added, not the conversation behind it.

    Attribution for the reused prefix is already on the committed nodes, so
    carrying it again made every turn's bookkeeping proportional to the whole
    context instead of to the turn.
    """
    renderer, recorder, fake = prime
    fake.bridge_result = FakeRendered(
        token_ids=[1, 2, 3, 4, 5, 6], message_indices=[-1, -1, -1, -1, 0, -1]
    )
    rendered = renderer.bridge([1, 2, 3], [4], [{"role": "user", "content": "next"}])
    assert rendered is not None
    assert rendered.reused_prefix_length == 4
    assert rendered.message_indices == (0, -1)


def test_a_bridge_that_loses_the_end_of_the_previous_turn_falls_back(prime):
    """A moved boundary declines the bridge rather than failing the turn.

    The library preserves ``prev_prompt + prev_completion`` only after
    ``trim_to_turn_close``, and reports no length, so the end of the previous
    turn not landing where we handed it over means our slice offset is wrong --
    not that the request is bad. A full render is always correct, so this costs
    a slow turn and never correctness.
    """
    renderer, recorder, fake = prime
    fake.bridge_result = FakeRendered(
        token_ids=[1, 2, 3, 77, 5], message_indices=[0, 0, 0, 1, -1]
    )
    assert renderer.bridge([1, 2, 3], [4], [{"role": "user", "content": "next"}]) is None
    assert renderer.bridge_boundary_rejected == 1


def test_a_bridge_that_returns_too_few_tokens_falls_back(prime):
    renderer, recorder, fake = prime
    fake.bridge_result = FakeRendered(token_ids=[1, 2], message_indices=[0, 0])
    assert renderer.bridge([1, 2, 3], [4], [{"role": "user", "content": "next"}]) is None
    assert renderer.bridge_boundary_rejected == 1


def test_no_bridge_falls_back_to_a_full_render(prime):
    renderer, _, fake = prime
    fake.bridge_result = None
    assert renderer.bridge([1, 2], [3], [{"role": "user", "content": "next"}]) is None


def test_stop_token_ids_come_from_the_library(prime):
    renderer, _, _ = prime
    assert renderer.get_stop_token_ids() == (99, 100)
