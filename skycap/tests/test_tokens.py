"""Token mode end to end: the OpenAI SDK, a capture server, a fake renderer and a mock engine."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import openai
import pytest
from aiohttp.test_utils import TestServer

from skycap import record
from skycap.hashing import MatchKey
from skycap.samples import build_samples
from skycap.server import CaptureServer
from skycap.tokens.backend import STATUS_HEADER, TokensBackend
from skycap.tokens.engine import VLLMEngine, unpack
from skycap.tokens.turn import TokenError, attribute
from tests.conftest import openai_client
from tests.fake_renderer import END, FakeRenderer, encode
from tests.mock_engine import MockEngine


@dataclass
class TokenStack:
    engine: MockEngine
    renderer: FakeRenderer
    server: CaptureServer
    url: str
    http: aiohttp.ClientSession

    async def create(self) -> dict:
        async with self.http.post(f"{self.url}/trajectories", json={}) as response:
            return await response.json()

    async def finish(self, trajectory_id: str, annotations: dict | None = None) -> dict:
        body = {"annotations": annotations or {}}
        async with self.http.post(f"{self.url}/trajectories/{trajectory_id}/finish", json=body) as response:
            assert response.status == 200
            return await response.json()


@asynccontextmanager
async def token_stack(
    *, engine: VLLMEngine | None = None, completion: Any = None, record_dir: Path | None = None, **options: Any
) -> AsyncIterator[TokenStack]:
    mock = MockEngine(completion)
    engine_server = TestServer(mock.app())
    await engine_server.start_server()
    renderer = FakeRenderer()
    backend = TokensBackend(str(engine_server.make_url("")).rstrip("/"), renderer, engine=engine, **options)
    server = CaptureServer(backend, record_dir=record_dir)
    capture = TestServer(server.app())
    await capture.start_server()
    try:
        async with aiohttp.ClientSession() as http:
            yield TokenStack(mock, renderer, server, str(capture.make_url("")).rstrip("/"), http)
    finally:
        await capture.close()
        await engine_server.close()


def client(base_url: str) -> openai.AsyncOpenAI:
    return openai_client(base_url)


def user(text: str) -> dict:
    return {"role": "user", "content": text}


async def converse(llm: openai.AsyncOpenAI, *texts: str, **kwargs: Any) -> list[dict]:
    """One user message per text, each answered; returns the full history."""
    messages: list[dict] = []
    for text in texts:
        messages.append(user(text))
        reply = await llm.chat.completions.create(model="policy", messages=messages, **kwargs)
        messages.append(reply.choices[0].message.model_dump(exclude_none=True))
    return messages


# -- attribution ---------------------------------------------------------------
TEXT_KEY = MatchKey.text("", "policy")
TOKEN_KEY = MatchKey.tokens("", "policy")


def test_match_keys_keep_the_hash_values_recorded_before_them() -> None:
    """Records store match hashes, so the values a ``MatchKey`` gives must not drift."""
    message = {
        "role": "assistant",
        "content": "hi",
        "refusal": None,
        "provider_specific_fields": {"x": 1},
        "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    }

    assert MatchKey.text("t", "policy")(message) == "f378894da4b0e48303e35fdf42be5375"
    assert MatchKey.tokens("t", "policy")(message) == "2bc94eedd73c39da36f6e5cee703ab40"


@pytest.mark.parametrize("fields", [{"refusal": None}, {"response_id": "resp_1", "refusal": None}])
def test_token_match_ignores_provider_specific_fields(fields: dict) -> None:
    reply = {"role": "assistant", "content": "answer"}
    replay = {**reply, "provider_specific_fields": fields}

    assert TOKEN_KEY(reply) == TOKEN_KEY(replay)
    assert TEXT_KEY(reply) != TEXT_KEY(replay)


def test_token_match_leaves_empty_tool_content_to_the_renderer() -> None:
    """``content: ""`` against no ``content`` is template-dependent, so the hash keeps them apart.

    Whether they are the same message is decided by rendering, when a request
    is planned: see ``test_empty_tool_call_content_replay_bridges_but_rewrite_forks``
    and ``test_a_replayed_message_the_template_renders_differently_forks``.
    """
    tool_call = {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    # What skycap answered with, and what the client replays in its next request's history.
    returned = {"role": "assistant", "content": "", "reasoning_content": "hmm", "tool_calls": [tool_call]}
    replayed = _without_content(returned)
    with_metadata = {**replayed, "provider_specific_fields": {"refusal": None}}

    assert TOKEN_KEY(with_metadata) == TOKEN_KEY(replayed)
    assert TOKEN_KEY(returned) != TOKEN_KEY(replayed)
    assert TEXT_KEY(returned) != TEXT_KEY(with_metadata)
    assert TOKEN_KEY(returned) != TOKEN_KEY({**replayed, "content": "Summary so far"})


def _without_content(message: dict) -> dict:
    replayed = dict(message)
    del replayed["content"]
    return replayed


def test_token_match_ignores_unrendered_fields_inside_tool_calls() -> None:
    tool_call = {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    returned = {"role": "assistant", "content": "x", "tool_calls": [tool_call]}
    replayed = {
        **returned,
        "refusal": None,
        "annotations": [],
        "audio": {"id": "audio_0"},
        "provider_specific_fields": {"refusal": None},
        "tool_calls": [
            {
                **tool_call,
                "provider_specific_fields": {"thought_signature": "sig"},
                "function": {**tool_call["function"], "provider_specific_fields": {"index": 0}},
            }
        ],
    }
    edited = {**returned, "tool_calls": [{**tool_call, "function": {"name": "search", "arguments": '{"q":1}'}}]}

    assert TOKEN_KEY(returned) == TOKEN_KEY(replayed)
    assert TOKEN_KEY(returned) != TOKEN_KEY(edited)


def test_scaffold_belongs_to_the_following_message_and_the_tail_to_the_reply() -> None:
    chunks, scaffold = attribute([9, 1, 1, 8, 2, 2, 7, 7], [-1, 0, 0, -1, 1, 1, -1, -1], 2)
    assert chunks == [[9, 1, 1], [8, 2, 2]]
    assert scaffold == [7, 7]


def test_attribution_out_of_range_or_order_is_refused() -> None:
    with pytest.raises(TokenError):
        attribute([1, 2], [0, 3], 2)
    with pytest.raises(TokenError):
        attribute([1, 2], [1, 0], 2)


# -- one path, bridged -----------------------------------------------------------
async def test_a_continued_conversation_extends_the_previous_tokens() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        first, second = stack.engine.requests
        previous = first["token_ids"] + [*encode(f"re{len(first['token_ids'])}"), END]

        assert second["token_ids"][: len(previous)] == previous
        assert stack.engine.headers[0]["X-Session-ID"] == created["id"]
        graph = stack.server.trajectories[created["id"]].graph
        (path,) = graph.paths()
        assert [graph.nodes[i].author for i in path] == ["client", "model", "client", "model"]


async def test_the_sample_is_exactly_what_inference_saw() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        graph = stack.server.trajectories[created["id"]].graph
        (sample,) = build_samples(graph)
        last = stack.engine.requests[-1]["token_ids"]
        completions = [[*encode(f"re{len(r['token_ids'])}"), END] for r in stack.engine.requests]

        assert sample.input_ids == last + completions[1]
        sampled = [i for i, bit in enumerate(sample.loss_mask) if bit]
        first_start = len(stack.engine.requests[0]["token_ids"])
        assert sampled == [
            *range(first_start, first_start + len(completions[0])),
            *range(len(last), len(last) + len(completions[1])),
        ]
        assert [sample.logprobs[i] for i in sampled[: len(completions[0])]] == pytest.approx(
            [-0.01 * (i + 1) for i in range(len(completions[0]))]
        )


async def test_routed_experts_align_and_the_placeholder_is_replaced() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)
        routed = sample.routed_experts

        assert routed is not None and routed.shape == (len(sample.input_ids), 2, 2)
        positions = np.arange(len(sample.input_ids)) % 256
        # Every row is its position's, including the end of the first reply,
        # which only the second call forwarded. The very last is a placeholder.
        np.testing.assert_array_equal(routed[:-1, 0, 0], positions[:-1])
        assert routed[-1, 0, 0] == routed[-2, 0, 0]


class _FullRoutesEngine(VLLMEngine):
    routes_from_supported = False


async def test_a_wire_without_routes_from_gets_every_calls_full_routes() -> None:
    async with token_stack(engine=_FullRoutesEngine()) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)

        assert not any("routed_experts_prompt_start" in r["sampling_params"] for r in stack.engine.requests)
        np.testing.assert_array_equal(sample.routed_experts[:-1, 0, 0], np.arange(len(sample.input_ids) - 1) % 256)


async def test_a_turn_fetches_only_the_routes_it_lacks_and_they_still_align() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more", "again")
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)
        routed = sample.routed_experts

        starts = [r["sampling_params"].get("routed_experts_prompt_start", 0) for r in stack.engine.requests]
        # The first call has no history; each later one starts at the previous reply's last token.
        assert starts[0] == 0 and all(0 < a < b for a, b in zip(starts[1:], starts[2:]))
        assert routed is not None and routed.shape == (len(sample.input_ids), 2, 2)
        np.testing.assert_array_equal(routed[:-1, 0, 0], np.arange(len(sample.input_ids) - 1) % 256)


async def test_the_sampling_mask_covers_each_trained_token() -> None:
    async with token_stack(sampling_mask=True) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi")
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)

        assert sample.sampling_mask is not None
        for position, bit in enumerate(sample.loss_mask):
            token = sample.input_ids[position]
            assert sample.sampling_mask[position] == ([token, token + 1] if bit else [])


# -- the graph's other shapes ------------------------------------------------------
async def test_an_identical_retry_is_one_node_with_two_calls() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        for _ in range(2):
            await llm.chat.completions.create(model="policy", messages=[user("q")])
        graph = stack.server.trajectories[created["id"]].graph

        assert len(graph) == 2
        assert len(graph.nodes[1].calls) == 2


async def test_an_append_only_conversation_bridges_every_call_after_the_first() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "a", "b", "c")
        graph = stack.server.trajectories[created["id"]].graph

        assert [call.bridged for node in graph if node.author == "model" for call in node.calls] == [None, True, True]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0


async def test_client_metadata_does_not_prevent_bridging() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        first = await llm.chat.completions.create(model="policy", messages=[user("q")])
        reply = first.choices[0].message.model_dump(exclude_none=True)
        reply["provider_specific_fields"] = {"refusal": None, "response_id": "resp_1"}
        async with stack.http.post(
            f"{created['base_url']}/chat/completions",
            json={"model": "policy", "messages": [user("q"), reply, user("next")]},
        ) as response:
            assert response.status == 200

        graph = stack.server.trajectories[created["id"]].graph
        assert [call.bridged for node in graph if node.author == "model" for call in node.calls] == [None, True]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0


async def test_empty_tool_call_content_replay_bridges_but_rewrite_forks() -> None:
    completion = [*encode("THINK:hmm|CALL:search:{}"), END]
    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        first = await llm.chat.completions.create(model="policy", messages=[user("q")], tools=tools)
        reply = first.choices[0].message.model_dump(exclude_none=True)
        assert reply["content"] == "" and reply["reasoning_content"] == "hmm" and reply["tool_calls"]

        replay = {key: value for key, value in reply.items() if key != "content"}
        replay["provider_specific_fields"] = {"refusal": None}
        tool_result = {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "found"}
        async with stack.http.post(
            f"{created['base_url']}/chat/completions",
            json={"model": "policy", "messages": [user("q"), replay, tool_result], "tools": tools},
        ) as response:
            assert response.status == 200

        async with stack.http.post(
            f"{created['base_url']}/chat/completions",
            json={
                "model": "policy",
                "messages": [user("q"), {"role": "assistant", "content": "Summary so far"}, user("next")],
                "tools": tools,
            },
        ) as response:
            assert response.status == 200

        graph = stack.server.trajectories[created["id"]].graph
        model_nodes = [node for node in graph if node.author == "model"]
        assert len(model_nodes) == 3
        assert model_nodes[0].id in graph.path_to(model_nodes[1].id)
        assert model_nodes[0].id not in graph.path_to(model_nodes[2].id)
        assert [call.bridged for node in model_nodes for call in node.calls] == [None, True, False]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 1


TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


async def _post(stack: TokenStack, created: dict, body: dict) -> None:
    async with stack.http.post(f"{created['base_url']}/chat/completions", json=body) as response:
        assert response.status == 200


async def _tool_call_then_replay_without_content(stack: TokenStack, created: dict) -> dict:
    """Answer ``q`` with a tool call, then replay it without its empty ``content``, plus the tool result."""
    first = await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")], tools=TOOLS)
    returned = first.choices[0].message.model_dump(exclude_none=True)
    assert returned["content"] == "" and returned["tool_calls"]
    replayed = _without_content(returned)
    tool_result = {"role": "tool", "tool_call_id": returned["tool_calls"][0]["id"], "content": "found"}
    body = {"model": "policy", "messages": [user("q"), replayed, tool_result], "tools": TOOLS}
    await _post(stack, created, body)
    return body


async def test_a_replayed_message_the_template_renders_differently_forks() -> None:
    """With a template that renders ``content: ""``, dropping it is an edit, not a respelling."""
    completion = [*encode("THINK:hmm|CALL:search:{}"), END]
    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.renderer.empty_content = "<empty>"
        created = await stack.create()
        await _tool_call_then_replay_without_content(stack, created)

        graph = stack.server.trajectories[created["id"]].graph
        model_nodes = [node for node in graph if node.author == "model"]
        assert model_nodes[0].id not in graph.path_to(model_nodes[1].id)
        assert [call.bridged for node in model_nodes for call in node.calls] == [None, False]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 1


async def test_a_respelled_message_is_rendered_once_then_matched_by_alias() -> None:
    completion = [*encode("THINK:hmm|CALL:search:{}"), END]
    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        created = await stack.create()
        body = await _tool_call_then_replay_without_content(stack, created)
        # The first call renders once; the second renders the request with each spelling.
        assert stack.renderer.renders == 3

        await _post(stack, created, body)

        assert stack.renderer.renders == 3
        graph = stack.server.trajectories[created["id"]].graph
        assert [call.bridged for node in graph if node.author == "model" for call in node.calls] == [None, True, True]


async def test_a_respelled_message_after_the_last_model_node_commits_as_that_node() -> None:
    """A bridged call commits every message after the model node; a respelled one must find its node, not fork."""
    async with token_stack() as stack:
        created = await stack.create()
        first = await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])
        answer = first.choices[0].message.model_dump(exclude_none=True)
        await _post(stack, created, {"model": "policy", "messages": [user("q"), answer, {**user("next"), "name": ""}]})
        graph = stack.server.trajectories[created["id"]].graph
        nodes = len(graph)

        await _post(stack, created, {"model": "policy", "messages": [user("q"), answer, user("next")]})

        assert len(graph) == nodes
        assert graph.branch_points() == []
        assert [call.bridged for node in graph if node.author == "model" for call in node.calls] == [None, True, True]


async def test_siblings_with_one_spelling_are_render_checked_once() -> None:
    """Two samples of the same reply (here under two temperatures) are one candidate spelling, not two."""
    completion = [*encode("THINK:hmm|CALL:search:{}"), END]
    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.renderer.empty_content = "<empty>"
        created = await stack.create()
        llm = client(created["base_url"])
        for temperature in (0.5, 1.0):
            first = await llm.chat.completions.create(
                model="policy", messages=[user("q")], tools=TOOLS, temperature=temperature
            )
        returned = first.choices[0].message.model_dump(exclude_none=True)
        graph = stack.server.trajectories[created["id"]].graph
        assert len([node for node in graph if node.author == "model"]) == 2
        renders = stack.renderer.renders

        replayed = _without_content(returned)
        tool_result = {"role": "tool", "tool_call_id": returned["tool_calls"][0]["id"], "content": "found"}
        await _post(stack, created, {"model": "policy", "messages": [user("q"), replayed, tool_result], "tools": TOOLS})

        # The request once, and one swapped-in spelling, which renders differently.
        assert stack.renderer.renders == renders + 2


async def test_a_rewrite_is_not_rendered_twice() -> None:
    """A message that differs from every sibling in more than empty strings is new without a render check."""
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        await llm.chat.completions.create(model="policy", messages=[user("q")])
        renders = stack.renderer.renders
        await llm.chat.completions.create(
            model="policy", messages=[user("q"), {"role": "assistant", "content": "Summary so far"}, user("next")]
        )

        assert stack.renderer.renders == renders + 1


async def test_stripped_reasoning_forks_and_trains_each_sample_once() -> None:
    thinking = [*encode("THINK:hmm|answer"), END]
    async with token_stack(completion=lambda prompt, sampling: thinking) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        reply = await llm.chat.completions.create(model="policy", messages=[user("q")])
        assert reply.choices[0].message.content == "answer"
        await llm.chat.completions.create(
            model="policy",
            messages=[user("q"), {"role": "assistant", "content": "answer"}, user("more")],
        )
        graph = stack.server.trajectories[created["id"]].graph
        samples = build_samples(graph)

        assert graph.branch_points() == [0]
        assert [(graph.nodes[i].author, graph.nodes[i].message.get("content")) for i in graph.children(0)] == [
            ("model", "answer"),
            ("client", "answer"),
        ]
        assert [len(s.targets) for s in samples] == [1, 1]
        # The second call's prompt was rendered, not extended, and finish says so.
        assert [call.bridged for call in graph.nodes[1].calls] == [None]
        second = next(n for n in graph if n.author == "model" and n.id != 1)
        assert [call.bridged for call in second.calls] == [False]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 1


async def test_use_raw_content_keeps_reasoning_inline_so_a_verbatim_replay_stays_one_path() -> None:
    thinking = [*encode("THINK:hmm|answer"), END]
    async with token_stack(completion=lambda prompt, sampling: thinking, use_raw_content=True) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        raw = await stack.http.post(
            f"{created['base_url']}/chat/completions", json={"model": "policy", "messages": [user("q")]}
        )
        # As vLLM with no reasoning parser answers: the field is there, and null.
        assert (await raw.json())["choices"][0]["message"]["reasoning_content"] is None
        reply = (await llm.chat.completions.create(model="policy", messages=[user("q")])).choices[0].message
        assert reply.content == "THINK:hmm|answer"
        # A harness that replays only `content` sends back exactly what it got.
        await llm.chat.completions.create(
            model="policy",
            messages=[user("q"), {"role": "assistant", "content": reply.content}, user("more")],
        )
        graph = stack.server.trajectories[created["id"]].graph

        assert graph.branch_points() == []
        assert len(graph.paths()) == 1
        first, second = stack.engine.requests[-2:]
        assert second["token_ids"][: len(first["token_ids"]) + len(thinking)] == first["token_ids"] + thinking


async def test_a_tool_call_is_parsed_and_the_tool_result_bridges() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        first = await llm.chat.completions.create(model="policy", messages=[user("TOOL please")], tools=tools)
        call = first.choices[0].message

        assert first.choices[0].finish_reason == "tool_calls"
        assert call.tool_calls[0].function.name == "search"
        history = [
            user("TOOL please"),
            call.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.tool_calls[0].id, "content": "found"},
        ]
        await llm.chat.completions.create(model="policy", messages=history, tools=tools)
        first_request, second_request = stack.engine.requests
        assert second_request["token_ids"][: len(first_request["token_ids"])] == first_request["token_ids"]
        graph = stack.server.trajectories[created["id"]].graph
        (call,) = graph.nodes[1].calls
        assert graph.tools[call.tools] == tools


async def test_a_streamed_answer_is_synthesized_from_the_completion() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        stream = await client(created["base_url"]).chat.completions.create(
            model="policy", messages=[user("q")], stream=True
        )
        text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])
        prompt = stack.engine.requests[0]["token_ids"]

        assert text == f"re{len(prompt)}"


# -- sampling ---------------------------------------------------------------------------
async def test_overrides_and_stop_tokens_reach_the_engine_and_the_node() -> None:
    async with token_stack(sampling_overrides={"top_k": 20}, max_model_len=10_000) as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(
            model="policy", messages=[user("q")], temperature=0.7, max_tokens=50
        )
        params = stack.engine.requests[0]["sampling_params"]

        assert params["top_k"] == 20 and params["temperature"] == 0.7 and params["max_tokens"] == 50
        assert params["stop_token_ids"] == [END] and params["logprobs"] == 0
        (call,) = stack.server.trajectories[created["id"]].graph.nodes[1].calls
        assert call.sampling == {"temperature": 0.7, "max_tokens": 50, "top_k": 20}


async def test_a_prompt_that_leaves_no_room_is_refused() -> None:
    async with token_stack(max_model_len=5) as stack:
        created = await stack.create()
        with pytest.raises(openai.BadRequestError) as raised:
            await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])
        assert "context_length_exceeded" in str(raised.value)
        assert stack.engine.requests == []


# -- failures ---------------------------------------------------------------------------
async def test_an_engine_error_is_a_failure_and_no_node() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        with pytest.raises(openai.InternalServerError):
            await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("BOOM")])
        trajectory = stack.server.trajectories[created["id"]]

        assert len(trajectory.graph) == 0
        assert [f.status for f in trajectory.failures] == [500]
        assert trajectory.is_open


async def test_an_unattributable_prompt_is_refused_before_inference() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        stack.renderer.corrupt = True
        with pytest.raises(openai.BadRequestError):
            await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])

        assert stack.engine.requests == []
        assert stack.server.trajectories[created["id"]].is_open


async def test_a_turn_that_cannot_commit_exactly_fails_the_trajectory_but_answers() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        stack.engine.bad_routing = True
        raw = await llm.chat.completions.with_raw_response.create(model="policy", messages=[user("q")])

        assert raw.parse().choices[0].message.content.startswith("re")
        assert raw.headers[STATUS_HEADER] == "failed"
        with pytest.raises(openai.APIStatusError) as closed:
            await llm.chat.completions.create(model="policy", messages=[user("q")])
        assert closed.value.status_code == 410
        finished = await stack.finish(created["id"], {"reward": 0.0})
        assert finished["status"] == "failed"
        assert finished["samples"] == []
        document = stack.server.trajectories[created["id"]].document()
        assert document["annotations"] == {"reward": 0.0}


class SessionEngine(VLLMEngine):
    """An engine behind a router that holds per-session state."""

    release_path = "/finish_session"


async def test_finish_releases_the_engine_session_when_the_engine_has_one() -> None:
    async with token_stack(engine=SessionEngine()) as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])
        finished = await stack.finish(created["id"])

        assert stack.engine.released == [created["id"]]
        (sample,) = finished["samples"]
        assert len(sample["input_ids"]) == len(sample["loss_mask"]) == len(sample["logprobs"])
        assert unpack(sample["routed_experts"]).shape == (len(sample["input_ids"]), 2, 2)


async def test_a_token_trajectory_round_trips_through_the_record(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path, sampling_mask=True) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        live = [s.to_json() for s in build_samples(stack.server.trajectories[created["id"]].graph)]
        finished = await stack.finish(created["id"])

        assert finished["samples"] == live
        loaded = record.load(tmp_path, created["id"])
        assert [s.to_json() for s in build_samples(loaded.graph)] == live
        assert loaded.capture["mode"] == "tokens" and loaded.capture["tokenizer"] == "fake"


async def test_the_record_carries_each_nodes_text_and_token_spans(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi", "more")
        await stack.finish(created["id"])

    graph = record.load(tmp_path, created["id"]).graph
    first_prompt = stack.engine.requests[0]["token_ids"]
    reply = graph.nodes[1].tokens
    assert reply is not None and reply.text is not None and reply.text_offsets is not None
    assert reply.text == f"<s>assistant\nre{len(first_prompt)}</s>"
    data = reply.text.encode()
    bounds = [*reply.text_offsets, len(data)]
    spans = [data[start:end].decode() for start, end in zip(bounds, bounds[1:])]
    assert spans[:3] == ["<s>", "a", "s"]
    assert spans[-1] == "</s>"
    for node in graph:
        assert node.tokens is not None and node.tokens.text is not None
        assert len(node.tokens.text_offsets) == len(node.tokens.token_ids)


async def test_the_vllm_engine_has_no_session_to_release() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])
        await stack.finish(created["id"])

        assert stack.engine.released == []


async def test_an_override_wins_over_a_callers_max_completion_tokens() -> None:
    async with token_stack(sampling_overrides={"max_tokens": 50}) as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(
            model="policy", messages=[user("q")], max_completion_tokens=100
        )

        assert stack.engine.requests[0]["sampling_params"]["max_tokens"] == 50


async def test_models_is_the_engines_list_without_a_configured_model() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        models = await client(created["base_url"]).models.list()

        assert [model.id for model in models.data] == ["engine-model"]


async def test_a_failed_trajectory_flushed_at_shutdown_still_takes_its_finish(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        stack.engine.bad_routing = True
        await client(created["base_url"]).chat.completions.with_raw_response.create(
            model="policy", messages=[user("q")]
        )
    assert record.read_document(tmp_path, created["id"])["ended"] is False

    async with token_stack(record_dir=tmp_path) as restarted:
        finished = await restarted.finish(created["id"], {"reward": 0.0})

    document = record.read_document(tmp_path, created["id"])
    assert finished["status"] == "failed"
    assert (document["ended"], document["annotations"]) == (True, {"reward": 0.0})


async def test_shutdown_releases_sessions_of_trajectories_that_never_ended(tmp_path: Path) -> None:
    async with token_stack(engine=SessionEngine(), record_dir=tmp_path) as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])

    assert stack.engine.released == [created["id"]]


async def test_ending_a_trajectory_drops_its_turn_lock() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("q")])
        assert created["id"] in stack.server.backend._locks
        await stack.finish(created["id"])

        assert created["id"] not in stack.server.backend._locks


def test_a_node_with_no_tokens_keeps_the_paths_routed_experts() -> None:
    from skycap.tokens.turn import _Routing

    routing = _Routing(np.zeros((5, 2, 2), dtype=np.uint8), 0, 6)
    empty = routing.slice(3, 0)

    assert empty is not None and empty.shape == (0, 2, 2)
