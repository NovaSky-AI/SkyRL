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
from skycap.samples import build_samples
from skycap.server import CaptureServer
from skycap.tokens.backend import STATUS_HEADER, TokensBackend
from skycap.tokens.engine import ENGINES, unpack
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
    *, engine: str = "vllm", completion: Any = None, record_dir: Path | None = None, **options: Any
) -> AsyncIterator[TokenStack]:
    mock = MockEngine(completion)
    engine_server = TestServer(mock.app())
    await engine_server.start_server()
    renderer = FakeRenderer()
    backend = TokensBackend(str(engine_server.make_url("")).rstrip("/"), renderer, engine=ENGINES[engine](), **options)
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


async def test_the_sampling_mask_covers_each_trained_token() -> None:
    async with token_stack(engine="skyrl", sampling_mask=True) as stack:
        created = await stack.create()
        await converse(client(created["base_url"]), "hi")
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)

        assert stack.engine.requests[0]["return_sample_support"] is True
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


async def test_finish_releases_the_engine_session_on_skyrl() -> None:
    async with token_stack(engine="skyrl") as stack:
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
