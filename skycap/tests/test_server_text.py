"""Text mode end to end: the real OpenAI SDK, a capture server, a mock upstream."""

from __future__ import annotations

import asyncio

import aiohttp
import openai
import orjson
import pytest
from aiohttp.test_utils import TestServer

from skycap.openai_chat import StreamAssembler
from skycap.server import CaptureServer
from skycap.text import TextBackend
from tests.conftest import Stack, openai_client
from tests.mock_openai import MockOpenAI

MODEL = "policy"


def client(base_url: str) -> openai.AsyncOpenAI:
    return openai_client(base_url, api_key="harness-key")


def user(text: str) -> dict:
    return {"role": "user", "content": text}


async def test_create_returns_a_route_on_this_server(stack: Stack) -> None:
    created = await stack.create({"task": "t1"})
    document = await stack.document(created["id"])

    assert created["id"].startswith("tr_")
    assert created["base_url"] == f"{stack.url}/t/{created['id']}/v1"
    assert document["meta"] == {"task": "t1"}


async def test_a_multi_turn_conversation_is_one_path(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    messages = [user("hi")]
    for text in ("second", "third"):
        reply = await llm.chat.completions.create(model=MODEL, messages=messages)
        messages.append(reply.choices[0].message.model_dump(exclude_none=True))
        messages.append(user(text))
    await llm.chat.completions.create(model=MODEL, messages=messages)

    finished = await stack.finish(created["id"], {"reward": 1.0})
    document = await stack.document(created["id"])

    assert finished["status"] == "finished"
    (sample,) = finished["samples"]
    contents = [message["content"] for message in sample["messages"]]
    assert contents == ["hi", "re: hi", "second", "re: second", "third", "re: third"]
    assert sample["targets"] == [1, 3, 5]
    assert document["annotations"] == {"reward": 1.0}
    model_nodes = [node for node in document["nodes"] if node["author"] == "model"]
    assert all(len(node["calls"]) == 1 for node in model_nodes)


async def test_the_client_credential_is_replaced_by_the_upstream_one(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    await llm.chat.completions.create(model=MODEL, messages=[user("q")])

    upstream_headers = stack.upstream.headers[-1]
    assert upstream_headers["Authorization"] == "Bearer upstream-secret"


async def test_a_wrong_upstream_key_reaches_the_harness_as_the_upstream_401() -> None:
    """The upstream's own answer is forwarded, and recorded as a failure."""
    upstream = TestServer(MockOpenAI().app())
    await upstream.start_server()
    backend = TextBackend(str(upstream.make_url("/v1")), api_key="wrong-key")
    capture = TestServer(CaptureServer(backend).app())
    await capture.start_server()
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(capture.make_url("/trajectories"), json={}) as response:
                created = await response.json()
            llm = client(created["base_url"])
            with pytest.raises(openai.AuthenticationError) as raised:
                await llm.chat.completions.create(model=MODEL, messages=[user("q")])
            async with http.get(capture.make_url(f"/trajectories/{created['id']}")) as response:
                document = await response.json()
    finally:
        await capture.close()
        await upstream.close()

    assert raised.value.status_code == 401
    assert document["nodes"] == []
    assert [failure["status"] for failure in document["failures"]] == [401]


async def test_streamed_tool_calls_are_reassembled(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    stream = await llm.chat.completions.create(
        model=MODEL,
        messages=[user("search")],
        stream=True,
        extra_body={"mock": {"tool_call": "search"}},
    )
    chunks = [chunk async for chunk in stream]
    document = await stack.document(created["id"])

    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    reply = document["nodes"][1]
    assert reply["author"] == "model"
    expected_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q": "x"}'},
    }
    assert reply["message"]["tool_calls"] == [expected_call]
    assert reply["calls"][0]["finish_reason"] == "tool_calls"


async def test_streamed_text_matches_the_non_streamed_node(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    messages = [user("q")]
    stream = await llm.chat.completions.create(model=MODEL, messages=messages, stream=True)
    text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    await llm.chat.completions.create(model=MODEL, messages=messages)
    document = await stack.document(created["id"])

    assert text == "re: q"
    assert len(document["nodes"]) == 2
    assert len(document["nodes"][1]["calls"]) == 2


async def test_an_upstream_error_is_a_failure_not_a_node(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    with pytest.raises(openai.InternalServerError):
        await llm.chat.completions.create(model=MODEL, messages=[user("q")], extra_body={"mock": {"status": 500}})
    document = await stack.document(created["id"])

    assert document["nodes"] == []
    assert [failure["status"] for failure in document["failures"]] == [500]


async def test_n_greater_than_one_is_refused(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    with pytest.raises(openai.BadRequestError):
        await llm.chat.completions.create(model=MODEL, messages=[user("q")], n=2)

    assert stack.upstream.requests == []


async def test_finish_seals_the_route(stack: Stack) -> None:
    created = await stack.create()
    first = await stack.finish(created["id"], {"reward": 1.0})
    repeated = await stack.finish(created["id"], {"reward": 1.0})
    empty = await stack.finish(created["id"])

    assert first == repeated == empty
    llm = client(created["base_url"])
    with pytest.raises(openai.APIStatusError) as raised:
        await llm.chat.completions.create(model=MODEL, messages=[user("q")])
    assert raised.value.status_code == 410


async def test_new_annotations_after_finish_are_refused_not_dropped(stack: Stack) -> None:
    created = await stack.create()
    await stack.finish(created["id"], {"reward": 1.0})
    await stack.finish(created["id"], {"late": True}, expect=409)
    await stack.finish(created["id"], {"reward": 0.0}, expect=409)
    document = await stack.document(created["id"])

    assert document["annotations"] == {"reward": 1.0}


async def test_finish_drops_a_call_still_in_flight(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    slow_call = llm.chat.completions.create(model=MODEL, messages=[user("q")], extra_body={"mock": {"delay": 0.3}})
    call = asyncio.create_task(slow_call)
    await asyncio.sleep(0.1)
    finished = await stack.finish(created["id"])
    with pytest.raises(openai.APIError):
        await call
    # Wait past the upstream's delay: the dropped call must never land.
    await asyncio.sleep(0.4)
    document = await stack.document(created["id"])

    assert finished["samples"] == []
    assert document["nodes"] == []


async def test_finish_after_a_slow_call_returns_its_sample(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    await llm.chat.completions.create(model=MODEL, messages=[user("q")], extra_body={"mock": {"delay": 0.3}})
    finished = await stack.finish(created["id"])

    (sample,) = finished["samples"]
    assert [message["content"] for message in sample["messages"]] == ["q", "re: q"]


async def test_unknown_trajectory_is_404(stack: Stack) -> None:
    llm = client(f"{stack.url}/t/tr_nope/v1")
    with pytest.raises(openai.NotFoundError):
        await llm.chat.completions.create(model=MODEL, messages=[user("q")])


async def test_models_passes_through(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    models = await llm.models.list()

    assert [model.id for model in models.data] == ["policy"]


async def test_a_fork_trains_the_shared_prefix_once(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    question = [user("q")]
    await llm.chat.completions.create(model=MODEL, messages=question, extra_body={"mock": {"reply": "a"}})
    await llm.chat.completions.create(model=MODEL, messages=question, extra_body={"mock": {"reply": "b"}})
    continued = [*question, {"role": "assistant", "content": "b"}, user("more")]
    await llm.chat.completions.create(model=MODEL, messages=continued)

    samples = (await stack.finish(created["id"]))["samples"]

    assert [sample["path"] for sample in samples] == [[0, 1], [0, 2, 3, 4]]
    assert [sample["targets"] for sample in samples] == [[1], [2, 4]]


async def test_the_document_flags_a_shadowed_sibling(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    for top_p in (1.0, 0.9):
        await llm.chat.completions.create(
            model=MODEL, messages=[user("q")], top_p=top_p, extra_body={"mock": {"reply": "a"}}
        )
    nodes = (await stack.document(created["id"]))["nodes"]

    assert [node["shadowed_by"] for node in nodes] == [None, 2, None]


async def test_many_concurrent_trajectories_stay_separate(stack: Stack) -> None:
    async def rollout(index: int) -> tuple[str, list[str]]:
        created = await stack.create({"index": index})
        llm = client(created["base_url"])
        messages = [user(f"task {index}")]
        for turn in range(3):
            reply = await llm.chat.completions.create(model=MODEL, messages=messages)
            messages.append(reply.choices[0].message.model_dump(exclude_none=True))
            messages.append(user(f"t{turn}"))
        finished = await stack.finish(created["id"])
        (sample,) = finished["samples"]
        return created["id"], [message["content"] for message in sample["messages"]]

    results = await asyncio.gather(*(rollout(index) for index in range(32)))

    assert len({trajectory_id for trajectory_id, _ in results}) == 32
    for index, (_, contents) in enumerate(results):
        assert contents == [f"task {index}", f"re: task {index}", "t0", "re: t0", "t1", "re: t1"]


async def test_a_streamed_reply_is_recorded_before_its_stream_ends(stack: Stack) -> None:
    """A client that stops at [DONE] and finishes at once finds its reply recorded."""
    created = await stack.create()
    llm = client(created["base_url"])
    stream = await llm.chat.completions.create(
        model=MODEL, messages=[user("q")], stream=True, extra_body={"mock": {"linger": 0.5}}
    )
    text = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    finished = await stack.finish(created["id"])

    assert text == "re: q"
    (sample,) = finished["samples"]
    assert [message["content"] for message in sample["messages"]] == ["q", "re: q"]


async def test_a_broken_upstream_stream_fails_the_call(stack: Stack) -> None:
    created = await stack.create()
    llm = client(created["base_url"])
    with pytest.raises(openai.APIError):
        stream = await llm.chat.completions.create(
            model=MODEL, messages=[user("q")], stream=True, extra_body={"mock": {"abort": True}}
        )
        async for _ in stream:
            pass
    document = await stack.document(created["id"])

    assert document["nodes"] == []
    assert [failure["error"].split(":")[0] for failure in document["failures"]] == ["upstream stream interrupted"]


async def test_malformed_json_on_the_control_plane_is_a_400(stack: Stack) -> None:
    created = await stack.create()
    finish_url = f"{stack.url}/trajectories/{created['id']}/finish"
    async with stack.http.post(finish_url, data=b'{"annotations": {"reward": 1.0', headers=_JSON) as response:
        assert response.status == 400
    async with stack.http.post(f"{stack.url}/trajectories", data=b"{nope", headers=_JSON) as response:
        assert response.status == 400
    finished = await stack.finish(created["id"], {"reward": 1.0})

    assert finished["status"] == "finished"
    assert (await stack.document(created["id"]))["annotations"] == {"reward": 1.0}


async def test_a_finish_during_the_body_read_cancels_the_call(stack: Stack) -> None:
    created = await stack.create()
    body_started = asyncio.Event()

    async def slow_body():  # noqa: ANN202
        yield b'{"model": "policy", '
        body_started.set()
        await asyncio.sleep(0.3)
        yield b'"messages": [{"role": "user", "content": "q"}]}'

    call = asyncio.create_task(
        stack.http.post(f"{created['base_url']}/chat/completions", data=slow_body(), headers=_JSON)
    )
    await body_started.wait()
    await asyncio.sleep(0.05)
    await stack.finish(created["id"])
    with pytest.raises(aiohttp.ClientError):
        response = await call
        await response.read()

    assert stack.upstream.requests == []
    assert (await stack.document(created["id"]))["nodes"] == []


def test_a_tool_call_delta_with_a_null_index_is_the_first_call() -> None:
    frame = {"choices": [{"delta": {"tool_calls": [{"index": None, "id": "c", "function": {"name": "f"}}]}}]}
    assembler = StreamAssembler()
    assembler.feed(b"data: " + orjson.dumps(frame) + b"\n\n")

    assert assembler.tool_calls[0]["function"]["name"] == "f"


_JSON = {"Content-Type": "application/json"}
