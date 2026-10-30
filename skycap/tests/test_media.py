"""Token mode with images: the engine gets every item in the prompt, and the sample carries them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from skycap import record
from skycap.samples import Sample, build_samples
from skycap.tokens.renderer import Media
from skycap.tokens.turn import TokenError, attribute_media
from tests.fake_renderer import IMG, image, image_data
from tests.test_tokens import client, token_stack, user


def look(text: str, *images: dict[str, Any]) -> dict[str, Any]:
    return {"role": "user", "content": [*images, {"type": "text", "text": text}]}


async def ask(llm: Any, messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Send ``message`` after ``messages`` and append it and the reply, as a harness does."""
    messages.append(message)
    reply = await llm.chat.completions.create(model="policy", messages=messages)
    messages.append(reply.choices[0].message.model_dump(exclude_none=True))


def placeholders(request: dict[str, Any]) -> list[tuple[int, int]]:
    return [(p["offset"], p["length"]) for p in request["features"]["mm_placeholders"]["image"]]


def assert_points_at_placeholders(token_ids: list[int], spans: list[tuple[int, int]]) -> None:
    for offset, length in spans:
        assert token_ids[offset : offset + length] == [IMG] * length
    assert sum(length for _, length in spans) == token_ids.count(IMG)


async def test_every_call_sends_the_images_of_its_whole_prompt() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        await ask(llm, messages, look("and this?", image("dog", 2)))
        await ask(llm, messages, user("thanks"))
        first, second, third = stack.engine.requests

        assert first["features"]["mm_hashes"] == {"image": ["cat"]}
        assert second["features"]["mm_hashes"] == {"image": ["cat", "dog"]}
        assert third["features"]["mm_hashes"] == {"image": ["cat", "dog"]}
        assert third["features"]["kwargs_data"] == {"image": ["encoded:cat", "encoded:dog"]}
        for request in (first, second, third):
            assert_points_at_placeholders(request["token_ids"], placeholders(request))
        # Each later call extended the one before, with the earlier images handed to the bridge.
        assert placeholders(second)[0] == placeholders(first)[0]
        assert [[item.hash for item in media] for media in stack.renderer.bridged_media] == [["cat"], ["cat", "dog"]]
        assert stack.renderer.renders == 1


async def test_a_text_only_conversation_sends_no_features() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, user("hi"))
        await ask(llm, messages, user("more"))

        assert all("features" not in request for request in stack.engine.requests)
        assert stack.renderer.bridged_media == []


async def test_each_image_belongs_to_the_message_that_sent_it() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("two at once", image("cat", 3), image("owl", 1)))
        await ask(llm, messages, look("and this?", image("dog", 2)))
        graph = stack.server.trajectories[created["id"]].graph
        (path,) = graph.paths()
        nodes = [graph.nodes[i] for i in path]

        assert [[item.hash for item in node.tokens.media] for node in nodes] == [["cat", "owl"], [], ["dog"], []]
        for node in nodes:
            assert_points_at_placeholders(node.tokens.token_ids, [(m.offset, m.length) for m in node.tokens.media])


async def test_the_sample_carries_its_images_in_order_with_their_arrays() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        await ask(llm, messages, look("and this?", image("dog", 2)))
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)

        assert [item.hash for item in sample.media] == ["cat", "dog"]
        assert_points_at_placeholders(sample.input_ids, [(m.offset, m.length) for m in sample.media])
        last = stack.engine.requests[-1]
        assert [(m.offset, m.length) for m in sample.media] == placeholders(last)
        np.testing.assert_array_equal(sample.media[1].data["pixel_values"], image_data("dog", 2)["pixel_values"])
        assert all(sample.loss_mask[m.offset + k] == 0 for m in sample.media for k in range(m.length))


async def test_samples_with_images_survive_the_finish_wire() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        (live,) = build_samples(stack.server.trajectories[created["id"]].graph)
        finished = await stack.finish(created["id"])
        (sent,) = [Sample.from_json(sample) for sample in finished["samples"]]

        assert [(m.modality, m.offset, m.length, m.hash) for m in sent.media] == [
            (m.modality, m.offset, m.length, m.hash) for m in live.media
        ]
        for key in ("pixel_values", "image_grid_thw"):
            np.testing.assert_array_equal(sent.media[0].data[key], live.media[0].data[key])
            assert sent.media[0].data[key].dtype == live.media[0].data[key].dtype


async def test_a_full_render_keeps_the_matched_nodes_and_sends_every_image() -> None:
    async with token_stack() as stack:
        stack.renderer.no_bridge = True
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        await ask(llm, messages, look("and this?", image("dog", 2)))
        graph = stack.server.trajectories[created["id"]].graph
        first, second = stack.engine.requests

        assert len(graph.paths()) == 1
        assert second["features"]["mm_hashes"] == {"image": ["cat", "dog"]}
        assert placeholders(second)[0] == placeholders(first)[0]
        assert_points_at_placeholders(second["token_ids"], placeholders(second))
        (sample,) = build_samples(graph)
        assert [(m.offset, m.length) for m in sample.media] == placeholders(second)


async def test_a_different_image_in_a_replayed_message_forks() -> None:
    async with token_stack() as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        # The same question about another image of the same size: identical placeholder tokens.
        edited = [look("what is this?", image("cow", 3)), *messages[1:]]
        await ask(llm, edited, user("sure?"))
        graph = stack.server.trajectories[created["id"]].graph

        assert len(graph.roots()) == 2
        assert stack.engine.requests[-1]["features"]["mm_hashes"] == {"image": ["cow"]}


async def test_the_record_keeps_placeholders_but_not_arrays(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        await stack.finish(created["id"])

    document = record.read_document(tmp_path, created["id"])
    first = document["nodes"][0]["tokens"]["media"]
    assert first == [{"modality": "image", "offset": first[0]["offset"], "length": 3, "hash": "cat"}]
    loaded = record.load(tmp_path, created["id"])
    (sample,) = build_samples(loaded.graph)
    assert [(m.hash, m.length, m.data) for m in sample.media] == [("cat", 3, None)]


def test_an_image_in_no_message_or_across_two_is_refused() -> None:
    chunks = [[0] * 5, [0] * 4]

    assert attribute_media([Media("image", 12, 2, "a")], 10, chunks) == [[Media("image", 2, 2, "a")], []]
    with pytest.raises(TokenError, match="crosses"):
        attribute_media([Media("image", 13, 3, "a")], 10, chunks)
    with pytest.raises(TokenError, match="in no message"):
        attribute_media([Media("image", 19, 1, "a")], 10, chunks)


async def test_a_message_whose_image_changed_is_a_new_node_with_the_new_image() -> None:
    async with token_stack() as stack:
        stack.renderer.no_bridge = True
        created = await stack.create()
        llm = client(created["base_url"])
        messages: list[dict[str, Any]] = []
        await ask(llm, messages, look("what is this?", image("cat", 3)))
        # The same message, the same placeholder tokens, but the image now processes to other content.
        stack.renderer.image_salt = "-v2"
        await ask(llm, messages, user("sure?"))
        graph = stack.server.trajectories[created["id"]].graph

        assert stack.engine.requests[-1]["features"]["mm_hashes"] == {"image": ["cat-v2"]}
        assert len(graph.roots()) == 2
        assert [[m.hash for m in graph.nodes[root].tokens.media] for root in graph.roots()] == [["cat"], ["cat-v2"]]
        (_, second) = build_samples(graph)
        assert [item.hash for item in second.media] == ["cat-v2"]


async def test_a_repeated_finish_of_a_recorded_trajectory_with_images_returns_no_samples(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        llm = client(created["base_url"])
        await ask(llm, [], look("what is this?", image("cat", 3)))
        first = await stack.finish(created["id"])
        repeat = await stack.finish(created["id"])

    assert len(first["samples"]) == 1 and first["samples"][0]["media"][0]["data"] is not None
    assert repeat["status"] == "finished" and repeat["samples"] == []


async def test_a_repeated_finish_of_a_recorded_text_trajectory_still_returns_its_samples(tmp_path: Path) -> None:
    async with token_stack(record_dir=tmp_path) as stack:
        created = await stack.create()
        await ask(client(created["base_url"]), [], user("hi"))
        first = await stack.finish(created["id"])
        repeat = await stack.finish(created["id"])

    assert repeat["samples"] == first["samples"] != []


async def test_images_a_model_cannot_send_to_the_engine_are_refused_but_text_still_works() -> None:
    import openai

    async with token_stack() as stack:
        # As the first call with images records it for a renderer whose family renders images but has no vLLM
        # encoder (test_the_first_check_records_why_a_renderer_cannot_send_images).
        stack.server.backend.media_error = "Multimodal serialization not implemented for KimiK25Renderer."
        created = await stack.create()
        llm = client(created["base_url"])
        with pytest.raises(openai.BadRequestError, match="images can't be sent to the engine"):
            await llm.chat.completions.create(model="policy", messages=[look("what is this?", image("cat", 3))])
        assert stack.engine.requests == []

        await llm.chat.completions.create(model="policy", messages=[user("hi")])
        assert len(stack.engine.requests) == 1


async def test_the_first_check_records_why_a_renderer_cannot_send_images() -> None:
    from skycap.tokens.backend import TokensBackend
    from tests.fake_renderer import FakeRenderer

    renderer = FakeRenderer()
    renderer.validate_media_encoding = lambda: "no encoder"  # type: ignore[attr-defined]
    # The engine isn't asked: images that can't be encoded aren't compared.
    renderer.processor_probes = lambda: pytest.fail("probed")  # type: ignore[attr-defined]
    backend = TokensBackend("http://engine", renderer, model="policy")
    await backend.start()
    try:
        assert backend.media_error is None  # nothing at start
        await backend._check_media()
        assert backend.media_error == "no encoder"
    finally:
        await backend.close()


def _has_image(messages: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(m["content"], list) and any(p.get("type") == "image_url" for p in m["content"]) for m in messages
    )


async def _check_with_probes(chat_tokens: Any, *, model: str | None = "policy") -> tuple[Any, str | None]:
    """The check of a renderer probing one text and one image prompt, against an engine answering ``chat_tokens``:
    the engine's chat requests, and the ``media_error`` it left."""
    from aiohttp.test_utils import TestServer

    from skycap.tokens.backend import TokensBackend
    from tests.fake_renderer import FakeRenderer
    from tests.mock_engine import MockEngine

    renderer = FakeRenderer()
    renderer.processor_probes = lambda: [("text", [user("hi")]), ("4x4 image", [look("what?", image("probe", 4))])]  # type: ignore[attr-defined]
    mock = MockEngine()
    mock.chat_tokens = None if chat_tokens is None else lambda messages: chat_tokens(renderer, messages)
    engine_server = TestServer(mock.app())
    await engine_server.start_server()
    try:
        backend = TokensBackend(str(engine_server.make_url("")).rstrip("/"), renderer, model=model)
        await backend.start()
        try:
            assert mock.chat_requests == []  # not at start: the engine may be asleep then
            await backend._check_media()
            return mock, backend.media_error
        finally:
            await backend.close()
    finally:
        await engine_server.close()


def _rendered(renderer: Any, messages: list[dict[str, Any]]) -> int:
    return len(renderer.render(messages, None).token_ids)


async def test_the_check_passes_when_the_engine_processes_images_alike() -> None:
    mock, error = await _check_with_probes(_rendered)

    assert error is None
    assert [len(r["messages"]) for r in mock.chat_requests] == [1, 1]
    assert all(r["max_tokens"] == 1 and r["model"] == "policy" for r in mock.chat_requests)


async def test_the_check_refuses_images_an_engine_processes_differently() -> None:
    _, error = await _check_with_probes(
        lambda renderer, messages: _rendered(renderer, messages) + 3 * _has_image(messages)
    )
    assert error is not None and "image processor settings differ" in error


async def test_a_chat_template_difference_or_an_unreachable_engine_only_warns() -> None:
    # The text probe already differs, so the images can't be compared.
    mock, error = await _check_with_probes(lambda renderer, messages: _rendered(renderer, messages) + 1)
    assert len(mock.chat_requests) == 1 and error is None
    # No chat endpoint (404), and no model name to ask for: skipped.
    _, error = await _check_with_probes(None)
    assert error is None
    mock, error = await _check_with_probes(_rendered, model=None)
    assert mock.chat_requests == [] and error is None


async def test_probes_that_cannot_be_built_leave_images_unchecked() -> None:
    from skycap.tokens.backend import TokensBackend
    from tests.fake_renderer import FakeRenderer

    def no_pillow() -> list[Any]:
        raise ImportError("No module named 'PIL'")

    renderer = FakeRenderer()
    renderer.processor_probes = no_pillow  # type: ignore[attr-defined]
    backend = TokensBackend("http://engine", renderer, model="policy")
    await backend.start()
    try:
        await backend._check_media()
        assert backend.media_error is None
    finally:
        await backend.close()


def _probing(monkeypatch: Any) -> None:
    """Every ``FakeRenderer`` probes one text and one image prompt, as ``_check_with_probes``'s does."""
    from tests.fake_renderer import FakeRenderer

    probes = lambda self: [("text", [user("hi")]), ("4x4 image", [look("what?", image("probe", 4))])]  # noqa: E731
    monkeypatch.setattr(FakeRenderer, "processor_probes", probes, raising=False)


async def test_the_first_call_with_images_checks_the_engine_once(monkeypatch: Any) -> None:
    _probing(monkeypatch)
    async with token_stack(model="policy") as stack:
        stack.engine.chat_tokens = lambda messages: _rendered(stack.renderer, messages)
        llm = client((await stack.create())["base_url"])

        await llm.chat.completions.create(model="policy", messages=[user("hi")])
        assert stack.engine.chat_requests == []  # text only: not checked
        await llm.chat.completions.create(model="policy", messages=[look("what is this?", image("cat", 3))])
        assert len(stack.engine.chat_requests) == 2
        await llm.chat.completions.create(model="policy", messages=[look("and this?", image("dog", 3))])
        assert len(stack.engine.chat_requests) == 2 and len(stack.engine.requests) == 3


async def test_an_engine_that_processes_images_differently_gets_no_images(monkeypatch: Any) -> None:
    import openai

    _probing(monkeypatch)
    async with token_stack(model="policy") as stack:
        stack.engine.chat_tokens = lambda messages: _rendered(stack.renderer, messages) + 3 * _has_image(messages)
        llm = client((await stack.create())["base_url"])
        for _ in range(2):
            with pytest.raises(openai.BadRequestError, match="image processor settings differ"):
                await llm.chat.completions.create(model="policy", messages=[look("what is this?", image("cat", 3))])
        assert stack.engine.requests == [] and len(stack.engine.chat_requests) == 2

        await llm.chat.completions.create(model="policy", messages=[user("hi")])
        assert len(stack.engine.requests) == 1


async def test_an_engine_that_doesnt_answer_the_check_is_left_unchecked(monkeypatch: Any) -> None:
    _probing(monkeypatch)
    # No chat endpoint: the text probe gets a 404.
    async with token_stack(model="policy") as stack:
        llm = client((await stack.create())["base_url"])
        for text in ("what is this?", "and this?"):
            await llm.chat.completions.create(model="policy", messages=[look(text, image("cat", 3))])
        assert len(stack.engine.chat_requests) == 1 and len(stack.engine.requests) == 2


async def test_concurrent_first_calls_with_images_check_once(monkeypatch: Any) -> None:
    import asyncio

    _probing(monkeypatch)
    async with token_stack(model="policy") as stack:
        stack.engine.chat_tokens = lambda messages: _rendered(stack.renderer, messages)
        bases = [(await stack.create())["base_url"] for _ in range(4)]
        await asyncio.gather(
            *(
                client(base).chat.completions.create(model="policy", messages=[look("what is this?", image("cat", 3))])
                for base in bases
            )
        )
        assert len(stack.engine.chat_requests) == 2 and len(stack.engine.requests) == 4


async def test_concurrent_first_calls_with_images_all_wait_for_a_check_that_refuses_them(monkeypatch: Any) -> None:
    import asyncio

    import openai

    _probing(monkeypatch)
    async with token_stack(model="policy") as stack:
        stack.engine.chat_tokens = lambda messages: _rendered(stack.renderer, messages) + 3 * _has_image(messages)
        bases = [(await stack.create())["base_url"] for _ in range(4)]
        results = await asyncio.gather(
            *(
                client(base).chat.completions.create(model="policy", messages=[look("what is this?", image("cat", 3))])
                for base in bases
            ),
            return_exceptions=True,
        )
        assert all(isinstance(result, openai.BadRequestError) for result in results)
        assert stack.engine.requests == [] and len(stack.engine.chat_requests) == 2
