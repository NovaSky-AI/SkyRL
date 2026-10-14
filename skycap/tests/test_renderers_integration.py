"""The real ``renderers`` library on a real tokenizer.

Needs the ``tokens`` extra and the Qwen3-0.6B tokenizer (downloaded, or in the
local Hugging Face cache); skipped otherwise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("renderers")

from skycap.samples import build_samples  # noqa: E402
from skycap.tokens.renderer import RenderersRenderer  # noqa: E402
from tests.test_tokens import client, token_stack, user  # noqa: E402

TOKENIZER = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def renderer() -> RenderersRenderer:
    try:
        return RenderersRenderer(TOKENIZER, size=1)
    except Exception as error:  # noqa: BLE001 - no network and no cache
        pytest.skip(f"tokenizer unavailable: {error}")


def test_render_bridge_and_parse_agree(renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    first = renderer.render([{"role": "user", "content": "hi"}], None)
    completion = tokenizer.encode("<think>\nhmm\n</think>\n\nhello<|im_end|>", add_special_tokens=False)
    message = renderer.parse(completion, None)

    assert message == {"role": "assistant", "content": "hello", "reasoning_content": "hmm"}
    bridged = renderer.bridge(first.token_ids, completion, [{"role": "user", "content": "more"}], None)
    assert bridged is not None
    assert bridged.token_ids[: bridged.reused] == first.token_ids + completion
    assert len(bridged.tail_indices) == len(bridged.token_ids) - bridged.reused
    assert set(bridged.tail_indices) == {-1, 0}


def test_provider_specific_fields_do_not_change_rendered_tokens(renderer: RenderersRenderer) -> None:
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    with_metadata = [*messages[:-1], {**messages[-1], "provider_specific_fields": {"response_id": "resp_1"}}]

    assert renderer.render(messages, None).token_ids == renderer.render(with_metadata, None).token_ids


def test_empty_tool_call_content_renders_like_absent_content(renderer: RenderersRenderer) -> None:
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
    tool_call = {"id": "call_0", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    reply = {"role": "assistant", "content": "", "reasoning_content": "hmm", "tool_calls": [tool_call]}
    replay = {key: value for key, value in reply.items() if key != "content"}
    replay["provider_specific_fields"] = {"refusal": None}

    assert (
        renderer.render([{"role": "user", "content": "hi"}, reply], tools).token_ids
        == renderer.render([{"role": "user", "content": "hi"}, replay], tools).token_ids
    )


async def test_real_renderer_bridges_empty_content_tool_call_replay(renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    completion = tokenizer.encode(
        '<think>\nhmm\n</think>\n\n<tool_call>\n{"name": "search", "arguments": {}}\n</tool_call><|im_end|>',
        add_special_tokens=False,
    )
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.server.backend.renderer = renderer  # type: ignore[attr-defined]
        created = await stack.create()
        llm = client(created["base_url"])
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

        first_request, second_request = stack.engine.requests
        exact_prefix = first_request["token_ids"] + completion
        assert second_request["token_ids"][: len(exact_prefix)] == exact_prefix
        graph = stack.server.trajectories[created["id"]].graph
        model_nodes = [node for node in graph if node.author == "model"]
        assert len(model_nodes) == 2
        assert model_nodes[0].id in graph.path_to(model_nodes[1].id)
        assert [call.bridged for node in model_nodes for call in node.calls] == [None, True]
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0


async def test_a_conversation_through_the_real_renderer(renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    reply = tokenizer.encode("<think>\nok\n</think>\n\nsure<|im_end|>", add_special_tokens=False)

    async with token_stack(completion=lambda prompt, sampling: reply) as stack:
        stack.server.backend.renderer = renderer  # type: ignore[attr-defined]
        created = await stack.create()
        llm = client(created["base_url"])
        messages = [{"role": "user", "content": "hi"}]
        first = await llm.chat.completions.create(model="policy", messages=messages)
        assert first.choices[0].message.content == "sure"
        messages += [
            first.choices[0].message.model_dump(exclude_none=True),
            {"role": "user", "content": "again"},
        ]
        await llm.chat.completions.create(model="policy", messages=messages)

        one, two = stack.engine.requests
        assert two["token_ids"][: len(one["token_ids"]) + len(reply)] == one["token_ids"] + reply
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)
        assert sample.input_ids == two["token_ids"] + reply
        assert sum(sample.loss_mask) == 2 * len(reply)


def test_decoded_spans_are_whole_characters_and_rejoin_to_the_text(renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    text = "<|im_start|>assistant\n<think>\nhé 🙂 中文\n</think>\n\nok<|im_end|>"
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded, offsets = renderer.decode_spans(token_ids)

    assert decoded == tokenizer.decode(token_ids, skip_special_tokens=False)
    data = decoded.encode()
    bounds = [*offsets, len(data)]
    spans = [data[start:end] for start, end in zip(bounds, bounds[1:])]
    assert all("\ufffd" not in span.decode() for span in spans)
    assert b"".join(spans) == data


PREFILLING_TOKENIZER = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="module")
def prefilling_renderer() -> RenderersRenderer:
    try:
        # Thinking is off by default below 4B; on, the template prefills the reasoning block.
        return RenderersRenderer(PREFILLING_TOKENIZER, size=1, chat_template_kwargs={"enable_thinking": True})
    except Exception as error:  # noqa: BLE001 - no network and no cache
        pytest.skip(f"tokenizer unavailable: {error}")


async def test_reasoning_is_split_when_the_prompt_opened_the_think_block(
    prefilling_renderer: RenderersRenderer,
) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(PREFILLING_TOKENIZER)
    # Qwen3.5 with thinking on ends the generation prompt in `<think>\n`, so the model's
    # completion starts inside the reasoning block and never samples the opening tag.
    prompt = prefilling_renderer.render([{"role": "user", "content": "hi"}], None).token_ids
    assert tokenizer.decode(prompt).endswith("<think>\n")
    completion = tokenizer.encode("hmm\n</think>\n\nhello<|im_end|>", add_special_tokens=False)

    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.server.backend.renderer = prefilling_renderer  # type: ignore[attr-defined]
        created = await stack.create()
        reply = await client(created["base_url"]).chat.completions.create(model="policy", messages=[user("hi")])

    message = reply.choices[0].message.model_dump(exclude_none=True)
    assert message["content"] == "hello"
    assert message["reasoning_content"] == "hmm"
