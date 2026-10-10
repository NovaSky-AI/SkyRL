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


VL_TOKENIZER = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def vl_renderer() -> RenderersRenderer:
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    from transformers import AutoProcessor

    # Only missing model files skip (transformers raises OSError without network or cache); a broken
    # renderer or processor setup fails the test.
    try:
        AutoProcessor.from_pretrained(VL_TOKENIZER)
    except OSError as error:
        pytest.skip(f"processor unavailable: {error}")
    return RenderersRenderer(VL_TOKENIZER, size=1, renderer="qwen3-vl", processor_kwargs={"max_pixels": 200704})


def _png(color: str, size: tuple[int, int]) -> dict:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, "PNG")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()},
    }


def test_an_image_turn_bridges_to_exactly_the_full_render(vl_renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(VL_TOKENIZER)
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    first_message = {"role": "user", "content": [_png("red", (1024, 768)), {"type": "text", "text": "color?"}]}
    second_message = {"role": "user", "content": [_png("blue", (96, 64)), {"type": "text", "text": "and?"}]}
    first = vl_renderer.render([first_message], None)
    completion = tokenizer.encode("red<|im_end|>", add_special_tokens=False)
    bridged = vl_renderer.bridge(first.token_ids, completion, [second_message], None, first.media)
    full = vl_renderer.render([first_message, {"role": "assistant", "content": "red"}, second_message], None)

    # The processor kwargs apply: 1024x768 under max_pixels=200704 is a 24x32 grid, 192 placeholders.
    (image,) = first.media
    assert image.length == 192 and image.data["image_grid_thw"].tolist() == [[1, 24, 32]]
    assert bridged is not None and bridged.token_ids == full.token_ids
    assert [m.hash for m in bridged.media] == [full.media[1].hash]
    assert [(m.offset, m.length) for m in [*first.media, *bridged.media]] == [(m.offset, m.length) for m in full.media]
    for item in full.media:
        assert full.token_ids[item.offset : item.offset + item.length] == [pad] * item.length


def test_processor_kwargs_for_a_text_only_model_are_refused(renderer: RenderersRenderer) -> None:
    with pytest.raises(ValueError, match="processor_kwargs are for a multimodal model"):
        RenderersRenderer(TOKENIZER, size=1, processor_kwargs={"max_pixels": 200704})


QWEN35_TOKENIZER = "Qwen/Qwen3.5-0.8B"
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


@pytest.fixture(scope="module")
def qwen35_renderer() -> RenderersRenderer:
    try:
        # Thinking on, as Qwen3.5's larger models default to; the small ones' template leaves it off.
        return RenderersRenderer(QWEN35_TOKENIZER, size=1, chat_template_kwargs={"enable_thinking": True})
    except OSError as error:  # no network and no cache
        pytest.skip(f"tokenizer unavailable: {error}")


def test_tools_render_as_the_chat_template_prints_them(qwen35_renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    # vLLM hands the template the request's tools as sent, so the wrapper is in the prompt.
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "list files"}]
    template = load_tokenizer(QWEN35_TOKENIZER).apply_chat_template(
        messages,
        tools=[BASH_TOOL],
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
        return_dict=False,
    )

    assert qwen35_renderer.render(messages, [BASH_TOOL]).token_ids == list(template)


async def test_thinking_opened_by_the_generation_prompt_is_split_and_its_replay_bridges(
    qwen35_renderer: RenderersRenderer,
) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(QWEN35_TOKENIZER)
    # Qwen3.5's generation prompt ends in `<think>\n`, so the completion starts inside the thinking block.
    completion = tokenizer.encode(
        "look first\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>",
        add_special_tokens=False,
    )

    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.server.backend.renderer = qwen35_renderer  # type: ignore[attr-defined]
        created = await stack.create()
        llm = client(created["base_url"])
        first = await llm.chat.completions.create(model="policy", messages=[user("q")], tools=[BASH_TOOL])
        reply = first.choices[0].message.model_dump(exclude_none=True)
        assert reply["reasoning_content"] == "look first"
        assert "</think>" not in (reply.get("content") or "")
        assert reply["tool_calls"][0]["function"] == {"name": "bash", "arguments": '{"command":"ls"}'}

        tool_result = {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "a.txt"}
        await llm.chat.completions.create(model="policy", messages=[user("q"), reply, tool_result], tools=[BASH_TOOL])

        first_request, second_request = stack.engine.requests
        exact_prefix = first_request["token_ids"] + completion
        assert second_request["token_ids"][: len(exact_prefix)] == exact_prefix
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0
