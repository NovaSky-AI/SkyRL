"""The real ``renderers`` library on a real tokenizer.

Needs the ``tokens`` extra and the Qwen3-0.6B tokenizer (downloaded, or in the
local Hugging Face cache); skipped otherwise.
"""

from __future__ import annotations

import json

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


async def test_malformed_json_tool_attempt_is_text_without_promoting_finish_reason(renderer: RenderersRenderer) -> None:
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)
    attempt = '{"name": "search", "arguments":'
    completion = tokenizer.encode(
        f"<think>\nhmm\n</think>\n\n<tool_call>\n{attempt}\n</tool_call><|im_end|>",
        add_special_tokens=False,
    )
    tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]

    async with token_stack(completion=lambda prompt, sampling: completion) as stack:
        stack.server.backend.renderer = renderer  # type: ignore[attr-defined]
        created = await stack.create()
        first = await client(created["base_url"]).chat.completions.create(
            model="policy", messages=[user("q")], tools=tools
        )

        assert first.choices[0].finish_reason == "stop"
        assert first.choices[0].message.tool_calls is None
        assert first.choices[0].message.content == attempt
        assert first.choices[0].message.reasoning_content == "hmm"
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)
        assert sample.input_ids == stack.engine.requests[0]["token_ids"] + completion
        assert sum(sample.loss_mask) == len(completion)


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


def test_qwen35_refuses_developer_messages_and_renders_them_folded(prefilling_renderer: RenderersRenderer) -> None:
    """Why token mode folds ``developer`` (``tokens.developer``): Qwen3.5's template has no such role, and its
    renderer raises on one, and on a system message that isn't first. Codex opens with several."""
    from skycap.tokens.developer import fold_developer, template_knows_developer

    messages = [
        {"role": "system", "content": "You are Codex."},
        {"role": "developer", "content": "Sandbox: workspace-write."},
        {"role": "developer", "content": "Approvals: never."},
        user("hi"),
    ]

    assert prefilling_renderer.chat_template is not None
    assert not template_knows_developer(prefilling_renderer.chat_template)
    with pytest.raises(ValueError):
        prefilling_renderer.render(messages, None)
    folded = fold_developer(messages, prefilling_renderer.chat_template)
    assert folded[0] == {
        "role": "system",
        "content": "You are Codex.\n\nSandbox: workspace-write.\n\nApprovals: never.",
    }
    assert prefilling_renderer.render(folded, None).token_ids


@pytest.fixture(scope="module", params=[PREFILLING_TOKENIZER, "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"])
def xml_renderer(request: pytest.FixtureRequest) -> RenderersRenderer:
    try:
        return RenderersRenderer(request.param, size=1, chat_template_kwargs={"enable_thinking": True})
    except Exception as error:  # noqa: BLE001 - no network and no cache
        pytest.skip(f"tokenizer unavailable: {error}")


async def test_invalid_numeric_tool_argument_can_be_corrected_without_losing_sampled_tokens(
    xml_renderer: RenderersRenderer,
) -> None:
    from renderers import ToolCallParseStatus
    from renderers.base import load_tokenizer

    tokenizer = load_tokenizer(xml_renderer.name)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_well",
                "parameters": {"type": "object", "properties": {"Xylene": {"type": "number"}}},
            },
        }
    ]
    messages = [user("Run a well.")]
    prompt = xml_renderer.render(messages, tools).token_ids
    assert tokenizer.decode(prompt).rstrip().endswith("<think>")
    completion_text = (
        "Let me run the first well.\n</think>\n"
        "<tool_call>\n<function=run_well>\n<parameter=Xylene>\n"
        "120 plus the remaining volume\n</parameter>\n</function>\n</tool_call><|im_end|>"
    )
    completion = tokenizer.encode(completion_text, add_special_tokens=False)
    correction = tokenizer.encode(
        completion_text.replace("120 plus the remaining volume", "120"), add_special_tokens=False
    )
    with xml_renderer._checkout() as (parser, _):
        parsed = parser.parse_response(completion, tools=tools, prompt_ids=prompt)
    assert parsed.content == ""
    assert parsed.tool_calls[0].status == ToolCallParseStatus.INVALID_JSON

    completions = iter([completion, correction])
    async with token_stack(completion=lambda prompt, sampling: next(completions)) as stack:
        stack.server.backend.renderer = xml_renderer  # type: ignore[attr-defined]
        created = await stack.create()
        llm = client(created["base_url"])
        first = await llm.chat.completions.create(model="policy", messages=messages, tools=tools)
        reply = first.choices[0].message.model_dump(exclude_none=True)

        assert first.choices[0].finish_reason == "tool_calls"
        assert reply["content"] == ""
        assert reply["reasoning_content"] == "Let me run the first well."
        (call,) = reply["tool_calls"]
        assert call["function"]["name"] == "run_well"
        assert json.loads(call["function"]["arguments"]) == {"Xylene": "120 plus the remaining volume"}

        messages += [
            reply,
            {"role": "tool", "tool_call_id": call["id"], "content": "Invalid arguments: Xylene must be a number."},
        ]
        second = await llm.chat.completions.create(model="policy", messages=messages, tools=tools)
        assert json.loads(second.choices[0].message.tool_calls[0].function.arguments) == {"Xylene": 120}

        first_request, second_request = stack.engine.requests
        exact_prefix = first_request["token_ids"] + completion
        assert second_request["token_ids"][: len(exact_prefix)] == exact_prefix
        (sample,) = build_samples(stack.server.trajectories[created["id"]].graph)
        assert sample.input_ids == second_request["token_ids"] + correction
        assert sum(sample.loss_mask) == len(completion) + len(correction)
        assert (await stack.finish(created["id"]))["unbridged_calls"] == 0
