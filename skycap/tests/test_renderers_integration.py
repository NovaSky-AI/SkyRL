"""The real ``renderers`` library on a real tokenizer.

Needs the ``tokens`` extra and the Qwen3-0.6B tokenizer (downloaded, or in the
local Hugging Face cache); skipped otherwise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("renderers")

from skycap.samples import build_samples  # noqa: E402
from skycap.tokens.renderer import RenderersRenderer  # noqa: E402
from tests.test_tokens import client, token_stack  # noqa: E402

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
