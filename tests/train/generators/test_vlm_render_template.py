"""The VLM render path honors generator.chat_template (issue #2075).

Unit seam: _render_conversation built without __init__ so the test needs no
engines. Asserts the request body carries the custom template exactly when
configured, and stays untouched otherwise.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator


def _bare_generator(custom_template, kwargs=None):
    gen = SkyRLVLMGymGenerator.__new__(SkyRLVLMGymGenerator)
    gen.custom_chat_template = custom_template
    gen.generator_cfg = SimpleNamespace(chat_template_kwargs=kwargs or {})
    gen.inference_engine_client = SimpleNamespace(
        model_name="test-model",
        render_chat_completion=AsyncMock(return_value={"token_ids": [1, 2], "features": None}),
    )
    return gen


@pytest.mark.asyncio
async def test_custom_template_reaches_render_body():
    gen = _bare_generator("{{ messages }}", {"enable_thinking": True})
    await gen._render_conversation([{"role": "user", "content": "hi"}])
    body = gen.inference_engine_client.render_chat_completion.call_args.args[0]["json"]
    assert body["chat_template"] == "{{ messages }}"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.asyncio
async def test_no_template_leaves_body_untouched():
    gen = _bare_generator(None)
    await gen._render_conversation([{"role": "user", "content": "hi"}])
    body = gen.inference_engine_client.render_chat_completion.call_args.args[0]["json"]
    assert "chat_template" not in body and "chat_template_kwargs" not in body
