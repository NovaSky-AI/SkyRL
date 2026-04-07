"""Unit tests for VLLMRenderer with a mocked RemoteInferenceClient."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.renderer import VLLMRenderer, render_model_input
from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageChunk,
    ModelInput,
)


def _make_mock_client() -> MagicMock:
    client = MagicMock()
    client.render_chat_completion = AsyncMock()
    return client


def _make_text_input(token_lists: list[list[int]]) -> ModelInput:
    return ModelInput(chunks=[EncodedTextChunk(tokens=toks) for toks in token_lists])


def _make_image_chunk(fmt: str = "jpeg") -> ImageChunk:
    return ImageChunk(data=b"\xff\xd8\xff\xe0", format=fmt)


def _make_render_response(
    token_ids: list[int],
    placeholders: list[dict],
    kwargs_data: list[str] | None = None,
) -> dict:
    """Build a canned response matching the shape of vLLM's /v1/chat/completions/render."""
    features: dict = {"mm_placeholders": {"image": placeholders}}
    if kwargs_data is not None:
        features["kwargs_data"] = {"image": kwargs_data}
    return {"token_ids": token_ids, "features": features}


class TestVLLMRendererTextOnly:
    @pytest.mark.asyncio
    async def test_concatenates_tokens(self):
        client = _make_mock_client()
        renderer = VLLMRenderer(client, model_name="test-model")

        mi = _make_text_input([[1, 2, 3], [4, 5]])
        results = await renderer([mi])

        assert len(results) == 1
        assert results[0].prompt_ids == [1, 2, 3, 4, 5]
        assert results[0].multi_modal_placeholders is None
        assert results[0].multi_modal_kwargs is None
        client.render_chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_text_inputs(self):
        client = _make_mock_client()
        renderer = VLLMRenderer(client, model_name="test-model")

        inputs = [_make_text_input([[10, 20]]), _make_text_input([[30]])]
        results = await renderer(inputs)

        assert len(results) == 2
        assert results[0].prompt_ids == [10, 20]
        assert results[1].prompt_ids == [30]
        client.render_chat_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_matches_render_model_input(self):
        """Text-only fast path should produce the same result as the free function."""
        client = _make_mock_client()
        renderer = VLLMRenderer(client, model_name="test-model")

        mi = _make_text_input([[1, 2], [3, 4, 5]])
        vllm_result = (await renderer([mi]))[0]
        free_fn_result = render_model_input([mi])[0]

        assert vllm_result.prompt_ids == free_fn_result.prompt_ids


class TestVLLMRendererMixed:
    @pytest.mark.asyncio
    async def test_text_image_text(self):
        """Interleaved [text, image, text] should produce tokens in correct order."""
        client = _make_mock_client()
        placeholder_tokens = [50, 51, 52]
        client.render_chat_completion.return_value = _make_render_response(
            token_ids=placeholder_tokens,
            placeholders=[{"offset": 0, "length": 3}],
            kwargs_data=["img-data"],
        )

        renderer = VLLMRenderer(client, model_name="test-model")
        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[1, 2]),
                _make_image_chunk(),
                EncodedTextChunk(tokens=[3, 4]),
            ]
        )
        results = await renderer([mi])

        assert results[0].prompt_ids == [1, 2, 50, 51, 52, 3, 4]
        assert results[0].multi_modal_placeholders is not None
        ph = results[0].multi_modal_placeholders[0]
        assert ph.offset == 2
        assert ph.length == 3
        assert results[0].multi_modal_kwargs == {"image": ["img-data"]}

    @pytest.mark.asyncio
    async def test_two_images(self):
        """Two images should produce two placeholder regions."""
        client = _make_mock_client()
        client.render_chat_completion.return_value = _make_render_response(
            token_ids=[10, 11, 20, 21, 22],
            placeholders=[
                {"offset": 0, "length": 2},
                {"offset": 2, "length": 3},
            ],
            kwargs_data=["data-a", "data-b"],
        )

        renderer = VLLMRenderer(client, model_name="test-model")
        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=[1]),
                _make_image_chunk(),
                _make_image_chunk(),
                EncodedTextChunk(tokens=[2]),
            ]
        )
        results = await renderer([mi])

        assert results[0].prompt_ids == [1, 10, 11, 20, 21, 22, 2]
        phs = results[0].multi_modal_placeholders
        assert phs is not None and len(phs) == 2
        assert phs[0].offset == 1
        assert phs[0].length == 2
        assert phs[1].offset == 3
        assert phs[1].length == 3
        assert results[0].multi_modal_kwargs == {"image": ["data-a", "data-b"]}
