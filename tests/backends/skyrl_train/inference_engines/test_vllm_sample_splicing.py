"""CPU-only unit tests for the /sample endpoint splicing helpers.

These tests cover:
  - _extract_render_info: parsing placeholder tokens and mm_hash from a render response
  - _assemble_tokens_from_chunks: assembling token_ids + features from ModelInput chunks

No vLLM, no GPU, no real HTTP calls required.
"""

import base64
import pytest
from unittest.mock import AsyncMock, patch

# Import helpers from the vLLM-free helper module so tests run without a GPU env.
from skyrl.backends.skyrl_train.inference_servers._sample_helpers import (
    _extract_render_info,
    _assemble_tokens_from_chunks,
)


# ---------------------------------------------------------------------------
# Helpers for building fake render responses
# ---------------------------------------------------------------------------


def _make_render_resp(token_ids: list[int], *, offset: int, length: int, mm_hash: str) -> list:
    """Build a fake /v1/chat/completions/render response.

    The render endpoint returns ``[conversation, engine_prompts]`` where each
    engine prompt uses a ``features`` list instead of separate
    ``mm_placeholders`` / ``mm_hashes`` dicts.
    """
    conversation = [{"role": "user", "content": [{"type": "image_url"}]}]
    engine_prompt = {
        "type": "multimodal",
        "prompt_token_ids": token_ids,
        "features": [
            {
                "modality": "image",
                "mm_hash": mm_hash,
                "offset": offset,
                "length": length,
                "kwargs_data": None,
            }
        ],
    }
    return [conversation, [engine_prompt]]


def _dummy_image_b64() -> str:
    """Return a trivially small base64 payload (not a real image)."""
    return base64.b64encode(b"\xff\xd8\xff").decode()  # fake JPEG header bytes


# ---------------------------------------------------------------------------
# _extract_render_info tests
# ---------------------------------------------------------------------------


class TestExtractRenderInfo:
    def test_extracts_placeholder_tokens(self):
        """Placeholder tokens are the slice token_ids[offset:offset+length]."""
        token_ids = [10, 20, 30, 40, 50, 60]
        render_resp = _make_render_resp(token_ids, offset=2, length=3, mm_hash="abc123")
        placeholder_tokens, mm_hash = _extract_render_info(render_resp)
        assert placeholder_tokens == [30, 40, 50]
        assert mm_hash == "abc123"

    def test_placeholder_at_start(self):
        token_ids = [100, 200, 300, 400]
        render_resp = _make_render_resp(token_ids, offset=0, length=2, mm_hash="hash0")
        placeholder_tokens, mm_hash = _extract_render_info(render_resp)
        assert placeholder_tokens == [100, 200]
        assert mm_hash == "hash0"

    def test_placeholder_at_end(self):
        token_ids = [1, 2, 3, 4, 5]
        render_resp = _make_render_resp(token_ids, offset=3, length=2, mm_hash="hashEnd")
        placeholder_tokens, mm_hash = _extract_render_info(render_resp)
        assert placeholder_tokens == [4, 5]
        assert mm_hash == "hashEnd"

    def test_single_token_placeholder(self):
        token_ids = [7, 8, 9]
        render_resp = _make_render_resp(token_ids, offset=1, length=1, mm_hash="single")
        placeholder_tokens, mm_hash = _extract_render_info(render_resp)
        assert placeholder_tokens == [8]


# ---------------------------------------------------------------------------
# _assemble_tokens_from_chunks tests (async; patches out real HTTP calls)
# ---------------------------------------------------------------------------

# Fake render response used across multiple image tests
_PLACEHOLDER_TOKENS_A = [1000, 1001, 1002]  # length 3
_MM_HASH_A = "hash_image_a"
_RENDER_RESP_A = _make_render_resp(
    token_ids=[999] + _PLACEHOLDER_TOKENS_A + [998],
    offset=1,
    length=3,
    mm_hash=_MM_HASH_A,
)

_PLACEHOLDER_TOKENS_B = [2000, 2001]  # length 2
_MM_HASH_B = "hash_image_b"
_RENDER_RESP_B = _make_render_resp(
    token_ids=[888] + _PLACEHOLDER_TOKENS_B + [887],
    offset=1,
    length=2,
    mm_hash=_MM_HASH_B,
)


@pytest.mark.asyncio
class TestAssembleTokensFromChunks:
    async def test_text_only_passthrough(self):
        """All encoded_text chunks → concatenated tokens, no render call."""
        chunks = [
            {"type": "encoded_text", "tokens": [1, 2, 3]},
            {"type": "encoded_text", "tokens": [4, 5]},
        ]
        token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")
        assert token_ids == [1, 2, 3, 4, 5]
        assert features is None

    async def test_text_only_empty(self):
        chunks = [{"type": "encoded_text", "tokens": []}]
        token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")
        assert token_ids == []
        assert features is None

    async def test_single_image_splice(self):
        """ImageChunk between two text chunks → placeholder tokens inserted at correct offset."""
        chunks = [
            {"type": "encoded_text", "tokens": [10, 11]},
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg"},
            {"type": "encoded_text", "tokens": [20, 21]},
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(return_value=_RENDER_RESP_A),
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        # Text before image (offset 0..1), placeholder (offset 2..4), text after (offset 5..6)
        assert token_ids == [10, 11] + _PLACEHOLDER_TOKENS_A + [20, 21]
        assert features is not None
        assert len(features) == 1
        f = features[0]
        assert f["modality"] == "image"
        assert f["mm_hash"] == _MM_HASH_A
        assert f["offset"] == 2  # after the two text tokens
        assert f["length"] == 3
        assert f["kwargs_data"] is None

    async def test_image_at_start(self):
        """ImageChunk is the first chunk — offset should be 0."""
        chunks = [
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg"},
            {"type": "encoded_text", "tokens": [30, 31]},
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(return_value=_RENDER_RESP_A),
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        assert token_ids == _PLACEHOLDER_TOKENS_A + [30, 31]
        assert features[0]["offset"] == 0
        assert features[0]["length"] == 3

    async def test_image_at_end(self):
        """ImageChunk is the last chunk — text precedes it."""
        chunks = [
            {"type": "encoded_text", "tokens": [50, 51, 52]},
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg"},
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(return_value=_RENDER_RESP_A),
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        assert token_ids == [50, 51, 52] + _PLACEHOLDER_TOKENS_A
        assert features[0]["offset"] == 3  # after the three text tokens
        assert features[0]["length"] == 3

    async def test_multiple_images(self):
        """Two images in sequence → both features at correct offsets."""
        chunks = [
            {"type": "encoded_text", "tokens": [1, 2]},
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg"},
            {"type": "encoded_text", "tokens": [3]},
            {"type": "image", "data": _dummy_image_b64(), "format": "png"},
        ]

        render_side_effect = [_RENDER_RESP_A, _RENDER_RESP_B]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(side_effect=render_side_effect),
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        # [1,2] + placeholder_A(3 tokens) + [3] + placeholder_B(2 tokens)
        expected_ids = [1, 2] + _PLACEHOLDER_TOKENS_A + [3] + _PLACEHOLDER_TOKENS_B
        assert token_ids == expected_ids

        assert len(features) == 2
        # First image: offset 2 (after [1,2]), length 3
        assert features[0]["offset"] == 2
        assert features[0]["length"] == 3
        assert features[0]["mm_hash"] == _MM_HASH_A
        # Second image: offset 2+3+1=6 (after [1,2] + placeholder_A + [3]), length 2
        assert features[1]["offset"] == 6
        assert features[1]["length"] == 2
        assert features[1]["mm_hash"] == _MM_HASH_B

    async def test_consistency_check_passes(self):
        """expected_tokens == actual placeholder length → no error raised."""
        chunks = [
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg", "expected_tokens": 3},
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(return_value=_RENDER_RESP_A),  # renders 3 placeholder tokens
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        assert token_ids == _PLACEHOLDER_TOKENS_A
        assert features[0]["length"] == 3

    async def test_consistency_check_fails(self):
        """expected_tokens != actual placeholder length → ValueError raised."""
        chunks = [
            {"type": "image", "data": _dummy_image_b64(), "format": "jpeg", "expected_tokens": 99},
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image",
            new=AsyncMock(return_value=_RENDER_RESP_A),  # renders 3 placeholder tokens, not 99
        ):
            with pytest.raises(ValueError, match="expected_tokens=99"):
                await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

    async def test_image_asset_pointer_chunk(self):
        """ImageAssetPointerChunk uses _render_image_url instead of _render_image."""
        chunks = [
            {"type": "encoded_text", "tokens": [5]},
            {
                "type": "image_asset_pointer",
                "format": "jpeg",
                "location": "http://example.com/image.jpg",
            },
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image_url",
            new=AsyncMock(return_value=_RENDER_RESP_B),
        ):
            token_ids, features = await _assemble_tokens_from_chunks(chunks, "http://fake", "model")

        assert token_ids == [5] + _PLACEHOLDER_TOKENS_B
        assert features[0]["offset"] == 1
        assert features[0]["length"] == 2
        assert features[0]["mm_hash"] == _MM_HASH_B

    async def test_image_asset_pointer_consistency_check_fails(self):
        """ImageAssetPointerChunk expected_tokens mismatch → ValueError."""
        chunks = [
            {
                "type": "image_asset_pointer",
                "format": "jpeg",
                "location": "http://example.com/image.jpg",
                "expected_tokens": 50,
            },
        ]

        with patch(
            "skyrl.backends.skyrl_train.inference_servers._sample_helpers._render_image_url",
            new=AsyncMock(return_value=_RENDER_RESP_B),  # renders 2 placeholder tokens, not 50
        ):
            with pytest.raises(ValueError, match="expected_tokens=50"):
                await _assemble_tokens_from_chunks(chunks, "http://fake", "model")
