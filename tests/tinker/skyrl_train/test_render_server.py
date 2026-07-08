"""End-to-end test for the CPU-only render server used by the VLM training path.

Starts vLLM's GPU-less render server (``CPURenderServer``) for a tiny VLM and
renders an image-bearing ModelInput through ``VLLMRenderer`` backed by
``RenderServerClient``. No inference engines (and no GPUs) are involved. Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest pytest tests/tinker/skyrl_train/test_render_server.py
"""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib

import pytest
import torch

# Skip if the SkyRL-Train backend deps (ray/vllm) cannot be imported
pytest.importorskip("skyrl.backends.skyrl_train_backend")

from skyrl.backends.renderer import VLLMRenderer  # noqa: E402
from skyrl.backends.skyrl_train.inference_servers.render_server import (  # noqa: E402
    CPURenderServer,
    RenderServerClient,
)
from skyrl.tinker import types  # noqa: E402

TINY_VLM = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _tiny_png(w: int = 64, h: int = 64) -> bytes:
    """Minimal valid RGB PNG (no PIL dependency)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x7f\x7f\x7f" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@pytest.fixture(scope="module")
def render_server():
    server = CPURenderServer(TINY_VLM)
    try:
        server.start()
        yield server
    finally:
        server.shutdown()


def test_render_image_input_via_cpu_server(render_server):
    """VLLMRenderer against the CPU render server yields placeholder tokens and vision tensors."""
    renderer = VLLMRenderer(RenderServerClient(render_server.url), model_name=TINY_VLM)

    prefix_tokens = [1, 2, 3]
    suffix_tokens = [7, 8]
    model_input = types.ModelInput(
        chunks=[
            types.EncodedTextChunk(tokens=prefix_tokens),
            types.ImageChunk(data=base64.b64encode(_tiny_png()), format="png"),
            types.EncodedTextChunk(tokens=suffix_tokens),
        ]
    )

    rendered = asyncio.run(renderer([model_input]))[0]

    # Placeholder tokens are spliced between the text chunks.
    assert rendered.multi_modal_placeholders is not None
    (placeholder,) = rendered.multi_modal_placeholders
    assert placeholder.offset == len(prefix_tokens)
    assert placeholder.length > 0
    assert len(rendered.prompt_ids) == len(prefix_tokens) + placeholder.length + len(suffix_tokens)
    assert rendered.prompt_ids[: len(prefix_tokens)] == prefix_tokens
    assert rendered.prompt_ids[-len(suffix_tokens) :] == suffix_tokens

    # Vision tensors are decoded from the server's serialized kwargs_data.
    mm_kwargs = rendered.multi_modal_kwargs
    assert isinstance(mm_kwargs["pixel_values"], torch.Tensor)
    assert isinstance(mm_kwargs["image_grid_thw"], torch.Tensor)
    assert mm_kwargs["image_grid_thw"].shape[-1] == 3  # (t, h, w) per image


def test_render_text_only_makes_no_http_call(render_server):
    """Text-only inputs are rendered locally even when a render client is wired."""

    class _ExplodingClient:
        async def render_chat_completion(self, request_payload):
            raise AssertionError("text-only input must not hit the render endpoint")

    renderer = VLLMRenderer(_ExplodingClient(), model_name=TINY_VLM)
    model_input = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])])
    rendered = asyncio.run(renderer([model_input]))[0]
    assert rendered.prompt_ids == [1, 2, 3]
