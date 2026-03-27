"""
GPU integration tests for VLLMRenderer with a real vLLM VLM server.

Spins up a Qwen3-VL server via InferenceEngineState and exercises the full
VLLMRenderer control flow: ModelInput -> HTTP /v1/chat/completions/render -> RenderedModelInput.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vllm_renderer_gpu.py -m vllm -v
"""

from __future__ import annotations

import base64
import io
import logging

import pytest
from PIL import Image

from skyrl.backends.renderer import VLLMRenderer
from skyrl.tinker.types import (
    EncodedTextChunk,
    ImageAssetPointerChunk,
    ImageChunk,
    ModelInput,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
TP_SIZE = 1


def _get_test_actor_config(num_inference_engines: int, model: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE * num_inference_engines
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = num_inference_engines
    cfg.generator.inference_engine.tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = MODEL
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


def _make_tiny_jpeg_bytes() -> bytes:
    """Return raw JPEG bytes for a minimal 8x8 red image."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_image_chunk(raw_bytes: bytes | None = None) -> ImageChunk:
    """Build an ImageChunk from raw JPEG bytes (creates a tiny image if not provided)."""
    if raw_bytes is None:
        raw_bytes = _make_tiny_jpeg_bytes()
    b64_encoded = base64.b64encode(raw_bytes)
    return ImageChunk(data=b64_encoded, format="jpeg")


@pytest.fixture(scope="module")
def vllm_renderer(module_scoped_ray_init_fixture):
    """Spin up a real Qwen3-VL vLLM server and yield a VLLMRenderer backed by it."""
    cfg = _get_test_actor_config(num_inference_engines=1, model=MODEL)
    cfg.generator.inference_engine.served_model_name = MODEL
    engines = InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL,
        sleep_level=1,
        engine_init_kwargs={
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 2, "video": 0},
        },
        use_new_inference_servers=True,
    )
    renderer = VLLMRenderer(client=engines.client, model_name=MODEL)
    yield renderer
    engines.close()


# ---------------------------------------------------------------------------
# Text-only fast path
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_text_only_no_http_call(vllm_renderer):
    """Text-only ModelInput should concatenate tokens locally without hitting the server."""
    renderer = vllm_renderer

    tokens_a = [1, 2, 3]
    tokens_b = [4, 5]
    mi = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=tokens_a),
            EncodedTextChunk(tokens=tokens_b),
        ]
    )

    results = renderer([mi])

    assert len(results) == 1
    result = results[0]
    assert result.prompt_ids == tokens_a + tokens_b
    assert result.multi_modal_placeholders is None
    assert result.multi_modal_kwargs is None


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_single_image_render(vllm_renderer):
    """A single ImageChunk should produce valid prompt_ids and exactly one multimodal placeholder."""
    renderer = vllm_renderer

    mi = ModelInput(chunks=[_make_image_chunk()])

    results = renderer([mi])

    assert len(results) == 1
    result = results[0]

    assert len(result.prompt_ids) > 0
    assert all(isinstance(t, int) for t in result.prompt_ids)

    assert result.multi_modal_placeholders is not None
    assert len(result.multi_modal_placeholders) == 1
    ph = result.multi_modal_placeholders[0]
    assert ph.offset == 0
    assert ph.length > 0
    assert ph.offset + ph.length <= len(result.prompt_ids)

    assert result.multi_modal_kwargs is None or result.multi_modal_kwargs == {}


# ---------------------------------------------------------------------------
# Mixed text + image assembly
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_text_image_text_assembly(vllm_renderer):
    """[text, image, text] should preserve chunk order with the placeholder offset matching the first text length."""
    renderer = vllm_renderer

    tokens_before = [10, 11, 12]
    tokens_after = [20, 21]
    mi = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=tokens_before),
            _make_image_chunk(),
            EncodedTextChunk(tokens=tokens_after),
        ]
    )

    results = renderer([mi])
    result = results[0]

    assert result.prompt_ids[: len(tokens_before)] == tokens_before
    assert result.prompt_ids[-len(tokens_after) :] == tokens_after

    assert result.multi_modal_placeholders is not None
    assert len(result.multi_modal_placeholders) == 1
    ph = result.multi_modal_placeholders[0]
    assert ph.offset == len(tokens_before)
    assert ph.length > 0

    placeholder_end = ph.offset + ph.length
    assert result.prompt_ids[placeholder_end:] == tokens_after


# ---------------------------------------------------------------------------
# Two images
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_two_images(vllm_renderer):
    """Two ImageChunks with text between them should yield two placeholders with correct offsets."""
    renderer = vllm_renderer

    text_tokens = [50, 51, 52]
    mi = ModelInput(
        chunks=[
            _make_image_chunk(),
            EncodedTextChunk(tokens=text_tokens),
            _make_image_chunk(),
        ]
    )

    results = renderer([mi])
    result = results[0]

    assert result.multi_modal_placeholders is not None
    assert len(result.multi_modal_placeholders) == 2

    ph0 = result.multi_modal_placeholders[0]
    ph1 = result.multi_modal_placeholders[1]

    assert ph0.offset == 0
    assert ph0.length > 0

    expected_second_offset = ph0.length + len(text_tokens)
    assert ph1.offset == expected_second_offset
    assert ph1.length > 0

    total_expected = ph0.length + len(text_tokens) + ph1.length
    assert len(result.prompt_ids) == total_expected

    mid_start = ph0.length
    mid_end = mid_start + len(text_tokens)
    assert result.prompt_ids[mid_start:mid_end] == text_tokens


# ---------------------------------------------------------------------------
# ImageAssetPointerChunk with a real URL
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_image_asset_pointer(vllm_renderer):
    """ImageAssetPointerChunk with a public URL should produce placeholder tokens."""
    renderer = vllm_renderer

    mi = ModelInput(
        chunks=[
            ImageAssetPointerChunk(
                format="jpeg",
                location="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            ),
        ]
    )

    results = renderer([mi])
    result = results[0]

    assert len(result.prompt_ids) > 0
    assert all(isinstance(t, int) for t in result.prompt_ids)

    assert result.multi_modal_placeholders is not None
    assert len(result.multi_modal_placeholders) == 1
    ph = result.multi_modal_placeholders[0]
    assert ph.length > 0
    assert ph.offset + ph.length <= len(result.prompt_ids)


# ---------------------------------------------------------------------------
# expected_tokens mismatch warning
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_expected_tokens_warning_on_mismatch(vllm_renderer, caplog):
    """A deliberate expected_tokens mismatch should log a warning but not raise."""
    renderer = vllm_renderer

    raw = _make_tiny_jpeg_bytes()
    b64 = base64.b64encode(raw)
    chunk = ImageChunk(data=b64, format="jpeg", expected_tokens=1)
    mi = ModelInput(chunks=[chunk])

    with caplog.at_level(logging.WARNING):
        results = renderer([mi])

    result = results[0]
    assert len(result.prompt_ids) > 0
    assert result.multi_modal_placeholders is not None
    assert result.multi_modal_placeholders[0].length > 1

    assert any(
        "expected_tokens=1" in record.message for record in caplog.records
    ), f"Expected a warning about expected_tokens mismatch, got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Batch rendering (text-only + image in one call)
# ---------------------------------------------------------------------------


@pytest.mark.vllm
def test_batch_rendering(vllm_renderer):
    """Multiple ModelInputs in a single call should all be processed correctly."""
    renderer = vllm_renderer

    text_tokens = [100, 101, 102]
    mi_text = ModelInput(chunks=[EncodedTextChunk(tokens=text_tokens)])

    prefix_tokens = [200, 201]
    mi_image = ModelInput(
        chunks=[
            EncodedTextChunk(tokens=prefix_tokens),
            _make_image_chunk(),
        ]
    )

    results = renderer([mi_text, mi_image])

    assert len(results) == 2

    r_text = results[0]
    assert r_text.prompt_ids == text_tokens
    assert r_text.multi_modal_placeholders is None

    r_image = results[1]
    assert r_image.prompt_ids[: len(prefix_tokens)] == prefix_tokens
    assert len(r_image.prompt_ids) > len(prefix_tokens)
    assert r_image.multi_modal_placeholders is not None
    assert len(r_image.multi_modal_placeholders) == 1
    assert r_image.multi_modal_placeholders[0].offset == len(prefix_tokens)
