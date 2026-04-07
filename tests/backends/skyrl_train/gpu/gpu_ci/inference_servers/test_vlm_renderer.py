"""
Integration tests for VLLMRenderer against a real vLLM instance with a VLM.

Requires a local vLLM install with /v1/chat/completions/render support.
Run with:
    SKYRL_LOCAL_VLLM=1 python -m pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vlm_renderer.py -m vllm -v
"""

import base64
import io
import os

import pytest
from PIL import Image
from transformers import AutoTokenizer

from skyrl.backends.renderer import VLLMRenderer
from skyrl.tinker.types import EncodedTextChunk, ImageChunk, ModelInput
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

requires_local_vllm = pytest.mark.skipif(
    os.environ.get("SKYRL_LOCAL_VLLM") != "1",
    reason="Requires local vLLM with /v1/chat/completions/render support",
)

MODEL_QWEN3_VL = "Qwen/Qwen3-VL-2B-Instruct"
TP_SIZE = 1


def _get_config(num_inference_engines: int = 1) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_QWEN3_VL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE * num_inference_engines
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = num_inference_engines
    cfg.generator.inference_engine.tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


def _make_tiny_jpeg_b64() -> bytes:
    """Create a tiny JPEG and return it base64-encoded, matching the tinker API input format."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


def _tokenize_text(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3_VL, trust_remote_code=True)
    return tokenizer.encode(text, add_special_tokens=False)


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_renderer_text_only(module_scoped_ray_init_fixture):
    """Text-only inputs should not trigger any HTTP calls to the render endpoint."""
    cfg = _get_config()
    async with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN3_VL,
        sleep_level=1,
        engine_init_kwargs={
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
        },
        use_new_inference_servers=True,
    ) as engines:
        renderer = VLLMRenderer(engines.client, model_name=MODEL_QWEN3_VL)

        tokens = _tokenize_text("Hello, world!")
        mi = ModelInput(chunks=[EncodedTextChunk(tokens=tokens)])
        results = await renderer([mi])

        assert len(results) == 1
        assert results[0].prompt_ids == tokens
        assert results[0].multi_modal_placeholders is None
        assert results[0].multi_modal_kwargs is None


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_renderer_with_image(module_scoped_ray_init_fixture):
    """Image input should produce placeholder tokens and multimodal kwargs."""
    cfg = _get_config()
    async with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN3_VL,
        sleep_level=1,
        engine_init_kwargs={
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
        },
        use_new_inference_servers=True,
    ) as engines:
        renderer = VLLMRenderer(engines.client, model_name=MODEL_QWEN3_VL)

        jpeg_b64 = _make_tiny_jpeg_b64()
        mi = ModelInput(chunks=[ImageChunk(data=jpeg_b64, format="jpeg")])
        results = await renderer([mi])

        assert len(results) == 1
        rendered = results[0]
        assert len(rendered.prompt_ids) > 0
        assert rendered.multi_modal_placeholders is not None
        assert len(rendered.multi_modal_placeholders) == 1
        ph = rendered.multi_modal_placeholders[0]
        assert ph.offset >= 0
        assert ph.length > 0
        assert ph.offset + ph.length <= len(rendered.prompt_ids)


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_renderer_mixed_text_and_image(module_scoped_ray_init_fixture):
    """Mixed text + image input should assemble tokens in chunk order."""
    cfg = _get_config()
    async with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN3_VL,
        sleep_level=1,
        engine_init_kwargs={
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
        },
        use_new_inference_servers=True,
    ) as engines:
        renderer = VLLMRenderer(engines.client, model_name=MODEL_QWEN3_VL)

        prefix_tokens = _tokenize_text("Describe this image:")
        suffix_tokens = _tokenize_text("Be concise.")
        jpeg_b64 = _make_tiny_jpeg_b64()

        mi = ModelInput(
            chunks=[
                EncodedTextChunk(tokens=prefix_tokens),
                ImageChunk(data=jpeg_b64, format="jpeg"),
                EncodedTextChunk(tokens=suffix_tokens),
            ]
        )
        results = await renderer([mi])

        assert len(results) == 1
        rendered = results[0]

        assert rendered.prompt_ids[: len(prefix_tokens)] == prefix_tokens
        assert rendered.prompt_ids[-len(suffix_tokens) :] == suffix_tokens

        assert rendered.multi_modal_placeholders is not None
        assert len(rendered.multi_modal_placeholders) == 1
        ph = rendered.multi_modal_placeholders[0]
        assert ph.offset == len(prefix_tokens)
        total_len = len(prefix_tokens) + ph.length + len(suffix_tokens)
        assert len(rendered.prompt_ids) == total_len
