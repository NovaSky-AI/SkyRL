"""
VLM integration tests for the new inference path and tinker renderer.


Tests:
    - vllm /v1/chat/completions/render with a VLM to verify multimodal inputs
    - sample() with multimodal Tinker prompts end-to-end
    - VLLMRenderer end-to-end

Requires a local vLLM install with /v1/chat/completions/render support.

# Run with:
SKYRL_LOCAL_VLLM=1 uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vlm_inference_generation.py -m vllm -v
"""

import base64
import io
import os

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

from skyrl.backends.renderer import VLLMRenderer
from skyrl.tinker.types import EncodedTextChunk, ImageChunk, ModelInput
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

requires_local_vllm = pytest.mark.skipif(
    os.environ.get("SKYRL_LOCAL_VLLM") != "1",
    reason="Requires local vLLM with multi-modal /v1/chat/completions/render support",
)

MODEL_QWEN3_VL = "Qwen/Qwen3-VL-2B-Instruct"
SERVED_MODEL_NAME = "my_qwen"
QWEN3_VL_IMAGE_PLACEHOLDER_TOKEN_ID = 151655
TP_SIZE = 1


def get_test_actor_config(num_inference_engines: int, model: str) -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE * num_inference_engines
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = num_inference_engines
    cfg.generator.inference_engine.tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = SERVED_MODEL_NAME
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


def _make_tiny_base64_image() -> str:
    """Create a minimal 8x8 JPEG image and return it as a data URI."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _make_tiny_jpeg_b64() -> bytes:
    """Create a tiny JPEG and return raw base64 bytes for ImageChunk usage."""
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


def _tokenize_text(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3_VL, trust_remote_code=True)
    return tokenizer.encode(text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Raw render endpoint test
# ---------------------------------------------------------------------------


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_render_chat_completion_multimodal(module_scoped_ray_init_fixture):
    """Test /v1/chat/completions/render with a multimodal (image) input on a VLM."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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
        client = engines.client

        data_uri = _make_tiny_base64_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "What is in this image?"},
                ],
            }
        ]

        result = await client.render_chat_completion({"json": {"model": MODEL_QWEN3_VL, "messages": messages}})

        assert isinstance(result, dict)

        assert "request_id" in result
        assert result["request_id"].startswith("chatcmpl-")

        assert "token_ids" in result
        token_ids = result["token_ids"]
        assert isinstance(token_ids, list)
        assert len(token_ids) > 0
        assert all(isinstance(t, int) for t in token_ids)

        assert "sampling_params" in result
        assert isinstance(result["sampling_params"], dict)

        assert "model" in result
        assert result["model"] == MODEL_QWEN3_VL

        features = result.get("features")
        assert features is not None, f"Expected multimodal features, got None. Keys: {list(result.keys())}"

        assert "mm_hashes" in features
        assert "image" in features["mm_hashes"]
        image_hashes = features["mm_hashes"]["image"]
        assert isinstance(image_hashes, list)
        assert len(image_hashes) == 1
        assert isinstance(image_hashes[0], str)
        assert len(image_hashes[0]) > 0

        assert "mm_placeholders" in features
        assert "image" in features["mm_placeholders"]
        image_placeholders = features["mm_placeholders"]["image"]
        assert isinstance(image_placeholders, list)
        assert len(image_placeholders) == 1

        placeholder = image_placeholders[0]
        assert "offset" in placeholder
        assert "length" in placeholder
        assert isinstance(placeholder["offset"], int)
        assert isinstance(placeholder["length"], int)
        assert placeholder["length"] > 0
        assert placeholder["offset"] + placeholder["length"] <= len(token_ids)


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_sample_with_multimodal_image(module_scoped_ray_init_fixture):
    """Test sample() with a Tinker prompt containing [text, image, text] chunks on a real VLM."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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
        client = engines.client
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3_VL)

        # Tokenize text portions using the model's tokenizer.
        prefix_text = "<|im_start|>user\n"
        suffix_text = "What color is this image? Answer in one word.<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Build a tiny JPEG as raw base64 (matching model_dump() output for Base64Bytes).
        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Construct Tinker-style prompt: [text chunk, image chunk, text chunk]
        prompt = {
            "chunks": [
                {"type": "encoded_text", "tokens": prefix_ids},
                {"type": "image", "data": image_b64, "format": "jpeg"},
                {"type": "encoded_text", "tokens": suffix_ids},
            ]
        }

        request_payload = {
            "json": {
                "prompt": prompt,
                "num_samples": 1,
                "sampling_params": {"temperature": 0.0, "max_tokens": 20},
            }
        }

        result = await client.sample(request_payload)

        assert result["type"] == "sample"
        sequences = result["sequences"]
        assert len(sequences) == 1

        seq = sequences[0]
        assert isinstance(seq["tokens"], list)
        assert len(seq["tokens"]) > 0
        assert seq["stop_reason"] in ("stop", "length")

        # Decode and check the model produced something reasonable.
        output_text = tokenizer.decode(seq["tokens"], skip_special_tokens=True)
        assert len(output_text.strip()) > 0


# ---------------------------------------------------------------------------
# VLLMRenderer tests
# ---------------------------------------------------------------------------


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_renderer_text_only(module_scoped_ray_init_fixture):
    """Text-only inputs should not trigger any HTTP calls to the render endpoint."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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
async def test_renderer_mixed_text_and_image(module_scoped_ray_init_fixture):
    """Mixed text + image input should assemble tokens in chunk order."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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

        placeholder_tokens = rendered.prompt_ids[ph.offset : ph.offset + ph.length]
        assert all(t == QWEN3_VL_IMAGE_PLACEHOLDER_TOKEN_ID for t in placeholder_tokens)


# ---------------------------------------------------------------------------
# multi_modal_kwargs tests
# ---------------------------------------------------------------------------


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_renderer_decodes_mm_kwargs(module_scoped_ray_init_fixture):
    """VLLMRenderer should produce decoded pixel_values and image_grid_thw in multi_modal_kwargs."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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
        mm = rendered.multi_modal_kwargs
        assert mm is not None

        pixel_values = mm["pixel_values"]
        image_grid_thw = mm["image_grid_thw"]

        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.numel() > 0
        assert pixel_values.ndim == 2

        assert isinstance(image_grid_thw, torch.Tensor)
        assert image_grid_thw.shape[1] == 3
        assert image_grid_thw.dtype == torch.long


# ---------------------------------------------------------------------------
# Render + Generate round-trip test
# ---------------------------------------------------------------------------


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_generate_with_multimodal_features_red_square(module_scoped_ray_init_fixture):
    """Render a red square image, then generate with mm_features and verify the model sees red."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3_VL)
    cfg.generator.inference_engine.served_model_name = MODEL_QWEN3_VL
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
        client = engines.client

        # Step 1: Render the image prompt to get token_ids and multi-modal features
        data_uri = _make_tiny_base64_image()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": "What color is this square? Answer with just the color name."},
                ],
            }
        ]

        render_result = await client.render_chat_completion({"json": {"model": MODEL_QWEN3_VL, "messages": messages}})

        token_ids = render_result["token_ids"]
        features = render_result.get("features")
        assert features is not None, "Render should return multi-modal features for image input"

        # Step 2: Generate using the rendered token_ids + mm_features
        input_batch = {
            "prompt_token_ids": [token_ids],
            "sampling_params": {"max_tokens": 64, "temperature": 0.0},
            "mm_features": [features],
        }
        gen_result = await client.generate(input_batch)

        # Structural assertions
        assert len(gen_result["responses"]) == 1
        assert len(gen_result["response_ids"]) == 1
        assert len(gen_result["response_ids"][0]) > 0
        assert gen_result["stop_reasons"][0] in ("stop", "length")

        # Semantic assertion: the model should identify the red square
        response_text = gen_result["responses"][0].lower()
        assert "red" in response_text, f"Expected model to identify the red square, got: {gen_result['responses'][0]}"
