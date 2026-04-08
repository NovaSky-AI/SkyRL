"""
Multimodal render tests for the new inference path.

Tests /v1/chat/completions/render with a VLM to verify multimodal
inputs are correctly tokenized and multimodal features are returned.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_vlm_inference_generation.py -m vllm -v
"""

import base64
import io

import pytest
from PIL import Image

from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

MODEL_QWEN3_VL = "Qwen/Qwen3-VL-2B-Instruct"
SERVED_MODEL_NAME = "my_qwen"
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
