"""
Custom chat template tests for the the new inference path.

NOTE: This test is separate from `test_new_inference_generation.py` because we use separate engine configurations for each test parametrization.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_remote_inference_client_chat_template.py -m vllm -v
"""

import pytest
import asyncio
from pathlib import Path

import skyrl
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState
from transformers import AutoTokenizer

MODEL_QWEN3 = "Qwen/Qwen3-0.6B"
TP_SIZE = 1

TEMPLATE_PATH = str(Path(skyrl.train.utils.__file__).parent / "templates/qwen3_acc_thinking.jinja2")


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
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


@pytest.mark.vllm
@pytest.mark.parametrize("use_custom_template", [False, True])
def test_custom_chat_template(ray_init_fixture, use_custom_template: bool):
    """Test custom chat template via RemoteInferenceClient."""
    engines = None
    try:
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3)
        engines = InferenceEngineState.create(
            cfg=cfg,
            use_local=True,
            backend="vllm",
            model=MODEL_QWEN3,
            sleep_level=1,
            engine_init_kwargs={"chat_template": TEMPLATE_PATH} if use_custom_template else None,
            use_new_inference_servers=True,
        )
        client = engines.client

        # 2. Send request
        # Test that the custom template will not strip thinking tokens, unlike the default template.
        messages = [
            {
                "role": "user",
                "content": "Hello",
            },
            {
                "role": "assistant",
                "content": "<think>Thinking...</think>Hello",
            },
            {
                "role": "user",
                "content": "Hello",
            },
        ]
        payload = {
            "model": MODEL_QWEN3,
            "messages": messages,
            "max_tokens": 10,
            "return_token_ids": True,
        }

        async def _run():
            return await client.chat_completion({"json": payload})

        data = asyncio.run(_run())

        # 3. Check output
        assert "choices" in data and len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert isinstance(content, str)

        # 4. Check thinking tokens stripped or not
        assert "prompt_token_ids" in data, f"prompt_token_ids not found in response. Keys: {data.keys()}"
        prompt_token_ids = data["prompt_token_ids"]
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3)
        prompt_str = tokenizer.decode(prompt_token_ids)

        if use_custom_template:
            # The custom template qwen3_acc_thinking.jinja2 will keep the thinking tokens.
            assert "<think>" in prompt_str and "</think>" in prompt_str
        else:
            # Default template strips thinking tokens
            assert "<think>" not in prompt_str and "</think>" not in prompt_str

    finally:
        if engines is not None:
            engines.close()
