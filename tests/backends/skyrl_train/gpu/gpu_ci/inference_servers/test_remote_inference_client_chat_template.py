"""
Custom chat template tests for the the new inference path.

NOTE: This test is separate from `test_new_inference_generation.py` because we use separate engine configurations for each test parametrization.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_remote_inference_client_chat_template.py -m vllm -v
"""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

import skyrl
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

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
@pytest.mark.asyncio
@pytest.mark.parametrize("use_custom_template", [False, True])
async def test_custom_chat_template(ray_init_fixture, use_custom_template: bool):
    """Test custom chat template via RemoteInferenceClient.

    Uses render_chat_completion to get server-side tokenized prompt, then
    generates via /inference/v1/generate to avoid vllm-router's strict
    OpenAI API validation (which strips non-standard fields like prompt_token_ids).
    """
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3)
    async with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN3,
        sleep_level=1,
        engine_init_kwargs={"chat_template": TEMPLATE_PATH} if use_custom_template else None,
        use_new_inference_servers=True,
    ) as engines:
        client = engines.client

        # 1. Build chat messages with thinking tokens in assistant turn
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

        # 2. Render the chat template server-side to get prompt token IDs
        render_payload = {
            "model": MODEL_QWEN3,
            "messages": messages,
            "max_tokens": 10,
        }
        render_result = await client.render_chat_completion({"json": render_payload})
        prompt_token_ids = render_result["token_ids"]

        # 3. Generate using the rendered token IDs via /inference/v1/generate
        engine_input = InferenceEngineInput(
            prompt_token_ids=[prompt_token_ids],
            sampling_params={"max_tokens": 10},
        )
        output = await client.generate(engine_input)
        assert len(output["responses"]) == 1
        assert isinstance(output["responses"][0], str)

        # 4. Check thinking tokens stripped or not in the rendered prompt
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN3)
        prompt_str = tokenizer.decode(prompt_token_ids)

        if use_custom_template:
            assert "<think>" in prompt_str and "</think>" in prompt_str
        else:
            assert "<think>" not in prompt_str and "</think>" not in prompt_str
