"""
Generation and error handling tests for `RemoteInferenceClient` (new inference path) 

Requires _SKYRL_USE_NEW_INFERENCE=1.

# Run with:
_SKYRL_USE_NEW_INFERENCE=1 uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_remote_inference_client_generation.py -m vllm -v
"""

import json
import pytest
import asyncio
from http import HTTPStatus
from typing import Any, Dict, List
from pydantic import BaseModel


from skyrl.train.config import SkyRLConfig
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from tests.backends.skyrl_train.gpu.utils import get_test_prompts, InferenceEngineState
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.backends.skyrl_train.env_vars import _SKYRL_USE_NEW_INFERENCE
from transformers import AutoTokenizer

MODEL_QWEN2_5 = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_NAME = "my_qwen"
TP_SIZE = 1

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.skipif(
        not _SKYRL_USE_NEW_INFERENCE,
        reason="Requires _SKYRL_USE_NEW_INFERENCE=1",
    ),
]


def _get_test_sampling_params(backend: str, cfg: SkyRLConfig, endpoint: str) -> Dict[str, Any]:
    assert endpoint in ["chat_completions", "completions"]
    sampling_params = get_sampling_params_for_backend(backend, cfg.generator.sampling_params)
    sampling_params["logprobs"] = True
    if endpoint == "chat_completions":
        sampling_params["top_logprobs"] = 1
    sampling_params["return_tokens_as_token_ids"] = True
    return sampling_params


def get_test_actor_config(num_inference_engines: int, model: str) -> SkyRLConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = model
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE * num_inference_engines
    cfg.generator.async_engine = True
    cfg.generator.num_inference_engines = num_inference_engines
    cfg.generator.inference_engine_tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.served_model_name = SERVED_MODEL_NAME
    cfg.generator.sampling_params.max_generate_length = 256
    return cfg


@pytest.fixture(scope="module")
def vllm_server(module_scoped_ray_init_fixture):
    """Single vLLM server + router + RemoteInferenceClient."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    cfg.trainer.placement.colocate_all = True
    engines = InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN2_5,
        sleep_level=1,
        engine_init_kwargs={"max_model_len": 1024},  # for test_context_length_error_returns_400
    )
    yield engines
    engines.close()


def _check_chat_completions_outputs(outputs: List[Dict], test_type: str, num_samples: int, backend: str):
    for output in outputs:
        assert not ("error" in output or output.get("object", "") == "error"), f"Error in output: {output}"
    assert len(outputs) == num_samples
    for response_data in outputs:
        if test_type == "litellm":
            response_data = response_data.model_dump()
        if test_type != "litellm" and backend == "vllm":
            from vllm.entrypoints.openai.protocol import ChatCompletionResponse

            ChatCompletionResponse.model_validate(response_data)
        for key in ["id", "object", "created", "model", "choices"]:
            assert key in response_data
            assert response_data[key] is not None
        for i, choice in enumerate(response_data["choices"]):
            assert "index" in choice and "message" in choice and "finish_reason" in choice
            assert choice["index"] == i and choice["finish_reason"] in ["stop", "length"]
            message = choice["message"]
            assert "role" in message and "content" in message and message["role"] == "assistant"
            choice_data = response_data["choices"][i]
            assert "logprobs" in choice_data
            assert choice_data["logprobs"]["content"] is not None


def _check_completions_outputs(prompts: List, outputs: List[Dict], test_type: str, backend: str):
    for output in outputs:
        assert not ("error" in output or output.get("object", "") == "error"), f"Error in output: {output}"
    num_outputs = sum(len(output["choices"]) for output in outputs)
    assert num_outputs == len(prompts)
    for response_data in outputs:
        if test_type == "litellm":
            response_data = response_data.model_dump()
        if test_type != "litellm" and backend == "vllm":
            from vllm.entrypoints.openai.protocol import CompletionResponse

            CompletionResponse.model_validate(response_data)
        for key in ["id", "object", "created", "model", "choices"]:
            assert key in response_data
            assert response_data[key] is not None
        for i, choice in enumerate(response_data["choices"]):
            assert "index" in choice and "text" in choice and "finish_reason" in choice
            assert choice["index"] == i and choice["finish_reason"] in ["stop", "length"]
            assert "logprobs" in choice and choice["logprobs"] is not None
            assert "tokens" in choice["logprobs"]


# --- Group A: Completions ---


@pytest.mark.vllm
def test_served_model_name(vllm_server):
    """Test that served_model_name works and model path fails."""
    client = vllm_server.client
    messages = [{"role": "user", "content": "Hello, who are you?"}]

    # Request with served_model_name should succeed
    result = asyncio.run(
        client.chat_completion(
            {
                "json": {
                    "model": SERVED_MODEL_NAME,
                    "messages": messages,
                    "max_tokens": 50,
                }
            }
        )
    )
    assert "choices" in result and len(result["choices"]) > 0
    assert result["choices"][0]["message"]["content"] is not None

    # Request with model path should fail (model name mismatch)
    with pytest.raises(Exception):
        asyncio.run(
            client.chat_completion(
                {
                    "json": {
                        "model": MODEL_QWEN2_5,
                        "messages": messages,
                        "max_tokens": 50,
                    }
                }
            )
        )


@pytest.mark.vllm
def test_chat_completions(vllm_server):
    """Test chat completions via RemoteInferenceClient."""
    client = vllm_server.client
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = _get_test_sampling_params("vllm", cfg, "chat_completions")

    num_samples = 5
    test_prompts: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)

    async def _run():
        outputs = []
        for conv in test_prompts:
            payload = {
                "model": SERVED_MODEL_NAME,
                "messages": conv,
                "max_tokens": 50,
                **sampling_params,
            }
            result = await client.chat_completion({"json": payload})
            outputs.append(result)
        return outputs

    outputs = asyncio.run(_run())
    _check_chat_completions_outputs(outputs, "request_posting", num_samples, "vllm")


@pytest.mark.vllm
def test_completions(vllm_server):
    """Test completions via RemoteInferenceClient."""
    client = vllm_server.client
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = _get_test_sampling_params("vllm", cfg, "completions")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

    num_samples = 5
    test_prompts_conv: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)
    text_prompts = [
        tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) for conv in test_prompts_conv
    ]

    async def _run():
        outputs = []
        for prompt in text_prompts:
            payload = {
                "model": SERVED_MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 50,
                **sampling_params,
            }
            result = await client.completion({"json": payload})
            outputs.append(result)
        return outputs

    outputs = asyncio.run(_run())
    _check_completions_outputs(text_prompts, outputs, "request_posting", "vllm")


@pytest.mark.vllm
def test_structured_generation(vllm_server):
    """Test structured generation (JSON schema) via RemoteInferenceClient."""
    client = vllm_server.client

    class TestSchema(BaseModel):
        name: str
        job: str

    prompt = [
        {
            "role": "user",
            "content": f"Introduce yourself in JSON format briefly, following the schema {TestSchema.model_json_schema()}.",
        },
    ]

    payload = {
        "model": SERVED_MODEL_NAME,
        "messages": prompt,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "TestSchema",
                "schema": TestSchema.model_json_schema(),
                "strict": True,
            },
        },
    }

    async def _run():
        return await client.chat_completion({"json": payload})

    result = asyncio.run(_run())
    text = result["choices"][0]["message"]["content"]
    assert json.loads(text) is not None, f"Output is not valid JSON: {text}"


# --- Group B: Error handling ---


@pytest.mark.vllm
def test_error_handling(vllm_server):
    """Test error handling via RemoteInferenceClient."""
    client = vllm_server.client

    # Missing required field (messages)
    with pytest.raises(Exception) as exc_info:
        asyncio.run(client.chat_completion({"json": {"model": SERVED_MODEL_NAME}}))
    assert exc_info.value is not None

    # Wrong model name
    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            client.chat_completion(
                {
                    "json": {
                        "model": "wrong_model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10,
                    }
                }
            )
        )
    assert exc_info.value is not None


@pytest.mark.vllm
def test_context_length_error_returns_400(vllm_server):
    """Test that context length errors return HTTP 400."""
    client = vllm_server.client

    # Oversized prompt (max_model_len=1024 in fixture)
    messages_oversized = [{"role": "user", "content": "hello " * 1500}]

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            client.chat_completion(
                {
                    "json": {
                        "model": SERVED_MODEL_NAME,
                        "messages": messages_oversized,
                        "max_tokens": 10,
                    }
                }
            )
        )
    err = exc_info.value
    if hasattr(err, "status"):
        assert err.status == HTTPStatus.BAD_REQUEST
    print("Error message is", str(err))
    assert "maximum context length" in str(err).lower() or "context" in str(err).lower()
