"""
Generation and error handling tests for the new inference path.

Includes two groups of tests:

Group A: Generation and error handling tests that interact directly with the router's OpenAI-compatible
endpoints via `requests` or LiteLLM.

Group B: Generation and error handling tests that use the `RemoteInferenceClient`.

# Run with:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_new_inference_generation.py -m vllm -v
"""

# TODO (sumanthrh) (RemoteInferenceClient data-plane-deprecation): Remove the tests in Group B once we migrate all generation interactions to the router's HTTP API.

import json
import pytest
import asyncio
import aiohttp
import requests
from http import HTTPStatus
from typing import Any, Dict, List, Literal
from pydantic import BaseModel

from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType, InferenceEngineInput
from tests.backends.skyrl_train.gpu.utils import get_test_prompts, InferenceEngineState
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from transformers import AutoTokenizer

from litellm import acompletion as litellm_async_completion
from litellm import atext_completion as litellm_async_text_completion

MODEL_QWEN2_5 = "Qwen/Qwen2.5-0.5B-Instruct"
SERVED_MODEL_NAME = "my_qwen"
TP_SIZE = 1


def _get_test_sampling_params(backend: str, cfg: SkyRLTrainConfig, endpoint: str) -> Dict[str, Any]:
    assert endpoint in ["chat_completions", "completions"]
    sampling_params = get_sampling_params_for_backend(backend, cfg.generator.sampling_params)
    sampling_params["logprobs"] = True
    if endpoint == "chat_completions":
        sampling_params["top_logprobs"] = 1
    sampling_params["return_tokens_as_token_ids"] = True
    return sampling_params


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


def _get_base_url(engines: InferenceEngineState) -> str:
    """Get the router's base URL for OpenAI-compatible API (e.g. http://host:port/v1)."""
    proxy_url = engines.client.proxy_url
    return f"{proxy_url}/v1"


# Shared vllm server for all the tests in this file.
@pytest.fixture(scope="module")
def vllm_server(module_scoped_ray_init_fixture):
    """Single vLLM server + router. Tests hit the router's HTTP API directly."""
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    engines = InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL_QWEN2_5,
        sleep_level=1,
        engine_init_kwargs={"max_model_len": 1024},  # for test_context_length_error_returns_400
        use_new_inference_servers=True,
    )
    yield engines
    engines.close()


def _check_chat_completions_outputs(
    outputs: List[Dict], test_type: Literal["litellm", "request_posting"], num_samples: int, backend: str
):
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


def _check_completions_outputs(
    prompts: List, outputs: List[Dict], test_type: Literal["litellm", "request_posting"], backend: str
):
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


# --- Group A: Router HTTP API ---


@pytest.mark.vllm
def test_served_model_name(vllm_server: InferenceEngineState):
    """Test that served_model_name works and model path fails."""
    base_url = _get_base_url(vllm_server)
    messages = [{"role": "user", "content": "Hello, who are you?"}]

    # Request with served_model_name should succeed
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": messages,
            "max_tokens": 50,
        },
    )
    assert response.status_code == HTTPStatus.OK, f"Request failed: {response.status_code} {response.text}"
    result = response.json()
    assert "choices" in result and len(result["choices"]) > 0
    assert result["choices"][0]["message"]["content"] is not None

    # Request with model path should fail (model name mismatch)
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": MODEL_QWEN2_5,
            "messages": messages,
            "max_tokens": 50,
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.vllm
def test_chat_completions(vllm_server: InferenceEngineState):
    """Test chat completions via router HTTP API."""
    base_url = _get_base_url(vllm_server)
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = _get_test_sampling_params("vllm", cfg, "chat_completions")

    num_samples = 5
    test_prompts: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)

    outputs = []
    for conv in test_prompts:
        payload = {
            "model": SERVED_MODEL_NAME,
            "messages": conv,
            "max_tokens": 50,
            **sampling_params,
        }
        response = requests.post(f"{base_url}/chat/completions", json=payload)
        assert response.status_code == HTTPStatus.OK
        outputs.append(response.json())

    _check_chat_completions_outputs(outputs, "request_posting", num_samples, "vllm")


@pytest.mark.vllm
def test_completions(vllm_server: InferenceEngineState):
    """Test completions via router HTTP API."""
    base_url = _get_base_url(vllm_server)
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = _get_test_sampling_params("vllm", cfg, "completions")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

    num_samples = 5
    test_prompts_conv: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)
    text_prompts = [
        tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) for conv in test_prompts_conv
    ]

    outputs = []
    for prompt in text_prompts:
        payload = {
            "model": SERVED_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 50,
            **sampling_params,
        }
        response = requests.post(f"{base_url}/completions", json=payload)
        assert response.status_code == HTTPStatus.OK
        outputs.append(response.json())

    _check_completions_outputs(text_prompts, outputs, "request_posting", "vllm")


@pytest.mark.vllm
def test_structured_generation(vllm_server: InferenceEngineState):
    """Test structured generation (JSON schema) via router HTTP API."""
    base_url = _get_base_url(vllm_server)

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

    response = requests.post(f"{base_url}/chat/completions", json=payload)
    assert response.status_code == HTTPStatus.OK
    result = response.json()
    text = result["choices"][0]["message"]["content"]
    assert json.loads(text) is not None, f"Output is not valid JSON: {text}"


@pytest.mark.vllm
def test_error_handling(vllm_server: InferenceEngineState):
    """Test error handling via router HTTP API."""
    base_url = _get_base_url(vllm_server)

    # Missing required field (messages)
    response = requests.post(f"{base_url}/chat/completions", json={"model": SERVED_MODEL_NAME})
    assert response.status_code == HTTPStatus.BAD_REQUEST

    # Wrong model name
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": "wrong_model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        },
    )
    assert response.status_code == HTTPStatus.NOT_FOUND


@pytest.mark.vllm
def test_context_length_error_returns_400(vllm_server):
    """Test that context length errors return HTTP 400."""
    base_url = _get_base_url(vllm_server)

    # Oversized prompt (max_model_len=1024 in fixture)
    messages_oversized = [{"role": "user", "content": "hello " * 1500}]

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": SERVED_MODEL_NAME,
            "messages": messages_oversized,
            "max_tokens": 10,
        },
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    error_data = response.json()
    error_message = error_data.get("error", {}).get("message", str(error_data)).lower()
    assert "maximum context length" in error_message or "context" in error_message


# NOTE : We use LiteLLM because it supports sampling params such as min_tokens, skip_special_tokens, etc.,
# that are used in vllm/sglang, but are not supported by OpenAI.chat.completions.create().
@pytest.mark.vllm
def test_chat_completions_via_litellm(vllm_server: InferenceEngineState):
    """Test chat completions via LiteLLM (OpenAI API client style)."""
    base_url = _get_base_url(vllm_server)
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = _get_test_sampling_params("vllm", cfg, "chat_completions")

    num_samples = 5
    test_prompts: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)

    async def _run():
        outputs = []
        for conv in test_prompts:
            result = await litellm_async_completion(
                model=f"openai/{SERVED_MODEL_NAME}",
                messages=conv,
                api_base=base_url,
                api_key="DUMMY_KEY",
                **sampling_params,
            )
            outputs.append(result)
        return outputs

    outputs = asyncio.run(_run())
    _check_chat_completions_outputs(outputs, "litellm", num_samples, "vllm")


@pytest.mark.vllm
def test_completions_via_litellm(vllm_server: InferenceEngineState):
    """Test completions via LiteLLM (OpenAI API client style)."""
    base_url = _get_base_url(vllm_server)
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
            result = await litellm_async_text_completion(
                model=f"openai/{SERVED_MODEL_NAME}",
                prompt=[prompt],
                api_base=base_url,
                api_key="DUMMY_KEY",
                **sampling_params,
            )
            outputs.append(result)
        return outputs

    outputs = asyncio.run(_run())
    _check_completions_outputs(text_prompts, outputs, "litellm", "vllm")


# --- Group B: RemoteInferenceClient ---


@pytest.mark.vllm
def test_client_served_model_name(vllm_server: InferenceEngineState):
    """Test that served_model_name works and model path fails (via RemoteInferenceClient)."""
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
    with pytest.raises(aiohttp.ClientResponseError):
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
def test_client_error_handling(vllm_server: InferenceEngineState):
    """Test error handling via RemoteInferenceClient (raise_for_status path)."""
    client = vllm_server.client

    # Missing required field (messages)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(client.chat_completion({"json": {"model": SERVED_MODEL_NAME}}))

    # Wrong model name
    with pytest.raises(aiohttp.ClientResponseError):
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


@pytest.mark.vllm
def test_client_context_length_error_returns_400(vllm_server):
    """Test that context length errors return HTTP 400 (via RemoteInferenceClient)."""
    client = vllm_server.client

    # Oversized prompt (max_model_len=1024 in fixture)
    messages_oversized = [{"role": "user", "content": "hello " * 1500}]

    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
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
    assert err.status == HTTPStatus.BAD_REQUEST
    error_str = str(err).lower()
    assert "maximum context length" in error_str or "context" in error_str


@pytest.mark.vllm
def test_client_generate(vllm_server: InferenceEngineState):
    """Test token-in-token-out generation via RemoteInferenceClient.generate()."""
    client = vllm_server.client
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
    sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
    sampling_params["max_tokens"] = 50
    tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

    conv = get_test_prompts(MODEL_QWEN2_5, num_samples=1)[0]
    prompt_token_ids = tokenizer.apply_chat_template(
        [conv],
        add_generation_prompt=True,
        tokenize=True,
    )
    engine_input = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    output = asyncio.run(client.generate(engine_input))

    assert len(output["responses"]) == 1
    assert len(output["response_ids"]) == 1
    assert len(output["stop_reasons"]) == 1
    assert output["stop_reasons"][0] in ["stop", "length"]
    assert output["responses"][0] is not None
    assert len(output["response_ids"][0]) > 0


@pytest.mark.vllm
def test_client_tokenize_detokenize_roundtrip(vllm_server: InferenceEngineState):
    """Round-trip: tokenize text, detokenize back, verify."""
    # NOTE (sumanthrh): This test doesn't work for *any* tokenizer/ text because tokenization is not invertible in general.
    # but it is valid for this specific case.
    client = vllm_server.client
    text = "Hello, world!"

    token_ids = asyncio.run(client.tokenize([text]))[0]
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0

    decoded = asyncio.run(client.detokenize([token_ids]))[0]
    assert decoded == text
