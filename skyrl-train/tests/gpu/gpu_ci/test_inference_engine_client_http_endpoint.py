"""
Test the HTTP endpoint with LiteLLM and policy weight sync.

This uses the same workflow as test_policy_local_engines_e2e.py, but with the HTTP endpoint instead of
the inference client engine. Only requires 1 GPU.

# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_inference_engine_client_http_endpoint.py -m "vllm"
# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/gpu/gpu_ci/test_inference_engine_client_http_endpoint.py -m "sglang"
"""

import json
import pytest
import asyncio
from http import HTTPStatus
import ray
import hydra
import threading
import requests
import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from litellm import completion as litellm_completion
from litellm import acompletion as litellm_async_completion

from tests.gpu.utils import init_worker_with_type, get_test_prompts
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
from tests.gpu.utils import init_inference_engines
from concurrent.futures import ThreadPoolExecutor
from skyrl_train.inference_engines.openai_api_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    build_sampling_params,
)


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.inference_engine_tensor_parallel_size = TP_SIZE
        cfg.generator.run_engines_locally = True

        return cfg


# NOTE(Charlie): we do not test OpenAI client because it throws error when unsupported sampling params
# are passed into OpenAI.chat.completions.create() (e.g. min_tokens, skip_special_tokens, etc.),
# while these sampling params are used in vllm/sglang. Therefore, we instead use LiteLLM.
@pytest.mark.vllm
@pytest.mark.parametrize("test_type", ["request_posting", "aiohttp_client_session", "litellm"])
def test_http_endpoint_openai_api_with_weight_sync(test_type):
    """
    Test the HTTP endpoint with LiteLLM and policy weight sync.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = True  # Use colocate for simplicity
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        sampling_params = get_sampling_params_for_backend("vllm", cfg.generator.sampling_params)
        client, pg = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL,
        )

        # Start server in background thread using serve function directly
        def run_server():
            serve(client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready using the helper method
        wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

        # Weight sync as before
        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
            cfg=cfg,
        )
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        num_samples = 20
        test_prompts = get_test_prompts(MODEL, num_samples=num_samples)

        # Generate outputs based on test type
        if test_type == "request_posting":
            # 1.1 Test request posting
            def generate_output(prompt):
                return requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": prompt,
                        **sampling_params,
                    },
                ).json()

            # Default concurrency is low. Increase concurrency with max_workers arg in ThreadPoolExecutor.
            with ThreadPoolExecutor() as executor:
                output_tasks = [executor.submit(generate_output, prompt) for prompt in test_prompts]
                outputs = [task.result() for task in output_tasks]

        elif test_type == "aiohttp_client_session":
            # 1.2 Test aiohttp.ClientSession
            async def generate_outputs_async():
                # limit=0 means no limit; without conn, it has a cap of 100 concurrent connections
                conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
                async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
                    headers = {"Content-Type": "application/json"}
                    output_tasks = []

                    for prompt in test_prompts:
                        payload = {
                            "model": MODEL,
                            "messages": prompt,
                            **sampling_params,
                        }
                        output_tasks.append(session.post(f"{base_url}/chat/completions", json=payload, headers=headers))

                    responses = await asyncio.gather(*output_tasks)
                    return [await response.json() for response in responses]

            outputs = asyncio.run(generate_outputs_async())

        elif test_type == "litellm":
            # 1.3 Test litellm
            # Default concurrency limit is 100 due to HTTP client pool capacity.
            async def generate_outputs_async():
                async def generate_output(prompt):
                    return await litellm_async_completion(
                        model=f"openai/{MODEL}",  # Add openai/ prefix for custom endpoints
                        messages=prompt,
                        api_base=base_url,
                        # Otherwise runs into: litellm.llms.openai.common_utils.OpenAIError
                        api_key="DUMMY_KEY",
                        **sampling_params,
                    )

                tasks = [generate_output(prompt) for prompt in test_prompts]
                return await asyncio.gather(*tasks)

            outputs = asyncio.run(generate_outputs_async())

        else:
            raise ValueError(f"Invalid test type: {test_type}")

        print_n = 5
        assert len(outputs) == num_samples
        print(f"First {print_n} generated responses out of {num_samples} using {test_type}:")
        for i, output in enumerate(outputs[:print_n]):
            print(f"{i}: {output['choices'][0]['message']['content'][:100]}...")

        # 2. Check response structure
        for response_data in outputs:
            for key in ["id", "object", "created", "model", "choices"]:
                assert key in response_data
                assert response_data[key] is not None

            for choice in response_data["choices"]:
                assert "index" in choice and "message" in choice and "finish_reason" in choice
                assert choice["index"] == 0 and choice["finish_reason"] in ["stop", "length"]
                message = choice["message"]
                if test_type == "litellm":
                    message = message.model_dump()  # litellm returns a pydantic object
                assert "role" in message and "content" in message and message["role"] == "assistant"

        # Shutdown server
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

    finally:
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        ray.shutdown()


def _full_request():
    return ChatCompletionRequest(
        model=MODEL,
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=10,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=42,
        stop=["\n"],
        stop_token_ids=[2, 3],
        presence_penalty=0.0,
        frequency_penalty=0.0,
        ignore_eos=True,
        skip_special_tokens=True,
        include_stop_str_in_output=True,
        min_tokens=1,
        n=1,
        trajectory_id="test_trajectory_id",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "description": "test_description",
                "schema": {"type": "object"},
            },
        },
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("vllm", marks=pytest.mark.vllm),
        pytest.param("sglang", marks=pytest.mark.sglang),
    ],
)
def test_full_build_sampling_params(backend: str):
    full_req = _full_request()
    if backend == "vllm":
        from vllm import SamplingParams as VLLMSamplingParams

        full_params_vllm = build_sampling_params(full_req, "vllm")
        vllm_sampling_params = VLLMSamplingParams(**full_params_vllm)  # has __post_init__ to check validity
        assert vllm_sampling_params is not None
        assert vllm_sampling_params.guided_decoding.json_object is None
        assert vllm_sampling_params.guided_decoding.json == {"type": "object"}
    elif backend == "sglang":
        from sglang.srt.sampling.sampling_params import SamplingParams as SGLangSamplingParams

        # makes sure that the inclusion of `include_stop_str_in_output` will raise an error
        with pytest.raises(ValueError):
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.include_stop_str_in_output = None

        # makes sure that the inclusion of `seed` will raise an error
        with pytest.raises(ValueError):
            # makes sure that the inclusion of `seed` will raise an error
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.seed = None

        # makes sure that the inclusion of `min_tokens` will raise an error
        with pytest.raises(ValueError):
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.min_tokens = None

        # Now no errors should be raised
        full_params_sglang = build_sampling_params(full_req, "sglang")
        sglang_sampling_params = SGLangSamplingParams(**full_params_sglang)
        sglang_sampling_params.verify()  # checks validty
        assert sglang_sampling_params is not None
        assert sglang_sampling_params.json_schema == '{"type": "object"}'
    else:
        raise ValueError(f"Unsupported backend: {backend}")


@pytest.mark.vllm
def test_structured_generation():
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = True  # Use colocate for simplicity
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, _ = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL,
        )

        # Start server in background thread using serve function directly
        def run_server():
            serve(client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready using the helper method
        wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

        class TestSchema(BaseModel):
            name: str
            job: str

        prompt = [
            {
                "role": "user",
                "content": f"Introduce yourself in JSON format briefly, following the schema {TestSchema.model_json_schema()}.",
            },
        ]

        output = litellm_completion(
            model=f"openai/{MODEL}",
            api_base=base_url,
            api_key="DUMMY_KEY",
            messages=prompt,
            max_tokens=1024,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "TestSchema",
                    "schema": TestSchema.model_json_schema(),
                    "strict": True,
                },
            },
        )

        # assert is valid json
        text = output.choices[0].message.content
        print(f"Output: {text}")
        assert json.loads(text) is not None  # if json invalid
    finally:
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        ray.shutdown()


@pytest.mark.vllm
def test_http_endpoint_error_handling():
    """
    Test error handling for various invalid requests.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, _ = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL,
        )

        from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
            serve,
            wait_for_server_ready,
        )

        # Start server in background thread
        def run_server():
            serve(client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready
        wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)

        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"

        # Test 1: Invalid request - streaming not supported
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # 422
        error_data = response.json()
        assert "detail" in error_data
        # Pydantic returns detailed field validation errors
        print(f"Error data: {error_data}")
        assert any("stream" in str(detail) for detail in error_data["detail"])

        # Test 2: Invalid request - tools not supported
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"type": "function", "function": {"name": "test"}}],
            },
        )
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR  # 500
        error_data = response.json()
        assert "message" in error_data
        assert "Unsupported fields: tools" in str(error_data["message"])

        # Test 3: OAI can take fields not listed in the protocol.
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "xxx": "yyy"},
        )
        assert response.status_code == HTTPStatus.OK  # 200

        # Test 4: Invalid request - missing required fields
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                # Missing messages field
            },
        )
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # 422
        error_data = response.json()
        assert "detail" in error_data
        assert any("messages" in str(detail) for detail in error_data["detail"])

        # Test 5: Invalid request - malformed JSON
        response = requests.post(
            f"{base_url}/v1/chat/completions", data="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY  # 422

        # Test 6: Invalid request - empty messages array
        response = requests.post(f"{base_url}/v1/chat/completions", json={"model": MODEL, "messages": []})
        assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR  # 500

        # Test 7: Health check endpoint should work
        response = requests.get(f"{base_url}/health")
        assert response.status_code == HTTPStatus.OK  # 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

    finally:
        ray.shutdown()
