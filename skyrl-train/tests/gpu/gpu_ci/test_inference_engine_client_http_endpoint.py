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
from typing import Any, Dict
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
from tests.gpu.gpu_ci.test_engine_generation import init_remote_inference_servers
from tests.gpu.utils import init_inference_engines, initialize_ray
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


def _get_test_sampling_params(backend: str, cfg: DictConfig) -> Dict[str, Any]:
    sampling_params = get_sampling_params_for_backend(backend, cfg.generator.sampling_params)
    sampling_params["logprobs"] = True
    sampling_params["top_logprobs"] = 1
    sampling_params["return_tokens_as_token_ids"] = True
    return sampling_params


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


def _check_outputs(outputs, test_type, num_samples, backend):
    print_n = 5
    assert len(outputs) == num_samples
    print(f"First {print_n} generated responses out of {num_samples} using {test_type}:")
    print(f"outputs[0]: {outputs[0]}")
    for i, output in enumerate(outputs[:print_n]):
        print(f"{i}: {output['choices'][0]['message']['content'][:100]}...")

    # Check response structure
    for response_data in outputs:
        if test_type == "litellm":
            # litellm returns a pydantic object
            response_data = response_data.model_dump()

        if test_type != "litellm":
            # Cannot check for litellm because it returns it has its own pydantic object
            if backend == "vllm":
                from vllm.entrypoints.openai.protocol import ChatCompletionResponse

                ChatCompletionResponse.model_validate(response_data)  # will raise error if invalid
            else:
                # TODO(Charlie): add sglang checkings once we support it for http endpoint
                raise ValueError(f"Unsupported backend: {backend}")

        for key in ["id", "object", "created", "model", "choices"]:
            assert key in response_data
            assert response_data[key] is not None

        for choice in response_data["choices"]:
            assert "index" in choice and "message" in choice and "finish_reason" in choice
            assert choice["index"] == 0 and choice["finish_reason"] in ["stop", "length"]
            message = choice["message"]
            assert "role" in message and "content" in message and message["role"] == "assistant"

        # check token_logprobs
        choice = response_data["choices"][0]
        assert "logprobs" in choice
        assert choice["logprobs"]["content"] is not None
        # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm
        assert choice["logprobs"]["content"][0]["token"].split(":")[1].isdigit()


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
        sampling_params = _get_test_sampling_params("vllm", cfg)
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
            def generate_output(traj_id, prompt):
                return requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": prompt,
                        "trajectory_id": traj_id,
                        **sampling_params,
                    },
                ).json()

            # Default concurrency is low. Increase concurrency with max_workers arg in ThreadPoolExecutor.
            with ThreadPoolExecutor() as executor:
                output_tasks = [
                    executor.submit(generate_output, traj_id, prompt) for traj_id, prompt in enumerate(test_prompts)
                ]
                outputs = [task.result() for task in output_tasks]

        elif test_type == "aiohttp_client_session":
            # 1.2 Test aiohttp.ClientSession
            async def generate_outputs_async():
                # limit=0 means no limit; without conn, it has a cap of 100 concurrent connections
                conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
                async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
                    headers = {"Content-Type": "application/json"}
                    output_tasks = []

                    for traj_id, prompt in enumerate(test_prompts):
                        payload = {
                            "model": MODEL,
                            "messages": prompt,
                            "trajectory_id": traj_id,
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
                async def generate_output(traj_id, prompt):
                    return await litellm_async_completion(
                        model=f"openai/{MODEL}",  # Add openai/ prefix for custom endpoints
                        messages=prompt,
                        api_base=base_url,
                        # Otherwise runs into: litellm.llms.openai.common_utils.OpenAIError
                        api_key="DUMMY_KEY",
                        trajectory_id=traj_id,
                        **sampling_params,
                    )

                tasks = [generate_output(traj_id, prompt) for traj_id, prompt in enumerate(test_prompts)]
                return await asyncio.gather(*tasks)

            outputs = asyncio.run(generate_outputs_async())

        else:
            raise ValueError(f"Invalid test type: {test_type}")

        _check_outputs(outputs, test_type, num_samples, "vllm")

        # Shutdown server
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

    finally:
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        ray.shutdown()


@pytest.mark.parametrize(
    "backend,tp_size",
    [
        pytest.param("vllm", 2, marks=pytest.mark.vllm),
        # TODO(Charlie): add TP > 1 tests for sglang when we support it
        # TODO(Charlie): sglang remote server not supported for /chat/completion
        # yet because we have skip_tokenizer_init=True. Fix by getting tokens
        # via return logprobs instead.
        # pytest.param("sglang", 1, marks=pytest.mark.sglang),
    ],
    # ids=["vllm", "sglang"],
    ids=["vllm"],
)
def test_http_endpoint_with_remote_servers(backend, tp_size):
    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    server_port = None

    try:
        # 1. Initialize InferenceEngineClient client with remote servers
        cfg = get_test_actor_config()
        cfg.generator.backend = backend
        initialize_ray(cfg)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        client, remote_server_process = init_remote_inference_servers(tp_size, backend, tokenizer, cfg, MODEL)
        sampling_params = _get_test_sampling_params(backend, cfg)

        # 2. Start HTTP endpoint in background thread using serve function directly
        server_port = get_free_port()

        def run_server():
            serve(client, host=SERVER_HOST, port=server_port, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to be ready using the helper method
        wait_for_server_ready(host=SERVER_HOST, port=server_port, max_wait_seconds=30)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # 3. Generate outputs using litellm and check outputs
        num_samples = 20
        test_prompts = get_test_prompts(MODEL, num_samples=num_samples)

        # Default concurrency limit is 100 due to HTTP client pool capacity.
        async def generate_outputs_async():
            async def generate_output(traj_id, prompt):
                return await litellm_async_completion(
                    model=f"openai/{MODEL}",  # Add openai/ prefix for custom endpoints
                    messages=prompt,
                    api_base=base_url,
                    # Otherwise runs into: litellm.llms.openai.common_utils.OpenAIError
                    api_key="DUMMY_KEY",
                    trajectory_id=traj_id,
                    **sampling_params,
                )

            tasks = [generate_output(traj_id, prompt) for traj_id, prompt in enumerate(test_prompts)]
            return await asyncio.gather(*tasks)

        outputs = asyncio.run(generate_outputs_async())
        _check_outputs(outputs, "litellm", num_samples, backend)

        # 4. Shutdown server
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if "remote_server_process" in locals():
            remote_server_process.terminate()
            remote_server_process.wait()
        ray.shutdown()


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

        # Test 1: Invalid request - streaming not supported, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "Streaming is not supported" in error_data["error"]["message"]

        # Test 2: OAI can take fields not listed in the protocol
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "xxx": "yyy"},
        )
        assert response.status_code == HTTPStatus.OK  # 200

        # Test 3: Invalid request - missing required fields, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL,
                # Missing messages field
            },
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "messages" in error_data["error"]["message"]

        # Test 4: Invalid request - malformed JSON, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions", data="some invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "Invalid JSON error" in error_data["error"]["message"]  # JSON decode error

        # Test 5: Invalid request - empty messages array, raised by vLLM
        response = requests.post(f"{base_url}/v1/chat/completions", json={"model": MODEL, "messages": []})
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "list index out of range list index out of range" in error_data["error"]["message"]

        # Test 6: Wrong model name, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "wrong_model", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "Model name mismatch" in error_data["error"]["message"]

        # Test 7: Health check endpoint should work
        response = requests.get(f"{base_url}/health")
        assert response.status_code == HTTPStatus.OK  # 200
        health_data = response.json()
        assert health_data["status"] == "healthy"

    finally:
        ray.shutdown()
