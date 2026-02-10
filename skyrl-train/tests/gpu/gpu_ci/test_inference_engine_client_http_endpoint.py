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
from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from unittest.mock import patch
import ray
import threading
import requests
import aiohttp
from pydantic import BaseModel
import litellm
from litellm import completion as litellm_completion
from litellm import acompletion as litellm_async_completion
from litellm import atext_completion as litellm_async_text_completion
import logging

from skyrl_train.config import SkyRLConfig
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from tests.gpu.utils import init_worker_with_type, get_test_prompts
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
import skyrl_train.inference_engines.inference_engine_client_http_endpoint as http_endpoint_module
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
from tests.gpu.gpu_ci.test_engine_generation import init_remote_inference_servers
from tests.gpu.utils import init_inference_engines
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

MODEL_QWEN2_5 = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_QWEN3 = "Qwen/Qwen3-0.6B"
TP_SIZE = 1
SERVER_HOST = "127.0.0.1"


# Disable aiohttp transport in litellm to avoid unclosed connector warnings.
# This makes litellm use httpx's default transport instead of aiohttp.
# This is safe for tests since we don't need the performance benefits of aiohttp.
litellm.disable_aiohttp_transport = True


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
    return cfg


# ------------------------------------------
# Helper functions for setting up HTTP server
# ------------------------------------------


def set_up_http_server(client: InferenceEngineClient) -> Tuple[threading.Thread, int]:
    def _find_available_port(host: str) -> int:
        """Find an available port by binding to port 0."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    # Find an available port
    server_port = _find_available_port(SERVER_HOST)

    # Start server in background thread
    def run_server():
        serve(client, host=SERVER_HOST, port=server_port, log_level="warning")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    wait_for_server_ready(host=SERVER_HOST, port=server_port, max_wait_seconds=30)

    return server_thread, server_port


# --------------------------------------
# Helper functions for checking outputs
# --------------------------------------


def _check_chat_completions_outputs(outputs, test_type, num_samples, backend):
    # check error
    for output in outputs:
        assert not ("error" in output or output.get("object", "") == "error"), f"Error in output: {output}"
    print_n = 5
    assert len(outputs) == num_samples
    print(f"First {print_n} generated responses out of {num_samples} using {test_type}:")
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

        for i, choice in enumerate(response_data["choices"]):
            assert "index" in choice and "message" in choice and "finish_reason" in choice
            assert choice["index"] == i and choice["finish_reason"] in ["stop", "length"]
            message = choice["message"]
            assert "role" in message and "content" in message and message["role"] == "assistant"

            # check token_logprobs
            choice = response_data["choices"][i]
            assert "logprobs" in choice
            assert choice["logprobs"]["content"] is not None
            # tokens are token_id:<int> because we request `return_tokens_as_token_ids` from vllm
            assert choice["logprobs"]["content"][0]["token"].split(":")[1].isdigit()


def _check_completions_outputs(prompts, outputs, test_type, backend):
    # check error
    for output in outputs:
        assert not ("error" in output or output.get("object", "") == "error"), f"Error in output: {output}"

    num_outputs = sum(len(output["choices"]) for output in outputs)
    assert num_outputs == len(prompts)

    print_n = 5
    # Content checks
    print(f"First {print_n} generated responses out of {num_outputs} using completions:")
    # First flatten it output[i][choices] into a single list
    choice_list = [output["choices"] for output in outputs]
    choice_list = [item for sublist in choice_list for item in sublist]
    for i, output in enumerate(choice_list[:print_n]):
        data = output.model_dump() if test_type == "litellm" else output
        # CompletionResponse uses 'text' field
        preview = data.get("text") or str(data)[0:100]
        print(f"Prompt {i}: {prompts[i][:300]}...")
        print(f"Output {i}: {preview[:100]}...")

    # Formatting checks
    for response_data in outputs:
        if test_type == "litellm":
            response_data = response_data.model_dump()

        if test_type != "litellm":
            if backend == "vllm":
                from vllm.entrypoints.openai.protocol import CompletionResponse

                CompletionResponse.model_validate(response_data)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        for key in ["id", "object", "created", "model", "choices"]:
            assert key in response_data
            assert response_data[key] is not None

        for i, choice in enumerate(response_data["choices"]):
            assert "index" in choice and "text" in choice and "finish_reason" in choice
            assert choice["index"] == i and choice["finish_reason"] in ["stop", "length"]

            choice = response_data["choices"][i]
            assert "logprobs" in choice and choice["logprobs"] is not None
            assert "tokens" in choice["logprobs"]


# ------------------------------
# Tests for HTTP endpoint
# ------------------------------


@pytest.mark.vllm
def test_http_endpoint_completions_routing_and_batching(ray_init_fixture):
    """
    Since /completions endpoint supports both single and batched requests, and we support
    either using session_id or not, we test all combinations.

    We test with 2 engines of TP=1 to check if routing works.
    """
    batched_list = [False, True]
    with_traj_list = [False, True]

    try:
        # 1. Build engine
        cfg = get_test_actor_config(num_inference_engines=2, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        sampling_params = _get_test_sampling_params("vllm", cfg, "completions")
        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,  # since we do not explicitly sync weights
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # 2. Build prompts
        num_samples = 20
        test_prompts_conv_list: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)
        text_prompts = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in test_prompts_conv_list
        ]

        for batched in batched_list:
            for with_traj in with_traj_list:
                if not batched:
                    outputs = []
                    for i, p in enumerate(text_prompts):
                        payload = {"model": MODEL_QWEN2_5, "prompt": p, **sampling_params}
                        if with_traj:
                            payload["session_id"] = i
                        outputs.append(requests.post(f"{base_url}/completions", json=payload).json())
                else:
                    payload = {"model": MODEL_QWEN2_5, "prompt": text_prompts, **sampling_params}
                    if with_traj:
                        payload["session_id"] = list(range(len(text_prompts)))
                    outputs = [requests.post(f"{base_url}/completions", json=payload).json()]

                _check_completions_outputs(text_prompts, outputs, "request_posting", "vllm")
    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


# NOTE(Charlie): we do not test OpenAI client because it throws error when unsupported sampling params
# are passed into OpenAI.chat.completions.create() (e.g. min_tokens, skip_special_tokens, etc.),
# while these sampling params are used in vllm/sglang. Therefore, we instead use LiteLLM.
@pytest.mark.vllm
def test_http_endpoint_openai_api_with_weight_sync(ray_init_fixture):
    """
    Test the HTTP endpoint /chat/completions and /completions with policy weight sync.

    For `/completions`, we test by sending each prompt in its own request. For batching,
    behavior, we test in `test_http_endpoint_completions_routing_and_batching`.

    Besides we only tests single engine case.
    """
    test_types = ["request_posting", "aiohttp_client_session", "litellm"]
    endpoints = ["chat_completions", "completions"]
    try:
        # 1. Set up engine
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        client, pg, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=2,  # since we explicitly sync weights
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # Weight sync
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

        # 2. Do tests
        num_samples = 20
        test_prompts_conv_list: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)
        # For /completions, we test both string and token IDs input
        test_prompts_half_str_half_tokens_list: List[Union[str, List[int]]] = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in test_prompts_conv_list[: num_samples // 2]
        ] + [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            for conv in test_prompts_conv_list[num_samples // 2 :]
        ]

        def _generate_outputs(test_type, endpoint):
            def payload_builder(session_id, prompt):
                if endpoint == "chat_completions":
                    return {
                        "model": MODEL_QWEN2_5,
                        "messages": prompt,
                        "session_id": session_id,
                        **sampling_params,
                    }
                else:
                    return {
                        "model": MODEL_QWEN2_5,
                        "prompt": prompt,
                        "session_id": session_id,
                        **sampling_params,
                    }

            if endpoint == "chat_completions":
                path = "chat/completions"
                prompt_iterable = test_prompts_conv_list
                sampling_params = _get_test_sampling_params("vllm", cfg, "chat_completions")

            else:
                path = "completions"
                prompt_iterable = test_prompts_half_str_half_tokens_list
                sampling_params = _get_test_sampling_params("vllm", cfg, "completions")

            if test_type == "request_posting":

                def generate_output(session_id, prompt):
                    return requests.post(
                        f"{base_url}/{path}",
                        json=payload_builder(session_id, prompt),
                    ).json()

                with ThreadPoolExecutor() as executor:
                    output_tasks = [
                        executor.submit(generate_output, session_id, prompt)
                        for session_id, prompt in enumerate(prompt_iterable)
                    ]
                    outputs = [task.result() for task in output_tasks]

            elif test_type == "aiohttp_client_session":

                async def generate_outputs_async():
                    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
                    async with aiohttp.ClientSession(
                        connector=conn, timeout=aiohttp.ClientTimeout(total=None)
                    ) as session:
                        headers = {"Content-Type": "application/json"}
                        output_tasks = []
                        for session_id, prompt in enumerate(prompt_iterable):
                            payload = payload_builder(session_id, prompt)
                            output_tasks.append(session.post(f"{base_url}/{path}", json=payload, headers=headers))
                        responses = await asyncio.gather(*output_tasks)
                        return [await response.json() for response in responses]

                outputs = asyncio.run(generate_outputs_async())

            elif test_type == "litellm":

                async def generate_outputs_async():
                    async def generate_output(session_id, prompt):
                        if endpoint == "chat_completions":
                            return await litellm_async_completion(
                                model=f"openai/{MODEL_QWEN2_5}",
                                messages=prompt,
                                api_base=base_url,
                                api_key="DUMMY_KEY",
                                session_id=session_id,
                                **sampling_params,
                            )
                        else:
                            return await litellm_async_text_completion(
                                model=f"openai/{MODEL_QWEN2_5}",
                                prompt=[prompt],
                                api_base=base_url,
                                api_key="DUMMY_KEY",
                                session_id=[session_id],
                                **sampling_params,
                            )

                    tasks = [generate_output(session_id, prompt) for session_id, prompt in enumerate(prompt_iterable)]
                    return await asyncio.gather(*tasks)

                outputs = asyncio.run(generate_outputs_async())

            else:
                raise ValueError(f"Invalid test type: {test_type}")

            return outputs

        for test_type in test_types:
            for endpoint in endpoints:
                print(f"Testing {test_type} with {endpoint}")
                outputs = _generate_outputs(test_type, endpoint)
                if endpoint == "chat_completions":
                    _check_chat_completions_outputs(outputs, test_type, num_samples, "vllm")
                else:
                    _check_completions_outputs(test_prompts_half_str_half_tokens_list, outputs, test_type, "vllm")

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


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
def test_http_endpoint_with_remote_servers(ray_init_fixture, backend, tp_size):
    """Test sending both /chat/completions and /completions requests to remote servers."""
    endpoints = ["chat_completions", "completions"]

    try:
        # 1. Initialize InferenceEngineClient client with remote servers
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
        cfg.generator.backend = backend
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)

        client, remote_server_process = init_remote_inference_servers(tp_size, backend, tokenizer, cfg, MODEL_QWEN2_5)

        # 2. Start HTTP endpoint in background thread using serve function directly
        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # 3. Generate outputs using litellm and check outputs
        num_samples = 20
        test_prompts_conv_list: List[ConversationType] = get_test_prompts(MODEL_QWEN2_5, num_samples=num_samples)
        test_prompts_half_str_half_tokens_list: List[Union[str, List[int]]] = [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in test_prompts_conv_list[: num_samples // 2]
        ] + [
            tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
            for conv in test_prompts_conv_list[num_samples // 2 :]
        ]

        def _generate_outputs(endpoint):
            if endpoint == "chat_completions":
                sampling_params = _get_test_sampling_params(backend, cfg, "chat_completions")
                prompt_iterable = test_prompts_conv_list
            else:
                sampling_params = _get_test_sampling_params(backend, cfg, "completions")
                prompt_iterable = test_prompts_half_str_half_tokens_list

            # Default concurrency limit is 100 due to HTTP client pool capacity.
            async def generate_outputs_async():
                async def generate_output(session_id, prompt):
                    if endpoint == "chat_completions":
                        return await litellm_async_completion(
                            model=f"openai/{MODEL_QWEN2_5}",
                            messages=prompt,
                            api_base=base_url,
                            api_key="DUMMY_KEY",
                            session_id=session_id,
                            **sampling_params,
                        )
                    else:
                        return await litellm_async_text_completion(
                            model=f"openai/{MODEL_QWEN2_5}",
                            prompt=[prompt],
                            api_base=base_url,
                            api_key="DUMMY_KEY",
                            session_id=[session_id],
                            **sampling_params,
                        )

                tasks = [generate_output(session_id, prompt) for session_id, prompt in enumerate(prompt_iterable)]
                return await asyncio.gather(*tasks)

            return asyncio.run(generate_outputs_async())

        for endpoint in endpoints:
            outputs = _generate_outputs(endpoint)
            if endpoint == "chat_completions":
                _check_chat_completions_outputs(outputs, "litellm", num_samples, backend)
            else:
                _check_completions_outputs(test_prompts_half_str_half_tokens_list, outputs, "litellm", backend)

        # 4. Shutdown server
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if "remote_server_process" in locals():
            remote_server_process.terminate()
            remote_server_process.wait()


@pytest.mark.vllm
def test_structured_generation(ray_init_fixture):
    try:
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True  # Use colocate for simplicity
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,  # since we do not explicitly sync weights
        )

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

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
            model=f"openai/{MODEL_QWEN2_5}",
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
        assert json.loads(text) is not None, f"Output is not valid JSON: {text}"
    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


# TODO(Charlie): sglang has slightly different error response format. We need to handle it.
@pytest.mark.vllm
def test_http_endpoint_error_handling(ray_init_fixture, caplog):
    """
    Test error handling for various invalid requests and internal server errors.

    Tests validation errors (400) for invalid requests and verifies that internal
    server errors (500) are logged with traceback server-side (not exposed to client).
    """
    try:
        cfg = get_test_actor_config(num_inference_engines=2, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,  # since we do not explicitly sync weights
        )

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}"

        # Test 1: Invalid request - streaming not supported, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL_QWEN2_5, "messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "Streaming is not supported" in error_data["error"]["message"]

        # Test 2: OAI can take fields not listed in the protocol
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={"model": MODEL_QWEN2_5, "messages": [{"role": "user", "content": "Hello"}], "xxx": "yyy"},
        )
        assert response.status_code == HTTPStatus.OK  # 200

        # Test 3: Invalid request - missing required fields, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": MODEL_QWEN2_5,
                # Missing messages field
            },
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert error_data["error"]["message"] == "The field `messages` is required in your `/chat/completions` request."

        # Test 4: Invalid request - malformed JSON, raised by SkyRL
        response = requests.post(
            f"{base_url}/v1/chat/completions", data="some invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert "Invalid JSON error" in error_data["error"]["message"]  # JSON decode error

        # Test 5: Invalid request - empty messages array, raised by SkyRL
        response = requests.post(f"{base_url}/v1/chat/completions", json={"model": MODEL_QWEN2_5, "messages": []})
        assert response.status_code == HTTPStatus.BAD_REQUEST  # 400
        error_data = response.json()
        print(f"Error data: {error_data}")
        assert error_data["error"]["message"] == "The field `messages` in `/chat/completions` cannot be an empty list."

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

        # Test 8: Test internal server errors (500) return proper error responses
        # Traceback is logged server-side only (not exposed to client per CWE-209)
        caplog.set_level(logging.ERROR)
        original_client = http_endpoint_module._global_inference_engine_client

        internal_error_cases = [
            (
                "chat_completion",
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "Hello"}]},
                KeyError("choices"),
            ),
            ("completion", "/v1/completions", {"prompt": "Hello"}, RuntimeError("Simulated internal error")),
        ]
        for method_name, endpoint, extra_payload, exception in internal_error_cases:

            async def mock_raises(*args, exc=exception, **kwargs):
                raise exc

            caplog.clear()
            with patch.object(original_client, method_name, side_effect=mock_raises):
                response = requests.post(f"{base_url}{endpoint}", json={"model": MODEL_QWEN2_5, **extra_payload})
            assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
            error_data = response.json()
            error_message = error_data["error"]["message"]
            assert str(exception) in error_message or type(exception).__name__ in error_message
            assert "Traceback" not in error_message  # Not exposed to client (CWE-209)
            assert error_data["error"]["code"] == 500
            assert "Traceback (most recent call last):" in caplog.text  # Logged server-side
            assert type(exception).__name__ in caplog.text

        # Test 9: Context length errors return HTTP 400 (not 500)
        # This is tested in a separate test function with a custom max_model_len.
        # See test_context_length_error_returns_400() below.

        # Tests below are for `/completions` endpoint.
        # e.g. session id wrong length, etc.
        # Additional tests for /v1/completions
        # C1: streaming not supported
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"model": MODEL_QWEN2_5, "prompt": "Hello", "stream": True},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST
        # C2: wrong model
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"model": "wrong_model", "prompt": "Hello"},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST
        # C3: malformed json
        response = requests.post(
            f"{base_url}/v1/completions",
            data="some invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST
        # C4: n > 1
        response = requests.post(
            f"{base_url}/v1/completions",
            json={"model": MODEL_QWEN2_5, "prompt": "Hello", "n": 2},
        )
        assert response.status_code == HTTPStatus.BAD_REQUEST
        error_data = response.json()
        assert "n is not supported in SkyRL for /completions " in error_data["error"]["message"]

        # When batched and session_id wrong length -> 400 from server or client-side error
        bad_payload = {"model": MODEL_QWEN2_5, "prompt": ["hi", "hello", "ok"], "session_id": [0, 1]}
        r = requests.post(f"{base_url}/v1/completions", json=bad_payload)
        assert r.status_code == HTTPStatus.BAD_REQUEST

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


@pytest.mark.vllm
@pytest.mark.parametrize("use_custom_template", [False, True])
def test_http_endpoint_custom_chat_template(ray_init_fixture, use_custom_template):
    """
    Test the HTTP endpoint /chat/completions with and without custom chat template.
    Check the output correspondingly.
    """
    try:
        # 1. Set up engine
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN3)
        cfg.trainer.placement.colocate_all = True  # Use colocate for simplicity
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        template_path = "skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"
        engine_init_kwargs = {}
        if use_custom_template:
            # use relative path to workspace root
            # __file__ is skyrl-train/tests/gpu/gpu_ci/test_inference_engine_client_http_endpoint.py
            repo_root = Path(__file__).parent.parent.parent.parent
            engine_init_kwargs["chat_template"] = str(repo_root / template_path)

        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN3,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,  # since we do not explicitly sync weights
            engine_init_kwargs=engine_init_kwargs,
        )

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

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

        response = requests.post(f"{base_url}/chat/completions", json=payload)
        assert response.status_code == HTTPStatus.OK
        data = response.json()

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
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


@pytest.mark.vllm
def test_http_endpoint_served_model_name(ray_init_fixture):
    """
    Test that `generator.served_model_name` allows using a different model name in requests
    than the actual model path.

    This is useful when:
    - The model path is a local path or HuggingFace path that differs from the desired API model name
    - Using LiteLLM or other clients that expect a specific model name format
    - Harbor deployments where the served model name differs from the underlying model path

    See: https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
    """
    # Use a custom served model name that differs from the actual model path
    SERVED_MODEL_NAME = "my-custom-model-alias"

    try:
        # 1. Set up engine with served_model_name
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        # Set the served_model_name to be different from the model path
        cfg.generator.served_model_name = SERVED_MODEL_NAME

        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,  # since we do not explicitly sync weights
        )

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # 2. Test that requests with the served_model_name work
        messages = [{"role": "user", "content": "Hello, who are you?"}]
        payload = {
            "model": SERVED_MODEL_NAME,  # Use the served model name, not the path
            "messages": messages,
            "max_tokens": 50,
        }

        response = requests.post(f"{base_url}/chat/completions", json=payload)
        assert (
            response.status_code == HTTPStatus.OK
        ), f"Request with served_model_name failed: {response.status_code}, {response.json()}"
        data = response.json()
        assert "choices" in data and len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"] is not None

        # 3. Test that requests with the original model path should now fail
        # (since we're serving under a different name)
        payload_with_path = {
            "model": MODEL_QWEN2_5,  # Use the actual model path
            "messages": messages,
            "max_tokens": 50,
        }
        response = requests.post(f"{base_url}/chat/completions", json=payload_with_path)
        assert (
            response.status_code == HTTPStatus.BAD_REQUEST
        ), f"Request with model path should fail when served_model_name is set: {response.status_code}"
        error_data = response.json()
        assert "Model name mismatch" in error_data["error"]["message"]

        # 4. Test /completions endpoint as well
        tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN2_5)
        text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        completions_payload = {
            "model": SERVED_MODEL_NAME,
            "prompt": text_prompt,
            "max_tokens": 50,
        }
        response = requests.post(f"{base_url}/completions", json=completions_payload)
        assert (
            response.status_code == HTTPStatus.OK
        ), f"Completions request with served_model_name failed: {response.status_code}, {response.json()}"

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)


@pytest.mark.vllm
def test_context_length_error_returns_400(ray_init_fixture):
    """
    Test that context length errors return HTTP 400 (Bad Request), not 500.

    This is important for LiteLLM/Harbor integration: HTTP 400 gets wrapped as
    BadRequestError, which Harbor can detect as ContextLengthExceededError.
    HTTP 500 would be wrapped as InternalServerError and not detected.

    Tests both raw HTTP requests and LiteLLM client behavior.
    """
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    # Use a small max_model_len to make testing faster
    TEST_MAX_MODEL_LEN = 1024

    try:
        cfg = get_test_actor_config(num_inference_engines=1, model=MODEL_QWEN2_5)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, _, router, server_group = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL_QWEN2_5,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,
            engine_init_kwargs={"max_model_len": TEST_MAX_MODEL_LEN},
        )

        server_thread, server_port = set_up_http_server(client)
        base_url = f"http://{SERVER_HOST}:{server_port}/v1"

        # Test 1: Prompt alone exceeds max_model_len (1024) -> HTTP 400
        # vllm serve returns: "This model's maximum context length is 1024 tokens. However, your
        # request has {n} input tokens. Please reduce the length of the input messages."
        messages_oversized = [{"role": "user", "content": "hello " * 1500}]
        response = requests.post(
            f"{base_url}/chat/completions",
            json={"model": MODEL_QWEN2_5, "messages": messages_oversized},
        )
        assert (
            response.status_code == HTTPStatus.BAD_REQUEST
        ), f"Expected HTTP 400 for oversized prompt, got {response.status_code}: {response.json()}"
        error_data = response.json()
        assert "error" in error_data
        error_message = error_data["error"]["message"]
        assert (
            "maximum context length" in error_message.lower()
        ), f"Error message should mention 'maximum context length': {error_message}"

        # Test 2: Prompt fits, but prompt + max_tokens exceeds max_model_len -> HTTP 400
        # vllm serve returns: "'max_tokens' or 'max_completion_tokens' is too large: {max_tokens}.
        # This model's maximum context length is {max_model_len} tokens and your request has {n}
        # input tokens ({max_tokens} > {max_model_len} - {n})."
        messages_medium = [{"role": "user", "content": "hello " * 500}]
        response = requests.post(
            f"{base_url}/chat/completions",
            json={"model": MODEL_QWEN2_5, "messages": messages_medium, "max_tokens": 1000},
        )
        assert (
            response.status_code == HTTPStatus.BAD_REQUEST
        ), f"Expected HTTP 400 for prompt+max_tokens overflow, got {response.status_code}: {response.json()}"
        error_data = response.json()
        assert "error" in error_data
        error_message = error_data["error"]["message"]
        assert (
            "maximum context length" in error_message.lower()
        ), f"Error message should mention 'maximum context length': {error_message}"

        # Test 3: Valid request still works (regression test)
        response = requests.post(
            f"{base_url}/chat/completions",
            json={"model": MODEL_QWEN2_5, "messages": messages_medium, "max_tokens": 10},
        )
        assert (
            response.status_code == HTTPStatus.OK
        ), f"Expected HTTP 200 for valid request, got {response.status_code}: {response.json()}"

        # Test 4: LiteLLM wraps prompt+max_tokens error as BadRequestError (not InternalServerError).
        # This is critical for Harbor's ContextLengthExceededError detection.
        # Uses the same prompt+max_tokens case as Test 2.
        async def make_litellm_call():
            return await litellm_async_completion(
                model=f"hosted_vllm/{MODEL_QWEN2_5}",
                messages=messages_medium,
                api_base=base_url,
                api_key="DUMMY_KEY",
                max_tokens=1000,
                num_retries=0,
            )

        with pytest.raises(LiteLLMBadRequestError) as excinfo:
            asyncio.run(make_litellm_call())
        exception_raised = excinfo.value

        assert exception_raised is not None
        error_str = str(exception_raised).lower()
        assert (
            "maximum context length" in error_str
        ), f"Error message should mention 'maximum context length': {str(exception_raised)[:200]}"

    finally:
        shutdown_server(host=SERVER_HOST, port=server_port, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)
