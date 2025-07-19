"""
Test the HTTP server with OpenAI client and policy weight sync.

This uses the same workflow as test_policy_vllm_e2e.py, but with the HTTP server instead of
the inference client engine.

Run with:
uv run --isolated --extra dev --extra vllm pytest tests/gpu/test_http_inference_engine_client.py
"""

import pytest
import asyncio
import ray
import hydra
import threading
import requests
import aiohttp
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from transformers import AutoTokenizer
from ray.util.placement_group import placement_group
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.inference_engines.launch_inference_engine_http_server import serve, wait_for_server_ready, shutdown_server
from openai import OpenAI
from .test_policy_vllm_e2e import init_inference_engines
from concurrent.futures import ThreadPoolExecutor


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

@pytest.mark.parametrize("test_type", ["chat_completions_create", "request_posting", "aiohttp_client_session"])
def test_http_server_openai_api_with_weight_sync(test_type):
    """
    Test the HTTP server with OpenAI client and policy weight sync.
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = True  # Use colocate for simplicity
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"

        client, pg = init_inference_engines(
            cfg=cfg,
            v1=True,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
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

        num_samples = 50
        test_prompts = get_test_prompts(MODEL, num_samples=num_samples)

        # Generate outputs based on test type
        if test_type == "chat_completions_create":
            # 1.1 Test chat.completions.create
            # Create OpenAI client (with dummy API key since we don't authenticate)
            openai_client = OpenAI(
                base_url=base_url,
                api_key="dummy-key"  # Our server doesn't authenticate, but OpenAI client requires a key
            )
            def generate_output(prompt):
                return openai_client.chat.completions.create(
                    model=MODEL,
                    messages=prompt,
                ).model_dump()
            
            with ThreadPoolExecutor() as executor:
                output_tasks = [
                    executor.submit(generate_output, prompt) for prompt in test_prompts
                ]
                outputs = [task.result() for task in output_tasks]
                
        elif test_type == "request_posting":
            # 1.2 Test request posting
            def generate_output(prompt):
                return requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": prompt,
                    }
                ).json()
            
            with ThreadPoolExecutor() as executor:
                output_tasks = [
                    executor.submit(generate_output, prompt) for prompt in test_prompts
                ]
                outputs = [task.result() for task in output_tasks]
                
        elif test_type == "aiohttp_client_session":
            # 1.3 Test aiohttp.ClientSession
            async def generate_outputs_async():
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
                    headers = {"Content-Type": "application/json"}
                    output_tasks = []
                    
                    for prompt in test_prompts:
                        payload = {
                            "model": MODEL,
                            "messages": prompt,
                        }
                        output_tasks.append(
                            session.post(f"{base_url}/chat/completions", json=payload, headers=headers)
                        )
                    
                    responses = await asyncio.gather(*output_tasks)
                    return [await response.json() for response in responses]
            
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
                assert "role" in message and "content" in message and message["role"] == "assistant"

        # Shutdown server
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        if server_thread.is_alive():
            server_thread.join(timeout=5)

    finally:
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        ray.shutdown()


# def test_http_server_error_handling():
#     """
#     Test error handling for various invalid requests.
#     """
#     try:
#         cfg = get_test_actor_config()
#         cfg.trainer.placement.colocate_all = True
#         cfg.generator.weight_sync_backend = "nccl"
#         cfg.trainer.strategy = "fsdp2"

#         client, policy, pg = init_inference_engines(
#             cfg=cfg,
#             v1=True,
#             use_local=True,
#             async_engine=cfg.generator.async_engine,
#             tp_size=cfg.generator.inference_engine_tensor_parallel_size,
#             colocate_all=cfg.trainer.placement.colocate_all,
#         )

#         from skyrl_train.inference_engines.launch_inference_engine_http_server import serve, wait_for_server_ready
        
#         # Start server in background thread
#         def run_server():
#             serve(client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")
        
#         server_thread = threading.Thread(target=run_server, daemon=True)
#         server_thread.start()
        
#         # Wait for server to be ready
#         wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)
        
#         base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        
#         # Test 1: Invalid request - streaming not supported
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             json={
#                 "model": "test-model",
#                 "messages": [{"role": "user", "content": "Hello"}],
#                 "stream": True
#             }
#         )
#         assert response.status_code == 422
#         error_data = response.json()
#         assert "detail" in error_data
#         # Pydantic returns detailed field validation errors
#         assert any("stream" in str(detail) for detail in error_data["detail"])
        
#         # Test 2: Invalid request - tools not supported
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             json={
#                 "model": "test-model", 
#                 "messages": [{"role": "user", "content": "Hello"}],
#                 "tools": [{"type": "function", "function": {"name": "test"}}]
#             }
#         )
#         assert response.status_code == 422
#         error_data = response.json()
#         assert "detail" in error_data
#         assert any("tools" in str(detail) for detail in error_data["detail"])
        
#         # Test 3: Invalid request - response_format not supported
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             json={
#                 "model": "test-model",
#                 "messages": [{"role": "user", "content": "Hello"}],
#                 "response_format": {"type": "json_object"}
#             }
#         )
#         assert response.status_code == 422
#         error_data = response.json()
#         assert "detail" in error_data
#         assert any("response_format" in str(detail) for detail in error_data["detail"])
        
#         # Test 4: Invalid request - missing required fields
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             json={
#                 "model": "test-model"
#                 # Missing messages field
#             }
#         )
#         assert response.status_code == 422
#         error_data = response.json()
#         assert "detail" in error_data
#         assert any("messages" in str(detail) for detail in error_data["detail"])
        
#         # Test 5: Invalid request - malformed JSON
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             data="invalid json",
#             headers={"Content-Type": "application/json"}
#         )
#         assert response.status_code == 422
        
#         # Test 6: Invalid request - empty messages array
#         response = requests.post(
#             f"{base_url}/v1/chat/completions",
#             json={
#                 "model": "test-model",
#                 "messages": []
#             }
#         )
#         assert response.status_code == 422
#         error_data = response.json()
#         assert "detail" in error_data
        
#         # Test 7: Health check endpoint should work
#         response = requests.get(f"{base_url}/health")
#         assert response.status_code == 200
#         health_data = response.json()
#         assert health_data["status"] == "healthy"
        
#     finally:
#         ray.shutdown()

