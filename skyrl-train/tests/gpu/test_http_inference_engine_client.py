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

@pytest.mark.parametrize("test_type", ["chat_completions_create", "request_posting"])
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

        test_prompts = get_test_prompts(MODEL, num_samples=2)

        # Generate outputs
        if test_type == "chat_completions_create":
            # 1.1 Test chat.completions.create
            # Create OpenAI client (with dummy API key since we don't authenticate)
            openai_client = OpenAI(
                base_url=base_url,
                api_key="dummy-key"  # Our server doesn't authenticate, but OpenAI client requires a key
            )
            outputs = []
            for prompt in test_prompts:
                # Convert our ConversationType to OpenAI format
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt]
                
                response = openai_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    frequency_penalty=0.1,
                    presence_penalty=0.2,
                )
                print(f"Generated response: {response.choices[0].message.content[:100]}...")
                # Use model_dump() instead of deprecated json() method
                outputs.append(response.model_dump())

        else:
            # 1.2 Test request posting
            outputs = []
            for prompt in test_prompts:
                messages = [{"role": msg["role"], "content": msg["content"]} for msg in prompt]
                response = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "max_tokens": 50,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "frequency_penalty": 0.1,
                        "presence_penalty": 0.2,
                    }
                )
                assert response.status_code == 200, f"Response: {response.text}"
                response_data = response.json()
                print(f"Generated response: {response_data['choices'][0]['message']['content'][:100]}...")
                outputs.append(response_data)

        # 2. Check response structure
        for response_data in outputs:
            for key in ["id", "object", "created", "model", "choices", "usage"]:
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

