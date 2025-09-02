"""
Example of using LiteLLM with SkyRL's HTTP inference server. This simulates what your custom
generator looks like when posting requests to the server (as opposed to using `skyrl_gym_generator.py`).

When you want to use the HTTP endpoint, set the following configs in your bash script, as shown in
`run_gsm8k_with_inference_http_endpoint.sh`:
```
generator:
  enable_http_endpoint: true
  http_endpoint_host: "127.0.0.1"
  http_endpoint_port: 8000
```

Note that `init_to_simulate_trainer()` is not needed in your custom generator, we
do it here to simulate what the trainer will instantiate.

Also note that `trajectory_id` is important for better trajectory routing for prefix cache reuse.

Run with:
uv add litellm
uv run --isolated --extra dev --extra vllm python examples/inference_http_endpoint/rollout_with_http_endpoint_litellm.py
"""

import threading
import ray
import hydra
from omegaconf import DictConfig
from litellm import completion
from concurrent.futures import ThreadPoolExecutor
from skyrl_train.inference_engines.inference_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
from skyrl_train.entrypoints.main_base import config_dir
from tests.gpu.utils import init_inference_engines
from uuid import uuid4

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


def init_to_simulate_trainer():
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


def agent_loop(prompt: str, base_url: str):
    chat_history = [
        {"role": "user", "content": prompt},
    ]
    max_turns = 3
    trajectory_id = uuid4().hex

    while True:
        response = completion(
            model=f"openai/{MODEL}",  # Add openai/ prefix for custom endpoints
            messages=chat_history,
            api_base=base_url,
            # Otherwise runs into: litellm.llms.openai.common_utils.OpenAIError: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable
            api_key="DUMMY_KEY",
            temperature=0.7,
            max_tokens=100,
            # NOTE(Charlie): this is needed for better trajectory routing for prefix cache reuse
            trajectory_id=trajectory_id,
        )
        chat_history.append(response.choices[0].message)
        # dummy multi-turn chat
        max_turns -= 1
        if max_turns <= 0:
            break
        else:
            chat_history.append({"role": "user", "content": "Repeat what you just said."})
    return chat_history


def main():
    try:
        # 1. This method simulates what the trainer will do. When you set
        # `generator.enable_http_endpoint=True` in bash script,
        # you do not need to write these code at all. In your custom generator,
        # you can simply post request to `base_url`. The server will be ready automatically
        # before `CustomGenerator.generate()` is called.
        init_to_simulate_trainer()
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

        # 2. Pretend we have 500 tasks, and run one agent loop for each task parallely.
        prompts = ["Hello, how are you?"] * 500
        with ThreadPoolExecutor() as executor:
            output_tasks = [executor.submit(agent_loop, prompt, base_url) for prompt in prompts]
            outputs = [task.result() for task in output_tasks]
        print(f"len(outputs): {len(outputs)}")
        print(f"outputs[0]: {outputs[0]}")
        for output in outputs:
            assert len(output) == 3 * 2

    finally:
        # You do not need to call this in your custom generator either.
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        ray.shutdown()


if __name__ == "__main__":
    main()
