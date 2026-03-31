import time
import argparse
import ray
import threading
import uvicorn

from skyrl.tinker.config import APIConfig, EngineConfig, SkyRLTxConfig, add_model
from skyrl.backends.jax import JaxBackendConfig
from skyrl.utils.log import logger
from skyrl.tinker.api import app
from skyrl.tinker.engine import TinkerEngine

import tinker
import numpy as np
from tinker import types


def process_example(example, tokenizer):
    prompt = f"English: {example['input']}\nPig Latin:"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {example['output']}\n\n", add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),
        loss_fn_inputs=dict(weights=weights[1:], target_tokens=tokens[1:])
    )

def main():
    logger.info("Starting entrypoint...")
    parser = argparse.ArgumentParser(description="SkyRL Tinker Ray Orchestrator")
    add_model(parser, SkyRLTxConfig)
    args = parser.parse_args()

    # Create config from parsed arguments
    api_config = APIConfig.model_validate({k: v for k, v in vars(args).items() if k in APIConfig.model_fields})
    engine_config = EngineConfig.model_validate({k: v for k, v in vars(args).items() if k in EngineConfig.model_fields})
    jax_backend_config = JaxBackendConfig.model_validate({k: v for k, v in vars(args).items() if k in JaxBackendConfig.model_fields})

    engine_config.backend_config = jax_backend_config.model_dump()
    engine_config.database_url = "sqlite:////tmp/tinker.db"
    config = SkyRLTxConfig(api=api_config, engine=engine_config, jax_backend=jax_backend_config)

    # Force Ray orchestrated mode and ray_jax backend
    config.engine.use_ray = True
    config.engine.backend = "ray_jax"

    logger.info(f"Initializing Ray with address: {config.engine.ray_address or 'local'}")
    ray.init(address=config.engine.ray_address)
    logger.info("Starting Tinker API and Engine in driver process threads...")

    app.state.engine_config = config.engine

    def run_api():
        from skyrl.utils.log import get_uvicorn_log_config
        uvicorn.run(app, host=config.api.host, port=config.api.port, log_config=get_uvicorn_log_config())

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    engine_instance = TinkerEngine(config.engine)

    def run_engine():
        engine_instance.run()

    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()

    tinker_address = "localhost"
    logger.info(f"Tinker API and Engine started. API address: {tinker_address}:{config.api.port}")

    logger.info("Waiting for services to initialize...")
    time.sleep(30)
    service_client = tinker.ServiceClient(base_url=f"http://{tinker_address}:8000", api_key="tml-dummy")
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B")
    tokenizer = training_client.get_tokenizer()

    # Training examples
    examples = [
        {"input": "banana split", "output": "anana-bay plit-say"},
        {"input": "quantum physics", "output": "uantum-qay ysics-phay"},
        {"input": "coding wizard", "output": "oding-cay izard-way"},
    ]

    processed = [process_example(ex, tokenizer) for ex in examples]

    # Training loop
    for _ in range(6):
        fwdbwd = training_client.forward_backward(processed, "cross_entropy").result()
        training_client.optim_step(types.AdamParams(learning_rate=1e-4)).result()

        logprobs = np.concatenate([o['logprobs'].tolist() for o in fwdbwd.loss_fn_outputs])
        weights = np.concatenate([e.loss_fn_inputs['weights'].tolist() for e in processed])
        print(f"Loss: {-np.dot(logprobs, weights) / weights.sum():.4f}")


if __name__ == "__main__":
    main()
