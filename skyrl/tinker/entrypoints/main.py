import time
import argparse
import ray

from skyrl.tinker.config import APIConfig, EngineConfig, SkyRLTxConfig, add_model
from skyrl.backends.jax import JaxBackendConfig
from skyrl.tinker.ray import run_ray_detached_actors
from skyrl.utils.log import logger

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
    config = SkyRLTxConfig(api=api_config, engine=engine_config, jax_backend=jax_backend_config)
    
    # Force Ray orchestrated mode and ray_jax backend
    config.engine.use_ray = True
    config.engine.backend = "ray_jax"
    
    logger.info(f"Initializing Ray with address: {config.engine.ray_address or 'local'}")
    ray.init(address=config.engine.ray_address)
    run_ray_detached_actors(config)

    time.sleep(100)

    service_client = tinker.ServiceClient(base_url="http://localhost:8000", api_key="tml-dummy")
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
