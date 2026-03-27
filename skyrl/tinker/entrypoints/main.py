import argparse
import ray

from skyrl.tinker.config import APIConfig, EngineConfig, SkyRLTxConfig, add_model
from skyrl.backends.jax import JaxBackendConfig
from skyrl.tinker.ray_actors import TinkerAPIActor, TinkerEngineActor
from skyrl.utils.log import logger

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

    logger.info(f"Starting Tinker API Actor on {config.api.host}:{config.api.port}")
    api_actor = TinkerAPIActor.remote(config.engine)
    api_task = api_actor.run.remote(config.api.host, config.api.port)

    logger.info("Starting Tinker Engine Actor")
    engine_actor = TinkerEngineActor.remote(config.engine)
    engine_task = engine_actor.run.remote()

    logger.info("Ray Orchestrator running. Waiting for actors to complete.")
    try:
        ray.get([api_task, engine_task])
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
