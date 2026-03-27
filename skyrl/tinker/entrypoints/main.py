import argparse
import ray

from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.ray_actors import TinkerAPIActor, TinkerEngineActor
from skyrl.utils.log import logger

def main():
    parser = argparse.ArgumentParser(description="SkyRL Tinker Ray Orchestrator")
    add_model(parser, EngineConfig)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to for API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to for API Server")
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments
    config = EngineConfig.model_validate({k: v for k, v in vars(args).items() if k in EngineConfig.model_fields})
    
    # Force Ray orchestrated mode and ray_jax backend
    config.use_ray = True
    config.backend = "ray_jax"
    
    logger.info(f"Initializing Ray with address: {config.ray_address or 'local'}")
    ray.init(address=config.ray_address)

    logger.info(f"Starting Tinker API Actor on {args.host}:{args.port}")
    api_actor = TinkerAPIActor.remote(config)
    api_task = api_actor.run.remote(args.host, args.port)

    logger.info("Starting Tinker Engine Actor")
    engine_actor = TinkerEngineActor.remote(config)
    engine_task = engine_actor.run.remote()

    logger.info("Ray Orchestrator running. Waiting for actors to complete.")
    try:
        ray.get([api_task, engine_task])
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
