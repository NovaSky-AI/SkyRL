import ray
import uvicorn

from skyrl.tinker.config import EngineConfig, SkyRLTxConfig
from ray.util.placement_group import placement_group
from skyrl.utils.log import logger


def run_ray_detached_actors(config: SkyRLTxConfig):
    logger.info("Creating STRICT_PACK placement group to colocate API and Engine actors...")

    pg = placement_group([{"CPU": 4, "TPU": 8}], strategy="STRICT_PACK")
    ray.get(pg.ready())

    logger.info(f"Starting Tinker API Actor on {config.api.host}:{config.api.port} (Detached)...")
    api_actor = TinkerAPIActor.options(
        num_cpus=1,
        placement_group=pg,
        name="tinker_api",
        lifetime="detached"
    ).remote(config.engine)
    address = ray.get(api_actor.get_ip_address.remote())
    api_actor.run.remote(config.api.host, config.api.port)

    logger.info("Starting Tinker Engine Actor (Detached)...")
    engine_actor = TinkerEngineActor.options(
        num_cpus=1,
        placement_group=pg,
        name="tinker_engine",
        lifetime="detached"
    ).remote(config.engine)
    engine_actor.run.remote()

    logger.info("Ray actors started in detached mode. They will keep running. You can now run your training script.")
    return address


@ray.remote
class TinkerAPIActor:
    """Ray Actor wrapper for the Tinker API server (FastAPI + Uvicorn)."""
    def __init__(self, config: EngineConfig):
        self.config = config

    def run(self, host: str, port: int):
        from skyrl.tinker.api import app
        app.state.engine_config = self.config
        # Logging config can be customized if needed
        from skyrl.utils.log import get_uvicorn_log_config
        uvicorn.run(app, host=host, port=port, log_config=get_uvicorn_log_config())

    def get_ip_address(self):
        return ray.util.get_node_ip_address()



@ray.remote
class TinkerEngineActor:
    """Ray Actor wrapper for the Tinker background engine loop."""
    def __init__(self, config: EngineConfig):
        self.config = config

    def run(self):
        from skyrl.tinker.engine import TinkerEngine
        engine = TinkerEngine(self.config)
        engine.run()

