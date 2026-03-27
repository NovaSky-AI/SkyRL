import ray
import uvicorn
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.engine import TinkerEngine
from skyrl.tinker.api import app

@ray.remote
class TinkerAPIActor:
    """Ray Actor wrapper for the Tinker API server (FastAPI + Uvicorn)."""
    def __init__(self, config: EngineConfig):
        self.config = config

    def run(self, host: str, port: int):
        app.state.engine_config = self.config
        # Logging config can be customized if needed
        from skyrl.utils.log import get_uvicorn_log_config
        uvicorn.run(app, host=host, port=port, log_config=get_uvicorn_log_config())


@ray.remote
class TinkerEngineActor:
    """Ray Actor wrapper for the Tinker background engine loop."""
    def __init__(self, config: EngineConfig):
        self.config = config
    
    def run(self):
        engine = TinkerEngine(self.config)
        engine.run()
