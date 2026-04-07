from typing import Any, Dict, List, Optional, Union
from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput

class NoOpInferenceClient(InferenceEngineClient):
    """
    A formalized No-Op client for SkyRL external generation (e.g. Atropos-SHM).
    
    This client implements the InferenceEngineClient interface but no-ops all 
    infrastructure synchronization and generation calls, allowing the trainer 
    to operate in a standalone mode.
    """
    
    def __init__(self, *args, **kwargs):
        # We don't call super().__init__ because we don't want to initialize engines or HTTP ports.
        self.backend = "none"
        self.engines = []
        logger.info("NoOpInferenceClient initialized (Backend: none). All inference calls will be no-ops.")

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generation is handled externally, so this returns empty/dummy results if called."""
        return InferenceEngineOutput(
            responses=[],
            stop_reasons=[],
            response_ids=[],
            response_logprobs=None,
            rollout_expert_indices=None,
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        """No-op wake up."""
        return None

    async def sleep(self, *args: Any, **kwargs: Any):
        """No-op sleep."""
        return None

    async def init_weight_update_communicator(self, *args, **kwargs):
        """No-op weight synchronization initialization."""
        return None

    async def update_named_weights(self, *args, **kwargs):
        """No-op weight update."""
        return None

    async def broadcast_to_inference_engines(self, *args, **kwargs):
        """No-op weight broadcast."""
        return None

    def dp_size(self) -> int:
        return 1

    def tp_size(self) -> int:
        return 1

    def pp_size(self) -> int:
        return 1

    def __getattr__(self, name):
        """Catch-all for any other interface methods to ensure robustness."""
        return lambda *args, **kwargs: None
