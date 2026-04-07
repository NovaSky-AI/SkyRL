import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from .utils import ZeroCopySHMBuffer

logger = logging.getLogger(__name__)

class AtroposSHMGenerator(GeneratorInterface):
    """
    SkyRL Generator that pulls rollouts from Atropos via Zero-Copy Shared Memory.
    """

    def __init__(
        self,
        shm_name: str,
        batch_size: int,
        entry_size: int = 4096,
        poll_interval: float = 0.001,
        timeout: float = 300.0,
        **kwargs
    ):
        try:
            self.shm = ZeroCopySHMBuffer(name=shm_name, size=batch_size * 10, entry_size=entry_size, create=False)
            logger.info(f"AtroposSHMGenerator attached to SHM: {shm_name}")
        except Exception as e:
            logger.error(f"Failed to attach to SHM: {e}")
            raise

        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.timeout = timeout
        
        # Trajectory stash for out-of-order collection
        self.stash: Dict[str, Dict[str, Any]] = {}
        self.running = True

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Polls SHM for the specific trajectories requested in the batch.
        """
        target_ids: List[TrajectoryID] = input_batch.get("trajectory_ids", [])
        if not target_ids:
            return self._empty_output()

        results: Dict[str, Dict[str, Any]] = {}
        target_keys: Set[str] = {tid.to_string() for tid in target_ids}
        
        # Check stash first
        for key in list(self.stash.keys()):
            if key in target_keys:
                results[key] = self.stash.pop(key)

        # Poll SHM until all targets found or timeout
        start_time = time.time()
        while len(results) < len(target_keys) and (time.time() - start_time) < self.timeout:
            if not self.running: break
            entry = self.shm.read_next()
            if entry:
                key = entry["instance_id"]
                if key in target_keys:
                    results[key] = entry
                else:
                    self.stash[key] = entry
            else:
                await asyncio.sleep(self.poll_interval)

        # Format output in requested order
        output = self._empty_output_from_batch(len(target_ids))
        for i, tid in enumerate(target_ids):
            key = tid.to_string()
            if key in results:
                res = results[key]
                output["response_ids"][i] = res["tokens"]
                output["rewards"][i] = res["rewards"] if "rewards" in res else res.get("score", 0.0)
                output["loss_masks"][i] = [1] * len(res["tokens"])
                output["trajectory_ids"][i] = tid
                # Extract logprobs from metadata if present
                meta = res.get("metadata", {})
                output["rollout_logprobs"][i] = meta.get("logprobs")
                output["stop_reasons"][i] = meta.get("stop_reasons")
            else:
                # Mask out missing results to avoid breaking trainer
                output["loss_masks"][i] = [0]
                output["trajectory_ids"][i] = tid
                output["rollout_logprobs"][i] = None
                output["stop_reasons"][i] = None

        return output

    def _empty_output(self) -> GeneratorOutput:
        return {
            "prompt_token_ids": [],
            "response_ids": [],
            "rewards": [],
            "loss_masks": [],
            "trajectory_ids": [],
            "rollout_logprobs": [],
            "stop_reasons": []
        }

    def _empty_output_from_batch(self, size: int) -> GeneratorOutput:
        return {
            "prompt_token_ids": [[] for _ in range(size)],
            "response_ids": [[] for _ in range(size)],
            "rewards": [0.0 for _ in range(size)],
            "loss_masks": [[0] for _ in range(size)],
            "trajectory_ids": [None for _ in range(size)],
            "rollout_logprobs": [None for _ in range(size)],
            "stop_reasons": [None for _ in range(size)]
        }

    def close(self):
        self.running = False
        self.shm.close()
