import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from transformers import AutoTokenizer

from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.atropos_shm_utils import ZeroCopySHMBuffer

logger = logging.getLogger(__name__)


class AtroposSHMGenerator(GeneratorInterface):
    """
    SkyRL Generator that pulls rollouts from Atropos via Zero-Copy Shared Memory.
    Bypasses HTTP/vLLM for ultra-low latency training.
    """

    def __init__(
        self,
        shm_name: str = "atropos_shm",
        shm_size: int = 1000,
        tokenizer_name: str = "NousResearch/DeepHermes-3-Llama-3-3B-Preview",
        poll_interval: float = 0.05,
        timeout: float = 300.0,
    ):
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.poll_interval = poll_interval
        self.timeout = timeout
        
        # Attach to existing SHM (created by Atropos Env)
        try:
            self.shm = ZeroCopySHMBuffer(name=shm_name, size=shm_size, create=False)
            logger.info(f"AtroposSHMGenerator attached to SHM: {shm_name}")
        except Exception as e:
            logger.error(f"Failed to attach to Atropos SHM: {e}")
            raise

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Local stash for out-of-order arrivals
        self.stash: Dict[str, Dict[str, Any]] = {}

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Polls SHM for the specific trajectories requested in the batch.
        """
        target_ids: List[TrajectoryID] = input_batch.get("trajectory_ids", [])
        if not target_ids:
            logger.warning("AtroposSHMGenerator received empty trajectory_ids. Falling back to streaming.")
            return self._empty_output()

        results: Dict[str, Dict[str, Any]] = {}
        target_keys: Set[str] = {tid.to_string() for tid in target_ids}
        
        start_time = time.time()
        
        while len(results) < len(target_keys):
            if time.time() - start_time > self.timeout:
                logger.error(f"Timeout waiting for SHM results. Found {len(results)}/{len(target_keys)}")
                break

            # 1. Check stash first
            for key in list(self.stash.keys()):
                if key in target_keys and key not in results:
                    results[key] = self.stash.pop(key)

            if len(results) == len(target_keys):
                break

            # 2. Poll SHM for next available
            item = self.shm.read_next()
            if item:
                # Key is instance_id_repetition_id
                key = f"{item['instance_id']}_{item['repetition_id']}"
                if key in target_keys:
                    results[key] = item
                else:
                    # Not for this batch, stash it for later
                    self.stash[key] = item
            else:
                await asyncio.sleep(self.poll_interval)

        # Format output in the same order as input_batch
        output = self._empty_output_from_batch(len(target_ids))
        for i, tid in enumerate(target_ids):
            key = tid.to_string()
            if key in results:
                res = results[key]
                output["response_ids"][i] = res["tokens"]
                output["rewards"][i] = res["score"]
                output["loss_masks"][i] = [1] * len(res["tokens"]) # Simplified mask
                output["trajectory_ids"][i] = tid
            else:
                # Missing result: mask it out
                output["loss_masks"][i] = [0]
                output["trajectory_ids"][i] = tid

        return output

    def _empty_output(self) -> GeneratorOutput:
        return {
            "prompt_token_ids": [],
            "response_ids": [],
            "rewards": [],
            "loss_masks": [],
            "trajectory_ids": []
        }

    def _empty_output_from_batch(self, batch_size: int) -> GeneratorOutput:
        return {
            "prompt_token_ids": [[] for _ in range(batch_size)],
            "response_ids": [[] for _ in range(batch_size)],
            "rewards": [0.0 for _ in range(batch_size)],
            "loss_masks": [[0] for _ in range(batch_size)],
            "trajectory_ids": [None for _ in range(batch_size)]
        }
