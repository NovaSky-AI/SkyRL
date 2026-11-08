"""Trainers that stream rollouts one group at a time and train on every mini-batch of them.

- :class:`StreamingRayPPOTrainerBase` holds the loop, the generation workers, the buffer, checkpointing
  with consumed UIDs, and the hooks a placement-specific subclass fills in.
- :class:`FullyAsyncRayPPOTrainer` is the disaggregated trainer: inference on its own GPUs, generation
  never stops, admission bounded by :class:`AsyncStalenessManager`.
- A colocated partial-rollout trainer (generation cut at a fraction of the mini-batch, resumed under
  the new weights) is the other planned subclass.
"""

from skyrl.train.streaming.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
    RolloutStat,
)
from skyrl.train.streaming.base import StreamingRayPPOTrainerBase
from skyrl.train.streaming.dataloader import StreamingDataloader
from skyrl.train.streaming.group import GeneratedOutputGroup
from skyrl.train.streaming.trainer import FullyAsyncRayPPOTrainer

__all__ = [
    "StreamingDataloader",
    "AsyncStalenessManager",
    "StreamingRayPPOTrainerBase",
    "FullyAsyncRayPPOTrainer",
    "GeneratedOutputGroup",
    "RolloutAdmissionController",
    "RolloutStat",
]
