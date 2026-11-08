"""Trainers that stream rollouts one group at a time and train on every mini-batch of them.

- :class:`BaseFullyAsyncTrainer` holds the loop, the generation workers, the buffer, checkpointing
  with consumed UIDs, and the hooks a placement-specific subclass fills in.
- :class:`FullyAsyncRayPPOTrainer` is the disaggregated trainer: inference on its own GPUs, admission
  bounded by :class:`AsyncStalenessManager`.
"""

from skyrl.train.fully_async.admission import (
    AsyncStalenessManager,
    RolloutAdmissionController,
    RolloutStat,
)
from skyrl.train.fully_async.base import BaseFullyAsyncTrainer
from skyrl.train.fully_async.dataloader import AsyncDataloader
from skyrl.train.fully_async.group import GeneratedOutputGroup
from skyrl.train.fully_async.trainer import FullyAsyncRayPPOTrainer

__all__ = [
    "AsyncDataloader",
    "AsyncStalenessManager",
    "BaseFullyAsyncTrainer",
    "FullyAsyncRayPPOTrainer",
    "GeneratedOutputGroup",
    "RolloutAdmissionController",
    "RolloutStat",
]
