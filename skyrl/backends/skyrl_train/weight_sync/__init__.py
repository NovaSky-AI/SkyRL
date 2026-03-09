"""Weight synchronization abstractions for distributed RL training."""

from typing import Type

from .base import WeightChunk, WeightUpdateRequest, LoraLoadRequest
from .weight_extractor import WeightExtractor
from .weight_loader import WeightLoader
from .transfer_strategy import (
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
    WeightSyncInitInfo,
)
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferSender,
    BroadcastWeightTransferReceiver,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferSender,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightUpdateRequest,
)


def get_transfer_strategy_cls(weight_sync_backend: str, colocate_all: bool) -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Uses CUDA IPC when:
    - weight_sync_backend is "nccl"
    - colocate_all is True (training and inference on same nodes)

    Otherwise uses broadcast.

    Args:
        weight_sync_backend: The weight sync backend ("nccl" or other).
        colocate_all: Whether training and inference are colocated on same nodes.

    Returns:
        The strategy class (CudaIpcTransferStrategy or BroadcastTransferStrategy).
    """
    if weight_sync_backend == "nccl" and colocate_all:
        return CudaIpcTransferStrategy
    return BroadcastTransferStrategy


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightLoader",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightTransferReceiver",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "BroadcastWeightTransferReceiver",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "CudaIpcWeightTransferReceiver",
    "get_transfer_strategy_cls",
]
