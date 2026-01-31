"""Weight synchronization abstractions for distributed RL training."""

from typing import Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from skyrl_train.config import SkyRLConfig

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


def get_transfer_strategy_cls(cfg: "Union[SkyRLConfig, DictConfig]") -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Uses CUDA IPC when:
    - weight_sync_backend is "nccl"
    - colocate_all is True (training and inference on same nodes)

    Otherwise uses broadcast.

    Args:
        cfg: Configuration object containing generator and trainer settings.

    Returns:
        The strategy class (CudaIpcTransferStrategy or BroadcastTransferStrategy).
    """
    if cfg.generator.weight_sync_backend == "nccl" and cfg.trainer.placement.colocate_all:
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
