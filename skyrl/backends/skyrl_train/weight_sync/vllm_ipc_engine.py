"""IPC-based weight transfer engine using CUDA IPC for communication.

Backported from upstream vLLM (which added IPC support after 0.16).
Registered with vLLM's WeightTransferEngineFactory via register_ipc_engine()
so that WeightTransferConfig(backend="ipc") works on vLLM 0.16.
"""

import base64
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from vllm import envs
from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for IPC weight transfer backend. No init needed for IPC."""

    pass


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for IPC weight transfer backend.

    Accepts IPC handles either directly via ``ipc_handles`` (Ray transport)
    or as a base64-encoded pickle via ``ipc_handles_pickled`` (HTTP transport).
    Exactly one of the two must be provided; if ``ipc_handles_pickled`` is set
    it is unpickled into ``ipc_handles`` during ``__post_init__``.
    """

    names: List[str] = None  # type: ignore[assignment]
    dtype_names: List[str] = None  # type: ignore[assignment]
    shapes: List[List[int]] = None  # type: ignore[assignment]
    ipc_handles: Optional[List[Dict[str, Tuple[Callable, tuple]]]] = None
    ipc_handles_pickled: Optional[str] = None

    def __post_init__(self):
        if self.ipc_handles_pickled is not None:
            if self.ipc_handles is not None:
                raise ValueError("Cannot specify both `ipc_handles` and `ipc_handles_pickled`")

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without " "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            self.ipc_handles = pickle.loads(base64.b64decode(self.ipc_handles_pickled))
            self.ipc_handles_pickled = None

        if self.ipc_handles is None:
            raise ValueError("Either `ipc_handles` or `ipc_handles_pickled` must be provided")

        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: " f"got {len(self.shapes)} and {len(self.names)}"
            )
        if len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


class IPCWeightTransferEngine(WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]):
    """Weight transfer engine using CUDA IPC for communication.

    Uses CUDA IPC handles to share GPU memory between trainer and inference
    workers on the same node.
    """

    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
        super().__init__(config, parallel_config)

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """No initialization needed for IPC backend."""
        pass

    def receive_weights(
        self,
        update_info: IPCWeightTransferUpdateInfo,
        load_weights: Callable[[list], None],
    ) -> None:
        """Receive weights from the trainer via CUDA IPC handles.

        Each IPC handle maps a physical GPU UUID to a (func, args) tuple that
        reconstructs the tensor.  Index 6 in the args tuple is the device_index
        which must be patched to the current device (logical index may differ
        between sender and receiver).
        """
        assert update_info.ipc_handles is not None
        weights = []
        for name, _dtype_name, _shape, ipc_handle in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            update_info.ipc_handles,
        ):
            device_index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device_index)
            physical_gpu_id = str(props.uuid)

            if physical_gpu_id not in ipc_handle:
                raise ValueError(
                    f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                    f"Available UUIDs: {list(ipc_handle.keys())}"
                )

            handle = ipc_handle[physical_gpu_id]
            func, args = handle
            list_args = list(args)  # type: ignore
            list_args[6] = device_index
            weight = func(*list_args)  # type: ignore
            weights.append((name, weight))

        load_weights(weights)

    def shutdown(self) -> None:
        pass


# TODO (aaron): remove this once WeightTransferConfig accepts arbitrary backends
def _patch_weight_transfer_config() -> None:
    """Extend vLLM's WeightTransferConfig to accept "ipc" as a backend.

    vLLM 0.16 only has backend: Literal["nccl"]. We monkey-patch __init__
    so that backend="ipc" is accepted by constructing with "nccl" (passes
    pydantic validation) then overriding the attribute directly.
    """
    from vllm.config.weight_transfer import WeightTransferConfig

    if getattr(WeightTransferConfig, "_skyrl_patched", False):
        return

    _original_init = WeightTransferConfig.__init__

    def _patched_init(self, backend="nccl", **kwargs):
        if backend == "ipc":
            _original_init(self, backend="nccl", **kwargs)
            object.__setattr__(self, "backend", "ipc")
        else:
            _original_init(self, backend=backend, **kwargs)

    WeightTransferConfig.__init__ = _patched_init
    WeightTransferConfig._skyrl_patched = True


def register_ipc_engine() -> None:
    """Register the IPC engine with vLLM's WeightTransferEngineFactory.

    Safe to call multiple times; silently skips if already registered.
    Also patches WeightTransferConfig to accept "ipc" as a valid backend.
    """
    _patch_weight_transfer_config()

    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    if "ipc" not in WeightTransferEngineFactory._registry:
        WeightTransferEngineFactory.register_engine(
            "ipc",
            "skyrl.backends.skyrl_train.weight_sync.vllm_ipc_engine",
            "IPCWeightTransferEngine",
        )
