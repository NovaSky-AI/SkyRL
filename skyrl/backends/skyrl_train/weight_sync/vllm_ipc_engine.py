"""IPC-based weight transfer engine registration.

vLLM 0.18.0+ natively supports the IPC backend for WeightTransferConfig.
This module provides backward-compatible registration for older vLLM versions
that only supported backend="nccl".
"""

from loguru import logger
from packaging import version


def register_ipc_engine() -> None:
    """Register the IPC engine with vLLM's WeightTransferEngineFactory.

    On vLLM 0.18.0+, IPC is natively supported and this is a no-op.
    On older versions, patches WeightTransferConfig and registers a
    backported IPC engine implementation.
    """
    import vllm

    if version.parse(vllm.__version__) >= version.parse("0.18.0"):
        logger.debug("vLLM >= 0.18.0 has native IPC weight transfer support; skipping SkyRL registration")
        return

    _register_legacy_ipc_engine()


def _register_legacy_ipc_engine() -> None:
    """Register backported IPC engine for vLLM < 0.18.0."""
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
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    if "ipc" in WeightTransferEngineFactory._registry:
        return

    if not getattr(WeightTransferConfig, "_skyrl_patched", False):
        _original_init = WeightTransferConfig.__init__

        def _patched_init(self, backend="nccl", **kwargs):
            if backend == "ipc":
                _original_init(self, backend="nccl", **kwargs)
                object.__setattr__(self, "backend", "ipc")
            else:
                _original_init(self, backend=backend, **kwargs)

        WeightTransferConfig.__init__ = _patched_init
        WeightTransferConfig._skyrl_patched = True

    @dataclass
    class IPCWeightTransferInitInfo(WeightTransferInitInfo):
        pass

    @dataclass
    class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
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
                        "Refusing to deserialize `ipc_handles_pickled` without VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                    )
                self.ipc_handles = pickle.loads(base64.b64decode(self.ipc_handles_pickled))
                self.ipc_handles_pickled = None

            if self.ipc_handles is None:
                raise ValueError("Either `ipc_handles` or `ipc_handles_pickled` must be provided")

            num_params = len(self.names)
            for field_name, field_val in [
                ("dtype_names", self.dtype_names),
                ("shapes", self.shapes),
                ("ipc_handles", self.ipc_handles),
            ]:
                if len(field_val) != num_params:
                    raise ValueError(
                        f"`{field_name}` should be of the same size as `names`: "
                        f"got {len(field_val)} and {num_params}"
                    )

    class IPCWeightTransferEngine(WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]):
        init_info_cls = IPCWeightTransferInitInfo
        update_info_cls = IPCWeightTransferUpdateInfo

        def __init__(self, config: WeightTransferConfig, parallel_config: ParallelConfig) -> None:
            super().__init__(config, parallel_config)

        def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
            pass

        def receive_weights(self, update_info: IPCWeightTransferUpdateInfo, load_weights: Callable[[list], None]):
            assert update_info.ipc_handles is not None
            weights = []
            for name, _dtype_name, _shape, ipc_handle in zip(
                update_info.names, update_info.dtype_names, update_info.shapes, update_info.ipc_handles
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

    WeightTransferEngineFactory.register_engine(
        "ipc",
        "skyrl.backends.skyrl_train.weight_sync.vllm_ipc_engine",
        "IPCWeightTransferEngine",
    )
